"""running the morning decision loop, outcome scoring, and thesis review"""

import os
import argparse
from datetime import datetime, timezone
from engine.news_fetcher import fetch_and_archive
from engine.data_packet import build_packet
from engine.protocol import decide
from engine.risk_gate import apply_gate
from engine.thesis import propose_thesis, review_theses
from engine.screener import select_watchlist
from engine.shadow import (record_predictions,
                           score_model_predictions, model_report)
from engine.execution import (maybe_enter, maybe_exit,
                              sync_positions_table, paper_report, enabled)
from engine.memory import (insert_decision, get_unscored_decisions,
                           score_decision, upsert_market_context,
                           validate_ticker, save_screen_results,
                           get_recent_tickers, prune_news)

DEBATE_BUDGET = int(os.environ.get("DEBATE_BUDGET", "10"))
DEBATE_COOLDOWN_DAYS = int(os.environ.get("DEBATE_COOLDOWN_DAYS", "2"))
SCAN_LIMIT = os.environ.get("SCAN_LIMIT")   # optional ticker cap for testing
WATCHLIST = ["AAPL", "MSFT", "GOOGL", "NVDA", "JPM"]   # fallback only


def run_ticker(ticker):
    # deciding one ticker: news, packet, debate, gate, record
    ticker = validate_ticker(ticker)
    print(f"\n--- {ticker} ---")
    news = fetch_and_archive(ticker)
    packet = build_packet(ticker, news)
    verdict = decide(packet)
    action, note = apply_gate(ticker, verdict)
    print(f"panel: {verdict['decision']} | gate: {action} | {note}")

    sig = packet["cnn_signal"]
    insert_decision(ticker, action, sig.get("direction", "unavailable"),
                    sig.get("confidence", 0.0), verdict["bull_case"],
                    verdict["bear_case"], verdict["judge_votes"], note)

    # executing on paper only when the flag and keys are present
    if enabled():
        try:
            if action == "BUY":
                maybe_enter(ticker)
            elif action == "SELL":
                maybe_exit(ticker)
        except Exception as e:
            print(f"  [paper] {ticker} execution failed: {e}")
    return action


def market_summary():
    # composing a factual market snapshot from index and volatility data
    import yfinance as yf
    parts = []
    for name, sym in [("S&P 500", "SPY"), ("Nasdaq 100", "QQQ")]:
        try:
            closes = yf.download(sym, period="10d", auto_adjust=True,
                                 progress=False)["Close"].squeeze()
            d1 = closes.pct_change().iloc[-1] * 100
            d5 = closes.pct_change(5).iloc[-1] * 100
            parts.append(f"{name} {d1:+.1f}% last session, {d5:+.1f}% over 5 days")
        except Exception:
            continue
    try:
        vix = float(yf.download("^VIX", period="5d", auto_adjust=True,
                                progress=False)["Close"].squeeze().iloc[-1])
        mood = "calm" if vix < 15 else "elevated" if vix < 25 else "stressed"
        parts.append(f"VIX at {vix:.1f} ({mood})")
    except Exception:
        pass
    return "; ".join(parts) if parts else "market data unavailable"


def run_daily():
    # scanning the universe, then debating only the most interesting tickers
    upsert_market_context(market_summary())

    limit = int(SCAN_LIMIT) if SCAN_LIMIT else None
    recently_debated = set(get_recent_tickers(days=DEBATE_COOLDOWN_DAYS))
    watchlist, scan = select_watchlist(k=DEBATE_BUDGET, limit=limit,
                                       exclude=recently_debated)
    if scan:
        save_screen_results(scan)
    print(f"debating today: {watchlist} "
          f"(excluded {len(recently_debated)} on cooldown)")

    results = {}
    for ticker in watchlist:
        try:
            results[ticker] = run_ticker(ticker)
            record_predictions(ticker)
        except Exception as e:
            print(f"{ticker} failed: {e}")
            results[ticker] = "ERROR"
    if enabled():
        try:
            sync_positions_table()
        except Exception as e:
            print(f"[paper] position sync failed: {e}")
    print(f"\ndaily run complete: {results}")


def latest_prices(tickers):
    # fetching latest closes and returns for scoring and thesis review
    import yfinance as yf
    out = {}
    for t in tickers:
        try:
            hist = yf.download(t.replace(".", "-"), period="10d",
                               auto_adjust=True, progress=False)
            closes = hist["Close"].squeeze()
            out[t] = {"close": float(closes.iloc[-1]),
                      "ret_1d": float(closes.pct_change().iloc[-1]),
                      "ret_5d": float(closes.pct_change(5).iloc[-1])
                      if len(closes) > 5 else None}
        except Exception:
            out[t] = None
    return out


def score_outcomes():
    # scoring each decision against the first trading day after it was made
    import pandas as pd
    import yfinance as yf
    pending = get_unscored_decisions()
    if not pending:
        print("nothing to score")
        return
    for d in pending:
        try:
            decided = pd.Timestamp(d["decided_at"]).tz_localize(None)
            hist = yf.download(d["ticker"].replace(".", "-"), period="1mo",
                               auto_adjust=True, progress=False)
            closes = hist["Close"].squeeze()
            closes.index = pd.to_datetime(closes.index).tz_localize(None)

            # locating the last close at or before the decision moment
            before = closes[closes.index <= decided]
            after = closes[closes.index > decided]
            if before.empty or after.empty:
                print(f"skipping {d['ticker']} #{d['id']}: "
                      f"next trading day not complete yet")
                continue

            base = float(before.iloc[-1])
            ret_1d = float(after.iloc[0]) / base - 1
            ret_5d = (float(after.iloc[4]) / base - 1) if len(after) > 4 else None

            label = ("Up" if ret_1d > 0.01
                     else "Down" if ret_1d < -0.01 else "Neutral")
            correct = (d["action"] == "BUY" and label == "Up") or \
                      (d["action"] == "SELL" and label == "Down") or \
                      (d["action"] == "NO_TRADE" and label == "Neutral")
            score_decision(d["id"], ret_1d, ret_5d, label, correct)
            print(f"scored {d['ticker']} #{d['id']}: {label} "
                  f"({'correct' if correct else 'wrong'})")
        except Exception as e:
            print(f"skipping {d['ticker']} #{d['id']}: {e}")


def performance_report(window=60):
    # summarising panel accuracy and cnn drift over recent scored decisions
    from engine.memory import get_client
    rows = get_client().table("decisions") \
        .select("action,was_correct,cnn_direction,outcome_label") \
        .not_.is_("scored_at", "null") \
        .order("decided_at", desc=True).limit(int(window)).execute().data or []
    if not rows:
        print("performance: no scored decisions yet")
        return
    trades = [r for r in rows if r["action"] != "NO_TRADE"]
    holds = [r for r in rows if r["action"] == "NO_TRADE"]
    cnn_hits = sum(1 for r in rows if r["cnn_direction"] == r["outcome_label"])
    print(f"performance (last {len(rows)} scored):")
    print(f"  trades correct : "
          f"{sum(1 for r in trades if r['was_correct'])}/{len(trades)}")
    print(f"  holds correct  : "
          f"{sum(1 for r in holds if r['was_correct'])}/{len(holds)} "
          f"(missed moves: {sum(1 for r in holds if not r['was_correct'])})")
    print(f"  cnn hit rate   : {cnn_hits}/{len(rows)} "
          f"(random baseline ~{len(rows)//3}) — retrain when this sags")


def weekly_review():
    # scoring outcomes, reporting performance, reviewing recent tickers
    score_outcomes()
    score_model_predictions()
    performance_report()
    model_report()
    paper_report()
    tickers = get_recent_tickers(days=30) or WATCHLIST
    prices = latest_prices(tickers)
    review_theses(lambda t: (prices.get(t) or {}).get("ret_5d"))
    for ticker in tickers:
        proposed = propose_thesis(ticker)
        if proposed:
            print(f"new thesis for {ticker}: {proposed['thesis_text'][:80]}")
    prune_news(years=5)
    print("weekly review complete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="daily",
                        choices=["daily", "score", "weekly"])
    args = parser.parse_args()
    if args.mode == "daily":
        run_daily()
    elif args.mode == "score":
        score_outcomes()
        score_model_predictions()
    else:
        weekly_review()


if __name__ == "__main__":
    main()
