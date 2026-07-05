"""running the morning decision loop, outcome scoring, and thesis review"""

import argparse
from datetime import datetime, timezone
from engine.news_fetcher import fetch_and_archive
from engine.data_packet import build_packet
from engine.protocol import decide
from engine.risk_gate import apply_gate
from engine.thesis import propose_thesis, review_theses
from engine.memory import (insert_decision, get_unscored_decisions,
                           score_decision, upsert_market_context,
                           validate_ticker)

WATCHLIST = ["AAPL", "MSFT", "GOOGL", "NVDA", "JPM"]


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
    return action


def run_daily():
    # looping the full watchlist each morning before market open
    started = datetime.now(timezone.utc).isoformat()
    results = {}
    for ticker in WATCHLIST:
        try:
            results[ticker] = run_ticker(ticker)
        except Exception as e:
            print(f"{ticker} failed: {e}")
            results[ticker] = "ERROR"
    upsert_market_context(f"run at {started}: {results}")
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


def weekly_review():
    # scoring outcomes, reviewing theses, proposing new ones
    score_outcomes()
    prices = latest_prices(WATCHLIST)
    review_theses(lambda t: (prices.get(t) or {}).get("ret_5d"))
    for ticker in WATCHLIST:
        proposed = propose_thesis(ticker)
        if proposed:
            print(f"new thesis for {ticker}: {proposed['thesis_text'][:80]}")
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
    else:
        weekly_review()


if __name__ == "__main__":
    main()