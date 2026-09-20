"""assembling the grounded data packet every panel argues over"""

# signal source: engine/strategies (rule-based) elected by ritter-utility
# backtests in engine/strategy_election; no trained classifier is consulted

import json
from engine.memory import (get_recent_news, get_recent_decisions,
                           get_active_lessons, get_active_thesis,
                           get_market_context, validate_ticker,
                           get_open_position, get_ticker_stats)


def _indicator_block(df):
    # the plain price facts judges used to read off the model signal
    from engine.strategies.common import rsi
    close = df["close"]
    vol = df["volume"] if "volume" in df.columns else None
    ma50 = float(close.rolling(50).mean().iloc[-1])
    return {"close": round(float(close.iloc[-1]), 2),
            "rsi": round(float(rsi(close).iloc[-1]), 1),
            "return_5d": round(float(close.iloc[-1] / close.iloc[-6] - 1), 4),
            "return_10d": round(float(close.iloc[-1] / close.iloc[-11] - 1), 4),
            "pct_vs_ma50": round(float(close.iloc[-1] / ma50 - 1), 4),
            "vol_ratio": round(float(vol.iloc[-1] / vol.rolling(20).mean()
                                     .iloc[-1]), 2) if vol is not None else None}


def get_strategy_signal(ticker, news_items=None):
    # reading the elected rule strategy on live bars, showing every
    # strategy's vote alongside it and the market regime that gates them all
    from core.config import STRATEGY_CASH, BARS_LOOKBACK_DAYS
    from engine.market_data import bars_for
    from engine.strategies import get_strategy, all_signals
    from engine.strategies.regime import market_regime
    from engine.strategies import news_gate
    from engine.strategy_election import get_strategy_champion
    champion = get_strategy_champion()
    df = bars_for(ticker, days=BARS_LOOKBACK_DAYS)
    if df is None or len(df) < 60:
        return {"model": champion, "direction": "unavailable", "score": 0.0,
                "reason": "no bars"}
    regime = market_regime()
    if champion == STRATEGY_CASH:
        sig = {"model": champion, "direction": "NO_TRADE", "score": 0.0,
               "reason": "no strategy earned positive utility in the last "
                         "election — sitting in cash"}
    else:
        sig = get_strategy(champion).signal(df)
    sig = news_gate.apply(sig, news_items or [])
    if not regime.get("risk_on") and sig["direction"] == "BUY":
        sig["regime_note"] = "risk-off regime — the gate will block BUYs"
    sig.update(_indicator_block(df))
    sig["all_strategies"] = all_signals(df)
    sig["market_regime"] = regime
    return sig


def _structure_block(ticker):
    # summarizing price structure as citable evidence on a guaranteed ohlc frame
    try:
        import pandas as pd
        import yfinance as yf
        from engine.ta_structure import technical_structure_block
        hist = yf.download(ticker.replace(".", "-"), period="1y",
                           auto_adjust=True, progress=False)
        if hist is None or hist.empty or len(hist) < 60:
            return None
        # flattening the multiindex newer yfinance returns for single tickers
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        df = hist.rename(columns=str.lower)[["open", "high", "low", "close"]]
        return technical_structure_block(df)
    except Exception as e:
        print(f"  [structure] {ticker} block unavailable: {e}")
        return None


def _overnight_gap(ticker):
    # measuring the move since the prior close using yahoo prepost prints
    try:
        import pandas as pd
        import yfinance as yf
        from datetime import date
        sym = ticker.replace(".", "-")
        daily = yf.download(sym, period="5d", auto_adjust=True,
                            progress=False)
        if daily is None or daily.empty:
            return None
        if isinstance(daily.columns, pd.MultiIndex):
            daily.columns = daily.columns.get_level_values(0)
        # anchoring on the last fully completed session before today
        daily = daily[pd.to_datetime(daily.index).date < date.today()]
        if daily.empty:
            return None
        prev_close = float(daily["Close"].iloc[-1])
        intra = yf.download(sym, period="1d", interval="1m", prepost=True,
                            progress=False)
        if intra is None or intra.empty:
            return None
        if isinstance(intra.columns, pd.MultiIndex):
            intra.columns = intra.columns.get_level_values(0)
        last = float(intra["Close"].dropna().iloc[-1])
        return round(last / prev_close - 1, 4)
    except Exception as e:
        print(f"  [gap] {ticker} unavailable: {e}")
        return None


_reliability_cache = None


def _reliability_block():
    # attaching evidence grades once per run, tolerating any failure
    global _reliability_cache
    if _reliability_cache is None:
        try:
            from engine.evidence_weights import evidence_reliability_block
            _reliability_cache = evidence_reliability_block()
        except Exception as e:
            print(f"  [packet] reliability block failed: {e}")
            _reliability_cache = {}
    return _reliability_cache or None


def _macro_block():
    # attaching market-wide headlines, tolerating any failure
    try:
        from engine.news_fetcher import fetch_macro_news
        from datetime import datetime, timezone
        items = fetch_macro_news()
        out = []
        for i in items:
            age = None
            try:
                ts = datetime.fromisoformat(i["published_at"])
                age = round((datetime.now(timezone.utc) - ts)
                            .total_seconds() / 3600, 1)
            except Exception:
                pass
            out.append({"headline": i["headline"],
                        "sentiment": i["sentiment"],
                        "age_hours": age})
        return out or None
    except Exception as e:
        print(f"  [packet] macro block failed: {e}")
        return None


def _congress_block_safe(ticker):
    # attaching congressional disclosures, tolerating any failure
    try:
        from engine.congress import congress_block
        return congress_block(ticker)
    except Exception as e:
        print(f"  [packet] congress block failed: {e}")
        return None


def _insider_block(ticker):
    # attaching recent insider filing evidence, tolerating any failure
    try:
        from engine.smart_money import insider_activity
        return insider_activity(ticker)
    except Exception as e:
        print(f"  [insider] block failed for {ticker}: {e}")
        return None


def _value_area_block(ticker):
    # DAILY value-area signal adapted from the transcript's VWAP/deviation-band
    # idea. true VWAP is intraday; on daily bars we approximate the "value area"
    # with a volume-weighted moving average (VWMA) and +/- deviation bands, then
    # report where price sits relative to it. this is an APPROXIMATION, not
    # intraday VWAP — named "value_area" to avoid implying otherwise. tolerant
    # of any failure so it never blocks the packet.
    try:
        import numpy as np
        import yfinance as yf
        sym = ticker.replace(".", "-")
        hist = yf.download(sym, period="3mo", auto_adjust=True, progress=False)
        if hist is None or len(hist) < 25:
            return None
        import pandas as pd
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        close = hist["Close"].astype(float).to_numpy().ravel()
        vol = hist["Volume"].astype(float).to_numpy().ravel()
        n = 20
        if len(close) < n or vol[-n:].sum() <= 0:
            return None
        # volume-weighted moving average over the last n days
        vwma = float(np.sum(close[-n:] * vol[-n:]) / np.sum(vol[-n:]))
        # deviation band = std of price around the vwma over the window
        band = float(np.std(close[-n:]))
        price = float(close[-1])
        if band <= 0:
            return None
        # z = how many bands price sits above/below the value area centre
        z = round((price - vwma) / band, 2)
        if z >= 1.0:
            zone = "above_value"       # extended above value — trend/overbought
        elif z <= -1.0:
            zone = "below_value"       # extended below value — trend/oversold
        else:
            zone = "in_value"          # trading inside the value area
        return {
            "vwma_20d": round(vwma, 2),
            "price_vs_value_z": z,
            "zone": zone,
            "note": ("daily VWMA approximation of value area, "
                     "not intraday VWAP"),
        }
    except Exception:
        return None


def build_packet(ticker, news_items):
    # combining signal, structure, news, history, lessons, thesis, and context
    from engine.news_fetcher import fetch_next_earnings
    ticker = validate_ticker(ticker)
    sentiments = [n.get("sentiment") for n in news_items[:5]
                  if n.get("sentiment") is not None]
    _gap = _overnight_gap(ticker)
    # a meaningful gap with no news catalyst is "gapping to air" — the
    # corpus flags these as fade-prone versus catalyst-backed gaps which
    # carry a real thesis; naming the combination lets judges weigh it
    # directly instead of inferring it across two separate blocks
    _has_catalyst = any(abs(s) > 0.15 for s in sentiments)
    packet = {
        "ticker": ticker,
        "technical_structure": _structure_block(ticker),
        "overnight_gap_pct": _gap,
        "overnight_gap_no_catalyst": bool(
            _gap is not None and abs(_gap) >= 1.5 and not _has_catalyst),
        "insider_activity": _insider_block(ticker),
        "congress_trading": _congress_block_safe(ticker),
        "evidence_reliability": _reliability_block(),
        "macro_news": _macro_block(),
        "days_to_earnings": fetch_next_earnings(ticker),
        "news": [{"headline": n["headline"][:200],
                  "summary": (n.get("summary") or "")[:300],
                  "source": n.get("source", ""),
                  "sentiment": n.get("sentiment")} for n in news_items[:5]],
        "news_sentiment_avg": round(sum(sentiments) / len(sentiments), 3)
        if sentiments else None,
        # the elected rule strategy, with every strategy's vote shown
        "strategy_signal": get_strategy_signal(ticker, news_items),
        "value_area": _value_area_block(ticker),
        "recent_decisions": get_recent_decisions(ticker, limit=5),
        "lessons": get_active_lessons(limit=8),
        "active_thesis": get_active_thesis(ticker),
        "open_position": get_open_position(ticker),
        "ticker_track_record": get_ticker_stats(ticker),
        "market_context": get_market_context()[:1000],
    }
    return packet


def packet_to_text(packet):
    # serializing the packet as clearly delimited json for prompts
    return ("=== DATA PACKET (the only permitted evidence) ===\n"
            + json.dumps(packet, indent=1, default=str)[:6000]
            + "\n=== END DATA PACKET ===")