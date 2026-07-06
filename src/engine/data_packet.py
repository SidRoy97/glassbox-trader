"""assembling the grounded data packet every panel argues over"""

import json
from engine.memory import (get_recent_news, get_recent_decisions,
                           get_active_lessons, get_active_thesis,
                           get_market_context, validate_ticker,
                           get_open_position, get_ticker_stats)


def get_cnn_signal(ticker):
    # calling the local signal pipeline directly for the model prediction
    import numpy as np
    import torch
    from inference.predictors import load_seq_predictor
    from inference.live_features import build_live_frame, fill_missing_features
    seq = load_seq_predictor()
    if seq is None:
        return {"direction": "unavailable", "confidence": 0.0}
    df = build_live_frame(ticker)
    if df is None or df.empty:
        return {"direction": "unavailable", "confidence": 0.0}
    meta = seq["meta"]
    df = fill_missing_features(df, meta["feature_cols"], seq["scaler"])
    if len(df) < meta["window"]:
        return {"direction": "unavailable", "confidence": 0.0}
    win = df.iloc[-meta["window"]:][meta["feature_cols"]] \
        .values.astype("float32")
    win = seq["scaler"].transform(win)
    with torch.no_grad():
        out = seq["model"](torch.tensor(win).unsqueeze(0)).numpy().squeeze()
    probs = np.exp(out) / np.exp(out).sum()
    idx = int(probs.argmax())
    latest = df.iloc[-1]
    return {"direction": meta["classes"][idx],
            "confidence": round(float(probs[idx]), 4),
            "close": round(float(latest["close"]), 2),
            "rsi": round(float(latest["rsi"]), 1),
            "return_5d": round(float(latest["return_5d"]), 4),
            "return_10d": round(float(latest["return_10d"]), 4),
            "pct_vs_ma50": round(float(latest["close"] / latest["ma50"] - 1), 4),
            "vol_ratio": round(float(latest["vol_ratio"]), 2),
            "rel_to_sector": round(float(latest["rel_to_sector"]), 4)}


def build_packet(ticker, news_items):
    # combining signal, news, history, lessons, thesis, and context
    from engine.news_fetcher import fetch_next_earnings
    ticker = validate_ticker(ticker)
    sentiments = [n.get("sentiment") for n in news_items[:5]
                  if n.get("sentiment") is not None]
    packet = {
        "ticker": ticker,
        "cnn_signal": get_cnn_signal(ticker),
        "days_to_earnings": fetch_next_earnings(ticker),
        "news": [{"headline": n["headline"][:200],
                  "summary": (n.get("summary") or "")[:300],
                  "source": n.get("source", ""),
                  "sentiment": n.get("sentiment")} for n in news_items[:5]],
        "news_sentiment_avg": round(sum(sentiments) / len(sentiments), 3)
        if sentiments else None,
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
            + json.dumps(packet, indent=1, default=str)[:4500]
            + "\n=== END DATA PACKET ===")
