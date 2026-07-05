"""assembling the grounded data packet every panel argues over"""

import json
from engine.memory import (get_recent_news, get_recent_decisions,
                           get_active_lessons, get_active_thesis,
                           get_market_context, validate_ticker)


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
    return {"direction": meta["classes"][idx],
            "confidence": round(float(probs[idx]), 4),
            "close": round(float(df["close"].iloc[-1]), 2),
            "rsi": round(float(df["rsi"].iloc[-1]), 1),
            "return_5d": round(float(df["return_5d"].iloc[-1]), 4),
            "rel_to_sector": round(float(df["rel_to_sector"].iloc[-1]), 4)}


def build_packet(ticker, news_items):
    # combining signal, news, history, lessons, thesis, and context
    ticker = validate_ticker(ticker)
    packet = {
        "ticker": ticker,
        "cnn_signal": get_cnn_signal(ticker),
        "news": [{"headline": n["headline"][:200],
                  "summary": (n.get("summary") or "")[:300],
                  "source": n.get("source", "")} for n in news_items[:5]],
        "recent_decisions": get_recent_decisions(ticker, limit=5),
        "lessons": get_active_lessons(limit=8),
        "active_thesis": get_active_thesis(ticker),
        "market_context": get_market_context()[:1000],
    }
    return packet


def packet_to_text(packet):
    # serializing the packet as clearly delimited json for prompts
    return ("=== DATA PACKET (the only permitted evidence) ===\n"
            + json.dumps(packet, indent=1, default=str)[:6000]
            + "\n=== END DATA PACKET ===")
