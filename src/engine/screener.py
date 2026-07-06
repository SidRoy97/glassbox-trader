"""scanning the whole universe with the cnn and ranking debate candidates"""

import numpy as np
from datetime import date, timedelta
from core.helpers import log, section
from pipeline.oos_evaluation import prepare_oos_frame
from inference.predictors import load_seq_predictor

SCAN_LOOKBACK_DAYS = 240     # downloading enough history for indicator warmup
FALLBACK = ["AAPL", "MSFT", "GOOGL", "NVDA", "JPM"]


def scan_universe(limit=None):
    # running batch cnn inference over every ticker and scoring interest
    import torch
    seq = load_seq_predictor()
    if seq is None:
        log("screener: no sequence model loaded")
        return []
    meta = seq["meta"]
    window, fcols = meta["window"], meta["feature_cols"]

    start = date.today() - timedelta(days=SCAN_LOOKBACK_DAYS)
    df = prepare_oos_frame(limit=limit, start=str(start),
                           end=str(date.today() + timedelta(days=1)))
    if df is None or df.empty:
        log("screener: universe download failed")
        return []

    # filling any dataset-only columns with scaler means so they scale to zero
    means = dict(zip(fcols, seq["scaler"].mean_))
    for c in fcols:
        df[c] = df[c].fillna(means[c]) if c in df.columns else means[c]

    # building one latest window per ticker for batch inference
    windows, info = [], []
    for tk, grp in df.groupby("symbol", sort=False):
        grp = grp.sort_values("date")
        if len(grp) < window + 1:
            continue
        feats = seq["scaler"].transform(grp[fcols]).astype("float32")
        windows.append(feats[-window:])
        last = grp.iloc[-1]
        info.append({"ticker": tk,
                     "close": round(float(last["close"]), 2),
                     "return_1d": float(last.get("return_1d") or 0),
                     "vol_ratio": float(last.get("vol_ratio") or 1),
                     "rsi": round(float(last.get("rsi") or 50), 1)})
    if not windows:
        return []

    X = np.stack(windows)
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), 2048):
            outs.append(seq["model"](torch.tensor(X[i:i + 2048])).numpy())
    out = np.concatenate(outs)
    probs = np.exp(out) / np.exp(out).sum(axis=1, keepdims=True)

    # scoring interest from conviction, abnormal move, and abnormal volume
    results = []
    for row, p in zip(info, probs):
        idx = int(p.argmax())
        direction = meta["classes"][idx]
        conf = float(p[idx])
        directional = 0.0 if direction == "Neutral" else (conf - 1 / 3)
        move = min(abs(row["return_1d"]) * 10, 0.5)
        volume = min(max(row["vol_ratio"] - 1, 0) * 0.2, 0.4)
        results.append({**row, "direction": direction,
                        "confidence": round(conf, 4),
                        "score": round(directional + move + volume, 4)})

    results.sort(key=lambda r: r["score"], reverse=True)
    log(f"screener: scanned {len(results)} tickers, "
        f"top: {[r['ticker'] for r in results[:5]]}")
    return results


def select_watchlist(k=5, limit=None, exclude=None):
    # returning the top-k tickers for debate, rotating past recent picks
    exclude = exclude or set()
    results = scan_universe(limit=limit)
    if not results:
        log("screener: falling back to the core watchlist")
        return FALLBACK, []
    eligible = [r["ticker"] for r in results if r["ticker"] not in exclude]
    return eligible[:int(k)], results
