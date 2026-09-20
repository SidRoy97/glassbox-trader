"""sharing causal indicator helpers across the rule strategies"""

import numpy as np
import pandas as pd
from core.config import STRATEGY_MIN_DAYS, TREND_MA


def enough_history(df):
    # refusing to read a frame shorter than the longest lookback
    return df is not None and len(df) >= STRATEGY_MIN_DAYS


def sma(series, window):
    return series.rolling(int(window)).mean()


def rsi(close, span=14):
    # wilder rsi via exponential smoothing, causal by construction
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / span, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / span, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def bollinger_pctb(close, window, n_std):
    # %b = where price sits inside the band: 0 lower band, 1 upper band
    mid = sma(close, window)
    std = close.rolling(int(window)).std()
    upper, lower = mid + n_std * std, mid - n_std * std
    width = (upper - lower).replace(0, np.nan)
    return (close - lower) / width


def above_trend(close, window=TREND_MA):
    # the shared long-term filter: only buy names above their trend average
    return close > sma(close, window)


def flat_signal(name, reason):
    # the standard no-trade reply every strategy returns
    return {"model": name, "direction": "NO_TRADE", "score": 0.0,
            "reason": reason}


def latest_signal(name, scores, reason_fn):
    # turning the last score into the packet-facing signal dict
    last = float(scores.iloc[-1]) if len(scores) else 0.0
    if not np.isfinite(last) or last <= 0:
        return flat_signal(name, reason_fn(last))
    return {"model": name, "direction": "BUY",
            "score": round(min(last, 1.0), 4), "reason": reason_fn(last)}
