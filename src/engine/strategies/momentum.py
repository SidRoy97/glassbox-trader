"""momentum_12_1 — jansen ch4 momentum factor, long-only with a trend filter"""

import pandas as pd
from core.config import (MOMENTUM_LOOKBACK, MOMENTUM_SKIP, MOMENTUM_MIN_RETURN,
                         MOMENTUM_FULL_SCORE_RETURN, TREND_MA)
from engine.strategies.common import (enough_history, above_trend,
                                      flat_signal, latest_signal)

NAME = "momentum_12_1"
SOURCE = ("Jansen ch4 (momentum factor: 12-month return skipping the last "
          "month); trend filter from ML4T / Jansen ch5 risk management")


def score_series(df):
    # score > 0 means long candidate; the value is the 12-1 return itself,
    # so ranking across the universe is the cross-sectional momentum sort
    if not enough_history(df):
        return pd.Series(0.0, index=df.index if df is not None else [])
    close = df["close"]
    mom = close.shift(MOMENTUM_SKIP) / close.shift(MOMENTUM_LOOKBACK) - 1
    ok = (mom > MOMENTUM_MIN_RETURN) & above_trend(close, TREND_MA)
    score = mom.where(ok, 0.0) / MOMENTUM_FULL_SCORE_RETURN
    return score.clip(lower=0.0).fillna(0.0)


def signal(df):
    # reading the latest bar into a citable signal
    if not enough_history(df):
        return flat_signal(NAME, "insufficient history")
    scores = score_series(df)
    close = df["close"]
    mom = float(close.iloc[-1 - MOMENTUM_SKIP]
                / close.iloc[-1 - MOMENTUM_LOOKBACK] - 1)
    trend = bool(above_trend(close, TREND_MA).iloc[-1])

    def reason(_):
        return (f"12-1 momentum {mom:+.1%}, price "
                f"{'above' if trend else 'below'} {TREND_MA}dma")
    return latest_signal(NAME, scores, reason)
