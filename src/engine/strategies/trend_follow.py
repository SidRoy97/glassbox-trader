"""trend_follow_ma — ml4t moving-average crossover, long only while trending up"""

import pandas as pd
from core.config import TREND_FAST_MA, TREND_SLOW_MA, TREND_FULL_SCORE_GAP
from engine.strategies.common import enough_history, sma, flat_signal, \
    latest_signal

NAME = "trend_follow_ma"
SOURCE = ("ML4T ch4 moving-average crossover; a classic trend-following rule "
          "that holds only while the fast average leads the slow one and sits "
          "out downtrends entirely")


def score_series(df):
    # score > 0 while the fast MA is above the slow MA; the gap between them
    # sets the strength, so a strong uptrend ranks above a marginal one
    if not enough_history(df):
        return pd.Series(0.0, index=df.index if df is not None else [])
    close = df["close"]
    fast, slow = sma(close, TREND_FAST_MA), sma(close, TREND_SLOW_MA)
    gap = (fast - slow) / slow
    score = gap.where(gap > 0, 0.0) / TREND_FULL_SCORE_GAP
    return score.clip(lower=0.0, upper=1.0).fillna(0.0)


def signal(df):
    # reading the latest bar into a citable signal
    if not enough_history(df):
        return flat_signal(NAME, "insufficient history")
    scores = score_series(df)
    close = df["close"]
    fast = float(sma(close, TREND_FAST_MA).iloc[-1])
    slow = float(sma(close, TREND_SLOW_MA).iloc[-1])
    gap = (fast - slow) / slow

    def reason(_):
        return (f"{TREND_FAST_MA}dma {'above' if gap > 0 else 'below'} "
                f"{TREND_SLOW_MA}dma by {gap:+.1%}")
    return latest_signal(NAME, scores, reason)
