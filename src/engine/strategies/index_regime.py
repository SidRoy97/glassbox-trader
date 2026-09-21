"""index_regime — hold a broad etf basket while each etf trends up, else cash"""

import pandas as pd
from core.config import INDEX_BASKET, TREND_MA
from engine.strategies.common import (enough_history, above_trend, sma,
                                      flat_signal, latest_signal)

NAME = "index_regime"
SOURCE = ("Trend-filtered index exposure (ML4T ch4 market factor; Faber 2007 "
          "'A Quantitative Approach to Tactical Asset Allocation'): own the "
          "market via broad etfs only while above the long trend average")


def _is_basket(df):
    # only the etf basket is eligible; single names always score zero here
    return df is not None and df.attrs.get("ticker") in INDEX_BASKET


def score_series(df):
    # score > 0 while the etf holds above its trend; strength is the distance
    # above trend so the steadiest uptrend ranks first when slots are scarce
    if not _is_basket(df) or not enough_history(df):
        return pd.Series(0.0, index=df.index if df is not None else [])
    close = df["close"]
    gap = close / sma(close, TREND_MA) - 1
    score = gap.where(above_trend(close, TREND_MA), 0.0) / 0.10
    return score.clip(lower=0.0, upper=1.0).fillna(0.0)


def signal(df):
    # reading the latest bar into a citable signal
    if not _is_basket(df):
        return flat_signal(NAME, "not in the index basket")
    if not enough_history(df):
        return flat_signal(NAME, "insufficient history")
    scores = score_series(df)
    close = df["close"]
    gap = float(close.iloc[-1] / sma(close, TREND_MA).iloc[-1] - 1)

    def reason(_):
        return (f"{df.attrs.get('ticker')} {'above' if gap > 0 else 'below'} "
                f"{TREND_MA}dma by {gap:+.1%}")
    return latest_signal(NAME, scores, reason)
