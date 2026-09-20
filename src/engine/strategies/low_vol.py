"""low_vol_quality — jansen ch4 volatility premium: calm names still rising"""

import numpy as np
import pandas as pd
from core.config import (LOWVOL_WINDOW, LOWVOL_MAX_ANN_VOL, LOWVOL_TREND_MA,
                         BT_ANNUAL_DAYS)
from engine.strategies.common import (enough_history, above_trend,
                                      flat_signal, latest_signal)

NAME = "low_vol_quality"
SOURCE = ("Jansen ch4 (low-volatility / quality premia); ML4T 1.5 inverse-"
          "volatility sizing idea. price-based only: the 2016 fundamentals "
          "file is too stale to trust for quality")


def _ann_vol(close):
    return close.pct_change().rolling(LOWVOL_WINDOW).std() \
        * np.sqrt(BT_ANNUAL_DAYS)


def score_series(df):
    # lower realised vol scores higher, gated on a medium-term uptrend and a
    # positive window return so it never buys a quiet name drifting lower
    if not enough_history(df):
        return pd.Series(0.0, index=df.index if df is not None else [])
    close = df["close"]
    vol = _ann_vol(close)
    ret = close / close.shift(LOWVOL_WINDOW) - 1
    ok = (vol < LOWVOL_MAX_ANN_VOL) & (ret > 0) \
        & above_trend(close, LOWVOL_TREND_MA)
    score = ((LOWVOL_MAX_ANN_VOL - vol) / LOWVOL_MAX_ANN_VOL).where(ok, 0.0)
    return score.clip(lower=0.0, upper=1.0).fillna(0.0)


def signal(df):
    # reading the latest bar into a citable signal
    if not enough_history(df):
        return flat_signal(NAME, "insufficient history")
    scores = score_series(df)
    close = df["close"]
    vol = float(_ann_vol(close).iloc[-1])
    ret = float(close.iloc[-1] / close.iloc[-1 - LOWVOL_WINDOW] - 1)
    trend = bool(above_trend(close, LOWVOL_TREND_MA).iloc[-1])

    def reason(_):
        return (f"{LOWVOL_WINDOW}d realised vol {vol:.0%} "
                f"(max {LOWVOL_MAX_ANN_VOL:.0%}), {LOWVOL_WINDOW}d return "
                f"{ret:+.1%}, {'above' if trend else 'below'} "
                f"{LOWVOL_TREND_MA}dma")
    return latest_signal(NAME, scores, reason)
