"""mean_reversion_bb — ml4t 1.1.1 bollinger dip buying inside an uptrend"""

import numpy as np
import pandas as pd
from core.config import (MR_BB_WINDOW, MR_BB_STD, MR_ENTRY_PCTB, MR_EXIT_PCTB,
                         MR_RSI_MAX, MR_RSI_SPAN, MR_MAX_HOLD_DAYS, TREND_MA)
from engine.strategies.common import (enough_history, above_trend, rsi,
                                      bollinger_pctb, flat_signal,
                                      latest_signal)

NAME = "mean_reversion_bb"
SOURCE = ("ML4T 1.1.1 Bollinger Bands (%b) with RSI confirm; Jansen ch4 "
          "short-term reversal factor; trend filter keeps it long-only in "
          "uptrends")


def _entry_score(pctb, r):
    # deeper below the band and more oversold both raise the score
    band_part = max(0.0, (MR_ENTRY_PCTB - pctb) / max(MR_ENTRY_PCTB, 1e-9))
    rsi_part = max(0.0, (MR_RSI_MAX - r) / MR_RSI_MAX)
    return min(1.0, 0.5 + 0.25 * band_part + 0.25 * rsi_part)


def score_series(df):
    # a small state machine: enter on the dip, hold to the midline or the
    # time stop; the held score decays so fresher entries rank higher
    if not enough_history(df):
        return pd.Series(0.0, index=df.index if df is not None else [])
    close = df["close"]
    pctb = bollinger_pctb(close, MR_BB_WINDOW, MR_BB_STD).to_numpy()
    r = rsi(close, MR_RSI_SPAN).to_numpy()
    trend = above_trend(close, TREND_MA).to_numpy()
    out = np.zeros(len(close))
    held, entry_score = 0, 0.0
    for i in range(len(close)):
        if held > 0:
            held += 1
            if (np.isfinite(pctb[i]) and pctb[i] >= MR_EXIT_PCTB) \
                    or held > MR_MAX_HOLD_DAYS:
                held = 0
                continue
            out[i] = entry_score * (1 - held / (MR_MAX_HOLD_DAYS + 1))
            continue
        if np.isfinite(pctb[i]) and pctb[i] < MR_ENTRY_PCTB \
                and r[i] < MR_RSI_MAX and trend[i]:
            held = 1
            entry_score = _entry_score(pctb[i], r[i])
            out[i] = entry_score
    return pd.Series(out, index=close.index)


def signal(df):
    # reading the latest bar into a citable signal
    if not enough_history(df):
        return flat_signal(NAME, "insufficient history")
    scores = score_series(df)
    close = df["close"]
    pctb = float(bollinger_pctb(close, MR_BB_WINDOW, MR_BB_STD).iloc[-1])
    r = float(rsi(close, MR_RSI_SPAN).iloc[-1])
    trend = bool(above_trend(close, TREND_MA).iloc[-1])

    def reason(_):
        return (f"bollinger %b {pctb:.2f} (entry < {MR_ENTRY_PCTB}), "
                f"rsi {r:.0f} (max {MR_RSI_MAX:.0f}), "
                f"{'above' if trend else 'below'} {TREND_MA}dma")
    return latest_signal(NAME, scores, reason)
