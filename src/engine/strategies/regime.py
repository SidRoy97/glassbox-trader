"""market regime — risk-on only when the index trends up and vol is calm"""

import pandas as pd
from core.config import (REGIME_INDEX, REGIME_VOL_INDEX, REGIME_MA,
                         REGIME_VIX_MAX, BARS_LOOKBACK_DAYS)

SOURCE = ("ML4T ch4 CAPM market factor (beta dominates single-name returns); "
          "Jansen ch5 drawdown control — sit in cash when the market is not")

_cache = {}


def risk_on_series(index_close, vix_close):
    # a causal boolean series aligned to the index calendar
    trend = index_close > index_close.rolling(REGIME_MA).mean()
    vix = vix_close.reindex(index_close.index).ffill()
    calm = vix < REGIME_VIX_MAX
    return (trend & calm).fillna(False)


def market_regime(bars=None):
    # reading today's regime once per process; fails to risk-off on data error
    if "regime" in _cache:
        return _cache["regime"]
    try:
        from engine.market_data import download_bars
        bars = bars or download_bars([REGIME_INDEX, REGIME_VOL_INDEX],
                                     days=BARS_LOOKBACK_DAYS)
        idx = bars[REGIME_INDEX]["close"]
        vix = bars[REGIME_VOL_INDEX]["close"]
        series = risk_on_series(idx, vix)
        ma = float(idx.rolling(REGIME_MA).mean().iloc[-1])
        out = {"risk_on": bool(series.iloc[-1]),
               "index": REGIME_INDEX,
               "index_close": round(float(idx.iloc[-1]), 2),
               "index_ma": round(ma, 2),
               "vix": round(float(vix.iloc[-1]), 1),
               "vix_max": REGIME_VIX_MAX,
               "reason": (f"{REGIME_INDEX} {'above' if idx.iloc[-1] > ma else 'below'} "
                          f"{REGIME_MA}dma, VIX {float(vix.iloc[-1]):.1f} "
                          f"{'<' if vix.iloc[-1] < REGIME_VIX_MAX else '>='} "
                          f"{REGIME_VIX_MAX}")}
    except Exception as e:
        print(f"[regime] unavailable, defaulting to risk-off: {e}")
        out = {"risk_on": False, "index": REGIME_INDEX,
               "reason": f"regime data unavailable ({e})"}
    _cache["regime"] = out
    return out
