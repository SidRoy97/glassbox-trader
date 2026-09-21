"""dual_momentum — Antonacci absolute + relative momentum, long only vs index"""

import pandas as pd
from core.config import (DUALMOM_LOOKBACK, DUALMOM_SKIP, BENCHMARK_TICKER,
                         MOMENTUM_FULL_SCORE_RETURN, TREND_MA)
from engine.strategies.common import (enough_history, above_trend,
                                      flat_signal, latest_signal)

NAME = "dual_momentum"
SOURCE = ("Gary Antonacci dual momentum: hold a name only when its own 12-1 "
          "return is positive (absolute) AND beats the market index over the "
          "same window (relative); rotates to safety otherwise")

_bench_cache = {}


def _benchmark_return():
    # the index's 12-1 return, loaded once per process; None if unavailable so
    # the strategy degrades to plain absolute momentum rather than crashing
    if "ret" in _bench_cache:
        return _bench_cache["ret"]
    try:
        from engine.market_data import bars_for
        b = bars_for(BENCHMARK_TICKER)
        close = b["close"]
        ret = float(close.iloc[-1 - DUALMOM_SKIP]
                    / close.iloc[-1 - DUALMOM_LOOKBACK] - 1)
    except Exception as e:
        print(f"[dual_momentum] benchmark unavailable ({e}); "
              f"using absolute momentum only")
        ret = None
    _bench_cache["ret"] = ret
    return ret


def _relative_hurdle(bench_ret):
    # the bar a name must clear: positive AND above the index return
    return max(0.0, bench_ret if bench_ret is not None else 0.0)


def score_series(df):
    # score > 0 when the name's own 12-1 return clears both the zero line and
    # the index's return over the same window, and price is above trend
    if not enough_history(df):
        return pd.Series(0.0, index=df.index if df is not None else [])
    close = df["close"]
    mom = close.shift(DUALMOM_SKIP) / close.shift(DUALMOM_LOOKBACK) - 1
    hurdle = _relative_hurdle(_benchmark_return())
    ok = (mom > hurdle) & above_trend(close, TREND_MA)
    # score by how far the name beats the index, not its raw return
    score = (mom - hurdle).where(ok, 0.0) / MOMENTUM_FULL_SCORE_RETURN
    return score.clip(lower=0.0, upper=1.0).fillna(0.0)


def signal(df):
    # reading the latest bar into a citable signal
    if not enough_history(df):
        return flat_signal(NAME, "insufficient history")
    scores = score_series(df)
    close = df["close"]
    mom = float(close.iloc[-1 - DUALMOM_SKIP]
                / close.iloc[-1 - DUALMOM_LOOKBACK] - 1)
    bench = _benchmark_return()
    trend = bool(above_trend(close, TREND_MA).iloc[-1])

    def reason(_):
        b = f"{bench:+.1%}" if bench is not None else "n/a"
        return (f"12-1 momentum {mom:+.1%} vs {BENCHMARK_TICKER} {b}, "
                f"price {'above' if trend else 'below'} {TREND_MA}dma")
    return latest_signal(NAME, scores, reason)
