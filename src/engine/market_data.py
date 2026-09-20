"""loading the universe and downloading daily bars in one shared place"""

import os
import pandas as pd
from core.config import (DATA_PATH, UNIVERSE_FILE, ETF_UNIVERSE, ETF_SECTOR,
                         BARS_LOOKBACK_DAYS)

BAR_COLS = ["open", "high", "low", "close", "volume"]


def _yf_symbol(ticker):
    # yfinance wants the share-class dash form (BRK.B -> BRK-B)
    return ticker.replace(".", "-")


def load_universe(include_etfs=True, limit=None):
    # returning [(ticker, sector)] from universe.csv plus the etf basket
    path = os.path.join(DATA_PATH, UNIVERSE_FILE)
    rows = []
    if os.path.exists(path):
        frame = pd.read_csv(path, usecols=["Ticker symbol", "GICS Sector"])
        rows = [(str(t).strip().upper(), str(s))
                for t, s in zip(frame["Ticker symbol"], frame["GICS Sector"])]
    else:
        print(f"[market_data] universe file missing at {path}")
    if limit:
        rows = rows[:int(limit)]
    if include_etfs:
        seen = {t for t, _ in rows}
        rows += [(t, ETF_SECTOR) for t in ETF_UNIVERSE if t not in seen]
    return rows


def _normalise(frame):
    # lowercasing columns and dropping bars with no close
    if frame is None or frame.empty:
        return None
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame = frame.rename(columns=str.lower)
    if "close" not in frame.columns:
        return None
    frame = frame.dropna(subset=["close"])
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    frame.index.name = "date"
    return frame[[c for c in BAR_COLS if c in frame.columns]].astype(float)


def download_bars(tickers, days=BARS_LOOKBACK_DAYS):
    # downloading one batch of daily bars, returning {ticker: frame}
    import yfinance as yf
    if not tickers:
        return {}
    symbols = {t: _yf_symbol(t) for t in tickers}
    raw = yf.download(list(symbols.values()), period=f"{int(days)}d",
                      group_by="ticker", auto_adjust=True, progress=False,
                      threads=True)
    out = {}
    single = len(symbols) == 1
    for ticker, sym in symbols.items():
        try:
            frame = raw if single else raw[sym]
        except KeyError:
            print(f"[market_data] {ticker}: not in download")
            continue
        norm = _normalise(frame.copy())
        if norm is not None and not norm.empty:
            out[ticker] = norm
    print(f"[market_data] downloaded {len(out)}/{len(tickers)} tickers "
          f"({days}d)")
    return out


def bars_for(ticker, days=BARS_LOOKBACK_DAYS):
    # convenience wrapper for one ticker, None when unavailable
    return download_bars([ticker], days=days).get(ticker)
