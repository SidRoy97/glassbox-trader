"""loading S&P 500 price history from yfinance — recent data (2017-2026),
a drop-in replacement for the static Kaggle NYSE loader, returning the
identical (prices, fundamentals, securities) shape so every downstream
stage runs unchanged

the universe is fetched live from wikipedia so it stays current; history
comes from yfinance. fundamentals are synthesised (imputed downstream) since
yfinance fundamentals are too sparse for training
"""

import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from core.config import DATA_PATH, FUNDAMENTAL_COLS
from core.helpers import log, save_plot, section

# how many years of recent history to pull
YF_HISTORY_YEARS = int(os.environ.get("YF_HISTORY_YEARS", "10"))

PRICES_CACHE = os.path.join(DATA_PATH, "prices-split-adjusted.csv")
FUND_CACHE = os.path.join(DATA_PATH, "fundamentals.csv")
SEC_CACHE = os.path.join(DATA_PATH, "securities.csv")

WIKI_SP500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def fetch_sp500_constituents():
    # pulling the current S&P 500 list + GICS sectors from wikipedia; falling
    # back to a cached securities.csv if the fetch fails so a bad network call
    # never wipes the universe
    try:
        tables = pd.read_html(WIKI_SP500)
        df = tables[0]
        # wikipedia columns: Symbol, Security, GICS Sector, ...
        sym_col = next(c for c in df.columns if str(c).lower() == "symbol")
        sec_col = next(c for c in df.columns
                       if "gics sector" in str(c).lower())
        out = pd.DataFrame({
            "Ticker symbol": df[sym_col].astype(str).str.strip()
                             .str.replace(".", "-", regex=False),
            "GICS Sector": df[sec_col].astype(str).str.strip(),
        }).drop_duplicates("Ticker symbol").sort_values("Ticker symbol")
        if len(out) >= 400:
            out.to_csv(SEC_CACHE, index=False)
            return out
        raise RuntimeError(f"wikipedia list too short: {len(out)}")
    except Exception as e:
        log(f"S&P 500 wiki fetch failed ({e}); using cached securities.csv")
        if os.path.exists(SEC_CACHE):
            return pd.read_csv(SEC_CACHE)
        raise


def _normalize(df, ticker):
    # coercing a yfinance frame into the exact columns/dtypes downstream wants
    df = df.rename(columns={"Date": "date", "Open": "open", "High": "high",
                            "Low": "low", "Close": "close",
                            "Volume": "volume"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["symbol"] = ticker
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype("float32")
    return df[["date", "symbol", "open", "high", "low", "close", "volume"]]


def _download_prices(symbols):
    # batch-downloading daily OHLCV, retrying stragglers individually
    import yfinance as yf
    log(f"downloading {len(symbols)} S&P 500 tickers from yfinance "
        f"({YF_HISTORY_YEARS}y)...")
    raw = yf.download(symbols, period=f"{YF_HISTORY_YEARS}y",
                      auto_adjust=True, group_by="ticker",
                      progress=False, threads=True)
    frames, missing = [], []
    for t in symbols:
        try:
            df = raw[t].reset_index()
            if df.dropna(subset=["Close"]).empty:
                missing.append(t)
                continue
        except (KeyError, TypeError):
            missing.append(t)
            continue
        frames.append(_normalize(df, t))
    for t in missing:
        try:
            df = yf.Ticker(t).history(period=f"{YF_HISTORY_YEARS}y",
                                      auto_adjust=True).reset_index()
            if not df.dropna(subset=["Close"]).empty:
                frames.append(_normalize(df, t))
            time.sleep(0.3)
        except Exception:
            log(f"  could not fetch {t}; skipping")
    prices = pd.concat(frames, ignore_index=True)
    return prices.sort_values(["symbol", "date"]).reset_index(drop=True)


def _synth_fundamentals(prices):
    # neutral placeholder fundamentals; the pipeline imputes these downstream
    years = sorted(prices["date"].dt.year.unique())
    rows = []
    for t in prices["symbol"].unique():
        for y in years:
            rows.append({"Ticker Symbol": t, "Period Ending": f"{y}-12-31"})
    fund = pd.DataFrame(rows)
    for c in FUNDAMENTAL_COLS[2:]:
        fund[c] = pd.NA
    return fund[FUNDAMENTAL_COLS]


def stage_1_load():
    section("STAGE 1 — LOADING S&P 500 DATA (yfinance, recent)")

    securities = fetch_sp500_constituents()
    symbols = securities["Ticker symbol"].tolist()

    if os.path.exists(PRICES_CACHE):
        log("cached prices present, loading from disk")
        prices = pd.read_csv(PRICES_CACHE, parse_dates=["date"],
                             dtype={"open": "float32", "high": "float32",
                                    "low": "float32", "close": "float32",
                                    "volume": "float32"})
    else:
        prices = _download_prices(symbols)
        prices.to_csv(PRICES_CACHE, index=False)
        log(f"cached prices to {PRICES_CACHE}")

    fundamentals = _synth_fundamentals(prices)
    fundamentals.to_csv(FUND_CACHE, index=False)

    log(f"prices shape      : {prices.shape}")
    log(f"fundamentals shape: {fundamentals.shape}")
    log(f"securities shape  : {securities.shape}")
    log(f"date range        : {prices['date'].min().date()} "
        f"to {prices['date'].max().date()}")
    log(f"unique tickers    : {prices['symbol'].nunique()}")

    days = prices.groupby("symbol")["date"].count().sort_values()
    plt.figure(figsize=(12, 4))
    plt.plot(days.values)
    plt.title("trading days per ticker (S&P 500)")
    save_plot("s1_trading_days_per_ticker.png")

    plt.figure(figsize=(10, 4))
    sns.histplot(prices["close"].clip(upper=prices["close"].quantile(0.99)),
                 bins=100, kde=True)
    plt.title("distribution of closing prices (USD)")
    save_plot("s1_close_price_distribution.png")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, tk in zip(axes.flatten(), ["AAPL", "MSFT", "NVDA", "JPM"]):
        d = prices[prices["symbol"] == tk]
        if not d.empty:
            ax.plot(d["date"], d["close"])
            ax.set_title(tk)
    save_plot("s1_sample_ticker_prices.png")

    vol = prices.groupby("date")["volume"].sum()
    plt.figure(figsize=(12, 4))
    plt.plot(vol.index, vol.values)
    plt.title("total market volume over time")
    save_plot("s1_total_market_volume.png")

    log("stage 1 complete")
    return prices, fundamentals, securities
