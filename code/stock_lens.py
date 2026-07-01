"""
stock_lens.py
=============
Complete stock-lens pipeline in a single file.

Stages (each a function, run in order):
    stage_1_load()          load and inspect raw NYSE data
    stage_2_features()      technical indicators, fundamentals, labels, split
    stage_2b_enhanced()     lagged / return / relative features + multi-horizon labels
    stage_3_classify()      baseline classification (pooled, single horizon)
    stage_3b_experiments()  full experiment: 5 strategies x 3 horizons, keep what helps

Usage:
    python stock_lens.py --stage all
    python stock_lens.py --stage 3b
    STOCK_LENS_BASE=/workspace/stock-lens python stock_lens.py --stage all

All plots are saved to <BASE>/observations/ with descriptive names.
All printed observations are also appended to <BASE>/observations/run_log.txt.
"""

import os
import gc
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                     # rendering to file without a display
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 60)
pd.set_option("display.float_format", "{:.4f}".format)


# ===========================================================================
# CONFIG
# ===========================================================================
BASE_PATH = os.environ.get("STOCK_LENS_BASE", os.path.abspath("./stock-lens"))
DATA_PATH = os.path.join(BASE_PATH, "data")
MODEL_PATH = os.path.join(BASE_PATH, "models")
OBS_PATH = os.path.join(BASE_PATH, "observations")
for _p in (DATA_PATH, MODEL_PATH, OBS_PATH):
    os.makedirs(_p, exist_ok=True)

KAGGLE_DATASET = "dgawlik/nyse"
KAGGLE_FILES = ["prices-split-adjusted.csv", "fundamentals.csv",
                "securities.csv", "prices.csv"]

FUNDAMENTAL_COLS = ["Ticker Symbol", "Period Ending", "Earnings Per Share",
                    "Total Revenue", "Net Income", "Total Assets",
                    "Total Liabilities", "Profit Margin", "Total Equity",
                    "Operating Margin", "Current Ratio"]
FUND_FILL_COLS = ["Earnings Per Share", "Total Revenue", "Net Income",
                  "Total Assets", "Total Liabilities", "Profit Margin",
                  "Total Equity", "Operating Margin", "Current Ratio"]

BASE_FEATURE_COLS = ["open", "high", "low", "close", "volume",
                     "ma10", "ma30", "ma50", "rsi", "vol_ratio",
                     "MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9",
                     "BBU_20_2.0_2.0", "BBL_20_2.0_2.0", "BBM_20_2.0_2.0",
                     "Earnings Per Share", "Total Revenue", "Net Income",
                     "Total Assets", "Total Liabilities", "Profit Margin",
                     "Total Equity", "Operating Margin", "Current Ratio"]

LAG_COLS = ["close", "rsi", "MACD_12_26_9", "MACDh_12_26_9", "vol_ratio"]
LAG_DAYS = [1, 2, 3, 5]
RETURN_FEATURES = ["return_1d", "return_3d", "return_5d", "return_10d"]
RELATIVE_FEATURES = ["market_return", "rel_to_market", "sector_return", "rel_to_sector"]

HORIZONS = {"1d": {"days": 1, "threshold": 0.01},
            "5d": {"days": 5, "threshold": 0.02},
            "10d": {"days": 10, "threshold": 0.03}}

TRAIN_END = "2015-01-01"
VAL_END = "2016-01-01"
RANDOM_STATE = 42


# ===========================================================================
# HELPERS
# ===========================================================================
def log(msg):
    # printing to console and appending to a persistent run log
    print(msg)
    with open(os.path.join(OBS_PATH, "run_log.txt"), "a") as f:
        f.write(str(msg) + "\n")


def save_plot(name):
    # saving the current figure to the observations folder with a clear name
    path = os.path.join(OBS_PATH, name)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    log(f"  [plot saved] {name}")


def section(title):
    # printing a visible banner so long logs stay readable
    bar = "=" * 70
    log(f"\n{bar}\n{title}\n{bar}")


# ===========================================================================
# STAGE 1 — LOAD AND INSPECT
# ===========================================================================
def stage_1_load():
    section("STAGE 1 — LOADING AND INSPECTING RAW DATA")

    # downloading from kaggle only when files are missing
    if not all(os.path.exists(os.path.join(DATA_PATH, f)) for f in KAGGLE_FILES):
        log("downloading dataset from kaggle...")
        os.system("pip install kaggle -q")
        os.system(f"kaggle datasets download -d {KAGGLE_DATASET} -p {DATA_PATH} --unzip")
    else:
        log("data files already present")

    # loading the three source tables with memory-friendly dtypes
    prices = pd.read_csv(os.path.join(DATA_PATH, "prices-split-adjusted.csv"),
                         parse_dates=["date"],
                         dtype={"open": "float32", "high": "float32", "low": "float32",
                                "close": "float32", "volume": "float32"})
    fundamentals = pd.read_csv(os.path.join(DATA_PATH, "fundamentals.csv"),
                               usecols=FUNDAMENTAL_COLS)
    securities = pd.read_csv(os.path.join(DATA_PATH, "securities.csv"),
                             usecols=["Ticker symbol", "GICS Sector"])

    prices = prices.sort_values(["symbol", "date"]).reset_index(drop=True)

    # printing core shape and coverage observations
    log(f"prices shape      : {prices.shape}")
    log(f"fundamentals shape: {fundamentals.shape}")
    log(f"securities shape  : {securities.shape}")
    log(f"date range        : {prices['date'].min().date()} to {prices['date'].max().date()}")
    log(f"unique tickers    : {prices['symbol'].nunique()}")
    log(f"null counts in prices:\n{prices.isnull().sum()}")

    # plotting trading days per ticker to spot incomplete histories
    days_per_ticker = prices.groupby("symbol")["date"].count().sort_values()
    plt.figure(figsize=(12, 4))
    plt.plot(days_per_ticker.values)
    plt.title("trading days per ticker")
    plt.xlabel("ticker rank")
    plt.ylabel("number of trading days")
    save_plot("s1_trading_days_per_ticker.png")
    log(f"tickers with < 1000 trading days: {(days_per_ticker < 1000).sum()}")

    # plotting the distribution of closing prices across everything
    plt.figure(figsize=(10, 4))
    sns.histplot(prices["close"], bins=100, kde=True)
    plt.title("distribution of closing prices (all tickers, all dates)")
    plt.xlabel("closing price ($)")
    save_plot("s1_close_price_distribution.png")

    # plotting sample tickers to sanity check the raw series
    sample = ["AAPL", "GOOGL", "JPM", "XOM"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, tk in zip(axes.flatten(), sample):
        d = prices[prices["symbol"] == tk]
        ax.plot(d["date"], d["close"])
        ax.set_title(tk)
    plt.suptitle("closing price over time — sample tickers")
    save_plot("s1_sample_ticker_prices.png")

    # plotting total market volume over time to catch anomalies
    daily_volume = prices.groupby("date")["volume"].sum()
    plt.figure(figsize=(12, 4))
    plt.plot(daily_volume.index, daily_volume.values)
    plt.title("total market volume over time")
    save_plot("s1_total_market_volume.png")

    log("stage 1 complete")
    return prices, fundamentals, securities


# ===========================================================================
# STAGE 2 — FEATURES, FUNDAMENTALS, LABELS, SPLIT
# ===========================================================================
def _add_indicators(group):
    # computing indicators for a single ticker so nothing bleeds across stocks
    import pandas_ta as ta
    group = group.copy()
    group["ma10"] = group["close"].rolling(10).mean()
    group["ma30"] = group["close"].rolling(30).mean()
    group["ma50"] = group["close"].rolling(50).mean()
    group["rsi"] = ta.rsi(group["close"], length=14)
    group["vol_ratio"] = group["volume"] / group["volume"].rolling(20).mean()

    # computing macd and assigning each output column directly by aligned index
    macd = ta.macd(group["close"], fast=12, slow=26, signal=9)
    for col in macd.columns:
        group[col] = macd[col].values

    # computing bollinger bands and assigning each column directly by aligned index
    bb = ta.bbands(group["close"], length=20, std=2)
    for col in bb.columns:
        group[col] = bb[col].values

    return group


def stage_2_features(prices=None, fundamentals=None, securities=None):
    section("STAGE 2 — FEATURE ENGINEERING, LABELS, SPLIT")
    os.system("pip install pandas_ta -q")

    # reloading from disk if called standalone
    if prices is None:
        prices, fundamentals, securities = stage_1_load()

    # merging sector info onto every price row
    sector_info = securities.rename(columns={"Ticker symbol": "symbol",
                                             "GICS Sector": "sector"})
    prices = prices.merge(sector_info, on="symbol", how="left")
    log(f"sector null count after merge: {prices['sector'].isnull().sum()}")

    # plotting how many tickers sit in each sector
    sc = prices.groupby("sector")["symbol"].nunique().sort_values()
    plt.figure(figsize=(10, 5))
    sns.barplot(x=sc.values, y=sc.index, hue=sc.index, palette="Blues_r", legend=False)
    plt.title("number of unique tickers per sector")
    save_plot("s2_tickers_per_sector.png")

    # computing technical indicators ticker by ticker
    log("computing technical indicators...")
    prices = prices.groupby("symbol", group_keys=False).apply(_add_indicators)
    log(f"shape after indicators: {prices.shape}")

    # forward-filling annual fundamentals onto daily rows by year
    fundamentals = fundamentals.rename(columns={"Ticker Symbol": "symbol"})
    fundamentals["Period Ending"] = pd.to_datetime(fundamentals["Period Ending"])
    fundamentals["year"] = fundamentals["Period Ending"].dt.year
    prices["year"] = prices["date"].dt.year
    prices = prices.merge(fundamentals.drop(columns=["Period Ending"]),
                          on=["symbol", "year"], how="left").drop(columns=["year"])
    log(f"shape after fundamentals merge: {prices.shape}")

    # forward-filling fundamental gaps within each ticker
    prices = prices.sort_values(["symbol", "date"])
    prices[FUND_FILL_COLS] = prices.groupby("symbol")[FUND_FILL_COLS].ffill()

    # dropping warmup rows and remaining indicator nulls
    prices = prices.groupby("symbol", group_keys=False).apply(
        lambda x: x.iloc[50:]).reset_index(drop=True)
    prices = prices.dropna(subset=["ma50", "rsi", "vol_ratio",
                                   "MACD_12_26_9", "BBU_20_2.0_2.0"])
    log(f"shape after cleanup: {prices.shape}")

    # creating the next-day label with the +/-1% threshold
    def make_label(pct):
        if pct > 0.01:
            return "Up"
        if pct < -0.01:
            return "Down"
        return "Neutral"

    prices["next_day_return"] = prices.groupby("symbol")["close"].pct_change().shift(-1)
    prices["label"] = prices["next_day_return"].apply(make_label)
    prices = prices.dropna(subset=["next_day_return"])

    # plotting label balance
    lc = prices["label"].value_counts()
    plt.figure(figsize=(7, 4))
    sns.barplot(x=lc.index, y=lc.values, hue=lc.index,
                palette=["tomato", "steelblue", "seagreen"], legend=False)
    plt.title("label distribution (next-day, +/-1%)")
    save_plot("s2_label_distribution.png")
    log(f"label distribution:\n{prices['label'].value_counts(normalize=True).mul(100).round(1)}")

    # plotting a full indicator panel for AAPL to verify correctness
    _plot_indicator_panel(prices, "AAPL", "s2_AAPL_indicator_panel.png")

    # plotting a feature correlation heatmap
    corr_cols = ["close", "ma10", "ma30", "ma50", "rsi", "vol_ratio",
                 "MACD_12_26_9", "MACDh_12_26_9", "BBU_20_2.0_2.0", "BBL_20_2.0_2.0",
                 "Earnings Per Share", "Profit Margin", "Current Ratio"]
    plt.figure(figsize=(12, 9))
    sns.heatmap(prices[corr_cols].corr(), annot=True, fmt=".2f",
                cmap="coolwarm", center=0, linewidths=0.5)
    plt.title("feature correlation heatmap")
    save_plot("s2_feature_correlation.png")

    # splitting chronologically and saving
    train = prices[prices["date"] < TRAIN_END]
    val = prices[(prices["date"] >= TRAIN_END) & (prices["date"] < VAL_END)]
    test = prices[prices["date"] >= VAL_END]
    log(f"train: {train.shape} | val: {val.shape} | test: {test.shape}")

    prices.to_csv(os.path.join(DATA_PATH, "master.csv"), index=False)
    train.to_csv(os.path.join(DATA_PATH, "train.csv"), index=False)
    val.to_csv(os.path.join(DATA_PATH, "val.csv"), index=False)
    test.to_csv(os.path.join(DATA_PATH, "test.csv"), index=False)
    log("saved master/train/val/test csv")
    log("stage 2 complete")
    return prices


def _plot_indicator_panel(df, ticker, filename):
    # plotting price+MAs, RSI, MACD, Bollinger for one ticker in a 4-row panel
    d = df[df["symbol"] == ticker].copy()
    if d.empty:
        return
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    axes[0].plot(d["date"], d["close"], label="close", linewidth=1)
    for ma in ["ma10", "ma30", "ma50"]:
        axes[0].plot(d["date"], d[ma], label=ma.upper(), linestyle="--")
    axes[0].set_title(f"{ticker} — close + moving averages")
    axes[0].legend()

    axes[1].plot(d["date"], d["rsi"], color="purple")
    axes[1].axhline(70, color="red", linestyle="--", alpha=0.7)
    axes[1].axhline(30, color="green", linestyle="--", alpha=0.7)
    axes[1].set_title("RSI (14)")

    axes[2].plot(d["date"], d["MACD_12_26_9"], label="MACD")
    axes[2].plot(d["date"], d["MACDs_12_26_9"], label="signal")
    axes[2].bar(d["date"], d["MACDh_12_26_9"], label="histogram", alpha=0.4)
    axes[2].set_title("MACD")
    axes[2].legend()

    axes[3].plot(d["date"], d["close"], label="close")
    axes[3].plot(d["date"], d["BBU_20_2.0_2.0"], label="upper", linestyle="--")
    axes[3].plot(d["date"], d["BBL_20_2.0_2.0"], label="lower", linestyle="--")
    axes[3].fill_between(d["date"], d["BBL_20_2.0_2.0"], d["BBU_20_2.0_2.0"], alpha=0.1)
    axes[3].set_title("Bollinger Bands (20, 2sd)")
    axes[3].legend()

    plt.suptitle(f"{ticker} — indicator panel", fontsize=14)
    save_plot(filename)


# ===========================================================================
# STAGE 2B — ENHANCED FEATURES + MULTI-HORIZON LABELS
# ===========================================================================
def stage_2b_enhanced(master=None):
    section("STAGE 2B — ENHANCED FEATURES + MULTI-HORIZON LABELS")

    # reloading master if called standalone
    if master is None:
        master = pd.read_csv(os.path.join(DATA_PATH, "master.csv"), parse_dates=["date"])
    master = master.sort_values(["symbol", "date"]).reset_index(drop=True)

    # adding lagged indicator columns so trees can see recent history
    def add_lags(group):
        group = group.copy()
        for col in LAG_COLS:
            for lag in LAG_DAYS:
                group[f"{col}_lag{lag}"] = group[col].shift(lag)
        return group
    log("adding lagged features...")
    master = master.groupby("symbol", group_keys=False).apply(add_lags)

    # adding multi-day trailing returns as momentum features
    def add_returns(group):
        group = group.copy()
        group["return_1d"] = group["close"].pct_change(1)
        group["return_3d"] = group["close"].pct_change(3)
        group["return_5d"] = group["close"].pct_change(5)
        group["return_10d"] = group["close"].pct_change(10)
        return group
    log("adding return features...")
    master = master.groupby("symbol", group_keys=False).apply(add_returns)

    # adding market-relative and sector-relative performance
    master["market_return"] = master.groupby("date")["return_1d"].transform("mean")
    master["rel_to_market"] = master["return_1d"] - master["market_return"]
    master["sector_return"] = master.groupby(["date", "sector"])["return_1d"].transform("mean")
    master["rel_to_sector"] = master["return_1d"] - master["sector_return"]
    log("added relative features")

    # building direction labels for each horizon
    def make_label(pct, thr):
        if pct > thr:
            return "Up"
        if pct < -thr:
            return "Down"
        return "Neutral"

    def add_horizon_labels(group):
        group = group.copy()
        for name, cfg in HORIZONS.items():
            fwd = group["close"].pct_change(cfg["days"]).shift(-cfg["days"])
            group[f"fwd_return_{name}"] = fwd
            group[f"label_{name}"] = fwd.apply(lambda x: make_label(x, cfg["threshold"]))
        return group
    log("adding multi-horizon labels...")
    master = master.groupby("symbol", group_keys=False).apply(add_horizon_labels)

    # plotting label balance for each horizon
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, name in zip(axes, HORIZONS.keys()):
        counts = master[f"label_{name}"].value_counts()
        sns.barplot(x=counts.index, y=counts.values, hue=counts.index,
                    palette=["tomato", "steelblue", "seagreen"], legend=False, ax=ax)
        ax.set_title(f"label_{name}")
    plt.suptitle("label distribution across horizons")
    save_plot("s2b_horizon_label_distributions.png")
    for name in HORIZONS:
        log(f"label_{name} balance:\n"
            f"{master[f'label_{name}'].value_counts(normalize=True).mul(100).round(1)}")

    # dropping rows with nulls from lagging / forward returns
    before = master.shape[0]
    master = master.dropna().reset_index(drop=True)
    log(f"dropped {before - master.shape[0]:,} rows with nulls")
    log(f"enhanced master shape: {master.shape}")

    master.to_csv(os.path.join(DATA_PATH, "master_enhanced.csv"), index=False)
    log("saved master_enhanced.csv")
    log("stage 2b complete")
    return master


# ===========================================================================
# STAGE 3 — BASELINE CLASSIFICATION (POOLED, NEXT-DAY)
# ===========================================================================
def _prep_xy(train, val, test, feature_cols, label_col):
    # imputing, scaling, encoding using training stats only
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, LabelEncoder

    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(train[feature_cols])
    Xva = imp.transform(val[feature_cols])
    Xte = imp.transform(test[feature_cols])

    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr)
    Xva = sc.transform(Xva)
    Xte = sc.transform(Xte)

    le = LabelEncoder()
    ytr = le.fit_transform(train[label_col])
    yva = le.transform(val[label_col])
    yte = le.transform(test[label_col])
    return Xtr, Xva, Xte, ytr, yva, yte, imp, sc, le


def stage_3_classify():
    section("STAGE 3 — BASELINE CLASSIFICATION (POOLED, NEXT-DAY)")
    import pickle
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 ConfusionMatrixDisplay, f1_score, accuracy_score)
    from xgboost import XGBClassifier

    # loading splits from disk
    train = pd.read_csv(os.path.join(DATA_PATH, "train.csv"), parse_dates=["date"])
    val = pd.read_csv(os.path.join(DATA_PATH, "val.csv"), parse_dates=["date"])
    test = pd.read_csv(os.path.join(DATA_PATH, "test.csv"), parse_dates=["date"])

    Xtr, Xva, Xte, ytr, yva, yte, imp, sc, le = _prep_xy(
        train, val, test, BASE_FEATURE_COLS, "label")

    # training a class-weighted random forest baseline
    log("training random forest...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=12,
                                class_weight="balanced",
                                random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(Xtr, ytr)
    rf_pred = rf.predict(Xva)
    log("random forest — validation:\n" +
        classification_report(yva, rf_pred, target_names=le.classes_))

    # training a sample-weighted xgboost
    from sklearn.utils.class_weight import compute_sample_weight
    sw = compute_sample_weight("balanced", ytr)
    log("training xgboost...")
    xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                        eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1)
    xgb.fit(Xtr, ytr, sample_weight=sw, eval_set=[(Xva, yva)], verbose=False)
    xgb_pred = xgb.predict(Xva)
    log("xgboost — validation:\n" +
        classification_report(yva, xgb_pred, target_names=le.classes_))

    # plotting confusion matrices for both models
    for name, pred, cmap in [("rf", rf_pred, "Blues"), ("xgb", xgb_pred, "Oranges")]:
        cm = confusion_matrix(yva, pred)
        ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(cmap=cmap, colorbar=False)
        plt.title(f"{name} — confusion matrix (val)")
        save_plot(f"s3_{name}_confusion_val.png")

    # plotting rf feature importance
    fi = pd.DataFrame({"feature": BASE_FEATURE_COLS,
                       "importance": rf.feature_importances_}
                      ).sort_values("importance", ascending=False).head(15)
    plt.figure(figsize=(10, 6))
    sns.barplot(x="importance", y="feature", data=fi, hue="feature",
                palette="Blues_r", legend=False)
    plt.title("random forest — top 15 feature importances")
    save_plot("s3_rf_feature_importance.png")

    # evaluating the better model on the test set
    best = xgb if f1_score(yva, xgb_pred, average="macro") >= \
        f1_score(yva, rf_pred, average="macro") else rf
    best_name = "xgb" if best is xgb else "rf"
    test_pred = best.predict(Xte)
    log(f"best baseline model: {best_name}")
    log(f"{best_name} — TEST:\n" +
        classification_report(yte, test_pred, target_names=le.classes_))

    cm = confusion_matrix(yte, test_pred)
    ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(cmap="Purples", colorbar=False)
    plt.title(f"{best_name} — confusion matrix (test)")
    save_plot(f"s3_{best_name}_confusion_test.png")

    # saving artifacts for the chatbot stage
    for obj, fn in [(rf, "rf_model.pkl"), (xgb, "xgb_model.pkl"),
                    (sc, "scaler.pkl"), (imp, "imputer.pkl"),
                    (le, "label_encoder.pkl"), (BASE_FEATURE_COLS, "feature_cols.pkl")]:
        with open(os.path.join(MODEL_PATH, fn), "wb") as f:
            pickle.dump(obj, f)
    log("saved baseline models")
    log("stage 3 complete")


# ===========================================================================
# STAGE 3B — FULL EXPERIMENT (5 STRATEGIES x 3 HORIZONS)
# ===========================================================================
def _make_models():
    # building the five modelling strategies as fresh estimators
    from sklearn.ensemble import RandomForestClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from xgboost import XGBClassifier

    rf = RandomForestClassifier(n_estimators=200, max_depth=12,
                                class_weight="balanced",
                                random_state=RANDOM_STATE, n_jobs=-1)
    xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                        eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1)
    logreg = LogisticRegression(max_iter=500, class_weight="balanced", n_jobs=-1)
    stacked = StackingClassifier(
        estimators=[("rf", rf), ("xgb", xgb)],
        final_estimator=LogisticRegression(max_iter=500, class_weight="balanced"),
        n_jobs=-1)
    ovr = OneVsRestClassifier(
        XGBClassifier(n_estimators=150, max_depth=5, eval_metric="logloss",
                      random_state=RANDOM_STATE), n_jobs=-1)
    return {"random_forest": rf, "xgboost": xgb, "logistic_regression": logreg,
            "stacked_ensemble": stacked, "one_vs_rest": ovr}


def _fit_score(model, Xtr, ytr, Xva, yva, use_weight=False):
    # fitting one model and returning macro-f1 + accuracy on validation
    from sklearn.metrics import f1_score, accuracy_score
    from sklearn.utils.class_weight import compute_sample_weight
    try:
        if use_weight and model.__class__.__name__ == "XGBClassifier":
            model.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
        else:
            model.fit(Xtr, ytr)
        pred = model.predict(Xva)
        return (f1_score(yva, pred, average="macro"),
                accuracy_score(yva, pred), model)
    except Exception as e:
        log(f"    [skip] {model.__class__.__name__}: {e}")
        return (np.nan, np.nan, None)


def stage_3b_experiments():
    section("STAGE 3B — FULL EXPERIMENT (5 STRATEGIES x 3 HORIZONS)")

    # loading the enhanced dataset with lag/return/relative features
    df = pd.read_csv(os.path.join(DATA_PATH, "master_enhanced.csv"), parse_dates=["date"])

    # assembling the enhanced feature list
    lag_features = [f"{c}_lag{l}" for c in LAG_COLS for l in LAG_DAYS]
    feature_cols = BASE_FEATURE_COLS + lag_features + RETURN_FEATURES + RELATIVE_FEATURES
    feature_cols = [c for c in feature_cols if c in df.columns]
    log(f"using {len(feature_cols)} enhanced features")

    results = []          # collecting every (strategy, horizon, macro_f1) row

    # ---- POOLED and PER-SECTOR across all horizons and all five strategies ----
    for horizon in HORIZONS:
        label_col = f"label_{horizon}"
        section(f"HORIZON {horizon} — POOLED (all tickers, five strategies)")

        train = df[df["date"] < TRAIN_END]
        val = df[(df["date"] >= TRAIN_END) & (df["date"] < VAL_END)]
        Xtr, Xva, _, ytr, yva, _, _, _, _ = _prep_xy(train, val, val, feature_cols, label_col)

        for strat, model in _make_models().items():
            f1, acc, _ = _fit_score(model, Xtr, ytr, Xva, yva, use_weight=True)
            log(f"  pooled | {strat:20s} | macro_f1={f1:.4f} acc={acc:.4f}")
            results.append({"strategy": strat, "horizon": horizon,
                            "granularity": "pooled", "macro_f1": f1, "accuracy": acc})

        # per-sector with all five strategies, averaging macro-f1 across sectors
        section(f"HORIZON {horizon} — PER-SECTOR")
        for strat in _make_models():
            sector_f1s = []
            for sector in df["sector"].dropna().unique():
                sub = df[df["sector"] == sector]
                s_tr = sub[sub["date"] < TRAIN_END]
                s_va = sub[(sub["date"] >= TRAIN_END) & (sub["date"] < VAL_END)]
                if len(s_tr) < 500 or len(s_va) < 100:
                    continue
                Xtr, Xva, _, ytr, yva, _, _, _, _ = _prep_xy(
                    s_tr, s_va, s_va, feature_cols, label_col)
                model = _make_models()[strat]
                f1, _, _ = _fit_score(model, Xtr, ytr, Xva, yva, use_weight=True)
                if not np.isnan(f1):
                    sector_f1s.append(f1)
            avg = float(np.mean(sector_f1s)) if sector_f1s else np.nan
            log(f"  per-sector | {strat:20s} | avg macro_f1={avg:.4f} "
                f"({len(sector_f1s)} sectors)")
            results.append({"strategy": strat, "horizon": horizon,
                            "granularity": "per_sector", "macro_f1": avg,
                            "accuracy": np.nan})

    # ---- PER-STOCK with xgboost only (best pooled model), all horizons ----
    section("PER-STOCK (xgboost only, all horizons)")
    for horizon in HORIZONS:
        label_col = f"label_{horizon}"
        stock_f1s = []
        tickers = df["symbol"].unique()
        for i, tk in enumerate(tickers):
            sub = df[df["symbol"] == tk]
            s_tr = sub[sub["date"] < TRAIN_END]
            s_va = sub[(sub["date"] >= TRAIN_END) & (sub["date"] < VAL_END)]
            if len(s_tr) < 300 or len(s_va) < 50:
                continue
            try:
                Xtr, Xva, _, ytr, yva, _, _, _, _ = _prep_xy(
                    s_tr, s_va, s_va, feature_cols, label_col)
                from xgboost import XGBClassifier
                from sklearn.metrics import f1_score
                m = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                                  eval_metric="mlogloss", random_state=RANDOM_STATE)
                m.fit(Xtr, ytr)
                stock_f1s.append(f1_score(yva, m.predict(Xva), average="macro"))
            except Exception:
                continue
            if i % 100 == 0:
                log(f"    processed {i}/{len(tickers)} tickers for horizon {horizon}")
        avg = float(np.mean(stock_f1s)) if stock_f1s else np.nan
        log(f"  per-stock | xgboost | horizon {horizon} | "
            f"avg macro_f1={avg:.4f} ({len(stock_f1s)} stocks)")
        results.append({"strategy": "xgboost", "horizon": horizon,
                        "granularity": "per_stock", "macro_f1": avg, "accuracy": np.nan})

    # ---- results table + comparison plots ----
    res = pd.DataFrame(results)
    res.to_csv(os.path.join(OBS_PATH, "s3b_experiment_results.csv"), index=False)
    log("\nFULL EXPERIMENT RESULTS:\n" + res.to_string(index=False))

    # plotting macro-f1 by strategy for each horizon (pooled only, five strategies)
    pooled = res[res["granularity"] == "pooled"]
    plt.figure(figsize=(12, 5))
    sns.barplot(data=pooled, x="strategy", y="macro_f1", hue="horizon")
    plt.title("pooled — macro F1 by strategy and horizon")
    plt.xticks(rotation=30)
    plt.axhline(0.333, color="red", linestyle="--", label="random baseline")
    plt.legend()
    save_plot("s3b_pooled_macro_f1.png")

    # plotting granularity comparison (pooled vs per-sector vs per-stock) for xgboost
    xgb_rows = res[res["strategy"] == "xgboost"]
    plt.figure(figsize=(10, 5))
    sns.barplot(data=xgb_rows, x="horizon", y="macro_f1", hue="granularity")
    plt.title("xgboost — macro F1 by horizon and granularity")
    plt.axhline(0.333, color="red", linestyle="--", label="random baseline")
    plt.legend()
    save_plot("s3b_granularity_comparison.png")

    # plotting horizon effect averaged across strategies
    plt.figure(figsize=(9, 5))
    sns.barplot(data=res, x="horizon", y="macro_f1", hue="granularity")
    plt.title("macro F1 by horizon (all strategies)")
    plt.axhline(0.333, color="red", linestyle="--")
    save_plot("s3b_horizon_effect.png")

    # printing which combinations beat the next-day pooled baseline
    baseline = pooled[(pooled["horizon"] == "1d") &
                      (pooled["strategy"] == "xgboost")]["macro_f1"].values
    baseline = float(baseline[0]) if len(baseline) else np.nan
    log(f"\nnext-day pooled xgboost baseline macro_f1 = {baseline:.4f}")
    improved = res[res["macro_f1"] > baseline].sort_values("macro_f1", ascending=False)
    log("configurations improving on the baseline:\n" + improved.to_string(index=False))

    log("stage 3b complete")
    return res


# ===========================================================================
# ENTRY POINT
# ===========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all",
                        choices=["1", "2", "2b", "3", "3b", "all"])
    args = parser.parse_args()

    if args.stage in ("1", "all"):
        p, f, s = stage_1_load()
    if args.stage in ("2", "all"):
        stage_2_features()
    if args.stage in ("2b", "all"):
        stage_2b_enhanced()
    if args.stage in ("3", "all"):
        stage_3_classify()
    if args.stage in ("3b", "all"):
        stage_3b_experiments()

    log("\nALL REQUESTED STAGES COMPLETE")
    log(f"observations and plots saved in: {OBS_PATH}")


if __name__ == "__main__":
    main()