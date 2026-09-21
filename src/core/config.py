"""holding every path, constant, and strategy parameter in one place"""

import os

# resolving the base folder from the environment so machines can differ
BASE_PATH = os.environ.get("STOCK_LENS_BASE", os.path.abspath("./stock-lens"))
DATA_PATH = os.path.join(BASE_PATH, "data")
OBS_PATH = os.path.join(BASE_PATH, "observations")
for _p in (DATA_PATH, OBS_PATH):
    os.makedirs(_p, exist_ok=True)


def _envfloat(name, default):
    # reading a float env var, tolerating empty or malformed values
    raw = os.environ.get(name, "")
    try:
        return float(raw) if raw.strip() else default
    except ValueError:
        return default


def _envint(name, default):
    # reading an int env var, tolerating empty or malformed values
    raw = os.environ.get(name, "")
    try:
        return int(raw) if raw.strip() else default
    except ValueError:
        return default


# ---- universe -------------------------------------------------------------
# single names come from universe.csv (S&P 500); etfs are appended so the
# low-vol strategy can prefer broad, diversified vehicles over single stocks
UNIVERSE_FILE = "universe.csv"
ETF_UNIVERSE = ["SPY", "QQQ", "IWM", "DIA", "VTI", "VOO",
                "XLK", "XLF", "XLV", "XLE", "XLI", "XLP", "XLY", "XLU",
                "XLB", "XLRE", "XLC",
                "VEA", "VWO", "AGG", "TLT", "GLD"]
ETF_SECTOR = "ETF"
# enough daily bars for a 252-day momentum window plus indicator warmup
BARS_LOOKBACK_DAYS = _envint("BARS_LOOKBACK_DAYS", 420)
# minimum bars a ticker needs before any strategy will read it
STRATEGY_MIN_DAYS = 260

# ---- strategies (rule-based, long-only) -----------------------------------
# momentum_12_1: jansen ch4 momentum factor — 12 month return skipping the
# most recent month, taken only while price holds above the long trend average
MOMENTUM_LOOKBACK = 252
MOMENTUM_SKIP = 21
MOMENTUM_MIN_RETURN = _envfloat("MOMENTUM_MIN_RETURN", 0.0)
MOMENTUM_FULL_SCORE_RETURN = 0.50   # a 50% 12-1 return maps to score 1.0

# mean_reversion_bb: ml4t 1.1.1 bollinger bands — buying %b dips with an rsi
# confirm, only inside an uptrend, exiting at the band midline or on time
MR_BB_WINDOW = 20
MR_BB_STD = 2.0
MR_ENTRY_PCTB = _envfloat("MR_ENTRY_PCTB", 0.10)
MR_EXIT_PCTB = 0.50
MR_RSI_MAX = _envfloat("MR_RSI_MAX", 30.0)
MR_RSI_SPAN = 14
MR_MAX_HOLD_DAYS = 10

# low_vol_quality: jansen ch4 volatility premium — preferring calm names that
# are still trending up, so risk is taken where it is cheapest
LOWVOL_WINDOW = 60
LOWVOL_MAX_ANN_VOL = _envfloat("LOWVOL_MAX_ANN_VOL", 0.30)
LOWVOL_TREND_MA = 100

# shared long-term trend filter used by every strategy
TREND_MA = 200

# ---- market regime (ml4t market factor / capm) ----------------------------
REGIME_INDEX = "SPY"
REGIME_VOL_INDEX = "^VIX"
REGIME_MA = 200
REGIME_VIX_MAX = _envfloat("REGIME_VIX_MAX", 25.0)
REGIME_GATE = os.environ.get("REGIME_GATE", "1") == "1"

# ---- news gate (veto / confirm, never a predictor) ------------------------
NEWS_VETO_SENTIMENT = _envfloat("NEWS_VETO_SENTIMENT", -0.30)
NEWS_CONFIRM_SENTIMENT = _envfloat("NEWS_CONFIRM_SENTIMENT", 0.15)
NEWS_CONFIRM_BONUS = 0.10

# ---- backtest and strategy election (ritter 2017) -------------------------
# kappa is ritter's risk-aversion in the reward r - (kappa/2) r^2 on daily
# fractional returns. set once, not tuned per backtest: kappa=4 penalizes a
# 2% down day at roughly its own magnitude, which keeps the scorer risk-aware
# without being so strict that cash wins every week. raising it toward 10
# makes the system more defensive; lowering it toward 1 chases raw return.
BT_RISK_AVERSION_KAPPA = _envfloat("BT_RISK_AVERSION_KAPPA", 4.0)
# ritter eq 17-18 folded into basis points per unit of turnover
BT_SPREAD_BPS = 5.0
BT_IMPACT_BPS = 5.0
BT_LOOKBACK_MONTHS = _envint("BT_LOOKBACK_MONTHS", 12)
BT_MIN_EXPOSURE = 0.10        # a strategy that is almost never invested cannot win
BT_ANNUAL_DAYS = 252

# walk-forward validation: score each strategy on rolling out-of-sample folds
# instead of one trailing window, and rank on the WORST fold's utility. this
# rewards strategies that hold up across regimes and punishes ones that only
# worked in one stretch — the main guard against overfitting the last year.
BT_WALK_FORWARD = os.environ.get("BT_WALK_FORWARD", "1") == "1"
BT_WF_FOLDS = _envint("BT_WF_FOLDS", 4)
BT_WF_MIN_FOLD_DAYS = 40      # a fold shorter than this is not scored

# election policy: when no strategy clears positive utility, fall back to the
# least-risky one (smallest drawdown) IF its drawdown is tolerable, rather
# than always sitting in cash. keeps the system defensive but not inert.
STRATEGY_DEFAULT = "momentum_12_1"
STRATEGY_CASH = "cash"        # elected when even the fallback is too risky
FALLBACK_MAX_DRAWDOWN = _envfloat("FALLBACK_MAX_DRAWDOWN", 0.15)

# top-k blend: seat the best K strategies together (equal weight) instead of
# winner-take-all, to diversify across premia and cut single-strategy variance.
# 1 restores winner-take-all. the champion string becomes "a+b" when K>1.
STRATEGY_BLEND_K = _envint("STRATEGY_BLEND_K", 2)

# ---- portfolio limits shared by backtest and execution --------------------
MAX_OPEN_POSITIONS = _envint("MAX_OPEN_POSITIONS", 5)
