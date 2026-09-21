"""backtesting each rule strategy with ritter's risk-averse reward and costs"""

import numpy as np
import pandas as pd
from core.config import (BT_RISK_AVERSION_KAPPA, BT_SPREAD_BPS, BT_IMPACT_BPS,
                         BT_ANNUAL_DAYS, MAX_OPEN_POSITIONS, BT_WF_FOLDS,
                         BT_WF_MIN_FOLD_DAYS)


def score_matrix(bars, strategy):
    # building a dates x tickers frame of strategy scores (0 = flat)
    cols = {}
    for ticker, df in bars.items():
        try:
            cols[ticker] = strategy.score_series(df)
        except Exception as e:
            print(f"[backtest] {strategy.NAME} {ticker}: {e}")
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols).sort_index().fillna(0.0)


def close_matrix(bars):
    # aligning closes on one calendar
    return pd.DataFrame({t: df["close"] for t, df in bars.items()}) \
        .sort_index()


def target_weights(scores, max_holdings=MAX_OPEN_POSITIONS):
    # holding the top-n scores each day at 1/n each; unfilled slots stay cash,
    # matching how execution caps open positions in live trading
    if scores.empty:
        return scores
    ranks = scores.where(scores > 0).rank(axis=1, ascending=False,
                                         method="first")
    held = (ranks <= max_holdings) & (scores > 0)
    return held.astype(float) / float(max_holdings)


def run(bars, strategy, regime_mask=None, kappa=BT_RISK_AVERSION_KAPPA,
        max_holdings=MAX_OPEN_POSITIONS, cost_bps=None, start=None):
    # simulating one strategy and returning its metrics and equity curve;
    # indicators warm up on the full history, metrics count from `start`
    cost_bps = BT_SPREAD_BPS + BT_IMPACT_BPS if cost_bps is None else cost_bps
    scores = score_matrix(bars, strategy)
    if scores.empty:
        return None
    closes = close_matrix(bars).reindex(scores.index).ffill()
    weights = target_weights(scores, max_holdings)
    if regime_mask is not None:
        mask = regime_mask.reindex(weights.index).ffill().fillna(False)
        weights = weights.mul(mask.astype(float), axis=0)

    # weights decided at close t earn the return of t -> t+1 (no lookahead)
    rets = closes.pct_change().fillna(0.0)
    gross = (weights.shift(1).fillna(0.0) * rets).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    net = gross - turnover * cost_bps / 1e4
    if start is not None:
        keep = net.index >= pd.Timestamp(start)
        net, weights = net[keep], weights[keep]
    if net.empty:
        return None

    # ritter eq 13 with mu_hat = 0: reward = dw - (kappa/2) dw^2
    reward = net - (kappa / 2.0) * net ** 2
    equity = (1 + net).cumprod()
    peak = equity.cummax()
    drawdown = (equity / peak - 1).min()
    invested = weights.shift(1).sum(axis=1) > 0
    n = len(net)
    years = max(n / BT_ANNUAL_DAYS, 1e-9)
    std = float(net.std())
    return {
        "strategy": strategy.NAME,
        "utility": round(float(reward.mean()) * BT_ANNUAL_DAYS, 4),
        "sharpe": round(float(net.mean() / std * np.sqrt(BT_ANNUAL_DAYS)), 3)
        if std > 0 else 0.0,
        "cagr": round(float(equity.iloc[-1]) ** (1 / years) - 1, 4),
        "max_drawdown": round(float(drawdown), 4),
        "hit_rate": round(float((net[invested] > 0).mean()), 3)
        if invested.any() else 0.0,
        "exposure": round(float(weights.sum(axis=1).mean()), 3),
        "avg_turnover": round(float(turnover.mean()), 4),
        "n_days": int(n),
        "universe_size": int(scores.shape[1]),
        "equity": equity,
        "net_returns": net,
    }


def _utility_of(net, kappa=BT_RISK_AVERSION_KAPPA):
    # ritter reward mean, annualised, for one return series
    if net is None or len(net) == 0:
        return 0.0
    reward = net - (kappa / 2.0) * net ** 2
    return float(reward.mean()) * BT_ANNUAL_DAYS


def walk_forward_score(net, folds=BT_WF_FOLDS, kappa=BT_RISK_AVERSION_KAPPA):
    # splitting the return series into contiguous out-of-sample folds and
    # scoring each; the WORST fold utility is the robustness metric, the spread
    # tells you how regime-dependent the strategy is. a strategy that only
    # works in one fold shows a low worst-fold even with a high average.
    net = net.dropna()
    if len(net) < BT_WF_MIN_FOLD_DAYS * 2:
        u = _utility_of(net, kappa)
        return {"worst_fold_utility": round(u, 4),
                "mean_fold_utility": round(u, 4),
                "fold_spread": 0.0, "n_folds": 1}
    size = len(net) // folds
    utils = []
    for i in range(folds):
        lo = i * size
        hi = len(net) if i == folds - 1 else (i + 1) * size
        fold = net.iloc[lo:hi]
        if len(fold) >= BT_WF_MIN_FOLD_DAYS:
            utils.append(_utility_of(fold, kappa))
    if not utils:
        u = _utility_of(net, kappa)
        return {"worst_fold_utility": round(u, 4),
                "mean_fold_utility": round(u, 4),
                "fold_spread": 0.0, "n_folds": 1}
    return {"worst_fold_utility": round(min(utils), 4),
            "mean_fold_utility": round(sum(utils) / len(utils), 4),
            "fold_spread": round(max(utils) - min(utils), 4),
            "n_folds": len(utils)}


def summary_row(result):
    # dropping the series so the dict can be stored or printed
    return {k: v for k, v in result.items()
            if k not in ("equity", "net_returns")}
