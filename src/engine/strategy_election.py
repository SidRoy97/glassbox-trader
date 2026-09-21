"""electing the live strategy from ritter-utility backtests over the universe"""

from datetime import date, timedelta
from core.config import (STRATEGY_DEFAULT, STRATEGY_CASH, BT_LOOKBACK_MONTHS,
                         BT_MIN_EXPOSURE, BARS_LOOKBACK_DAYS, REGIME_INDEX,
                         REGIME_VOL_INDEX, REGIME_GATE, BT_WALK_FORWARD,
                         FALLBACK_MAX_DRAWDOWN, STRATEGY_BLEND_K)
from engine.memory import get_client

CONFIG_KEY = "strategy_champion"


def get_strategy_champion():
    # reading the elected strategy, falling back to the default rule
    try:
        res = get_client().table("config").select("value") \
            .eq("key", CONFIG_KEY).limit(1).execute().data
        return res[0]["value"] if res else STRATEGY_DEFAULT
    except Exception as e:
        print(f"[election] champion read failed, using default: {e}")
        return STRATEGY_DEFAULT


def set_strategy_champion(name):
    # recording a new elected strategy
    get_client().table("config").upsert(
        {"key": CONFIG_KEY, "value": str(name)[:40]}).execute()


def _regime_mask(bars):
    # building the causal risk-on mask over the backtest window
    from engine.strategies.regime import risk_on_series
    if not REGIME_GATE:
        return None
    try:
        return risk_on_series(bars[REGIME_INDEX]["close"],
                              bars[REGIME_VOL_INDEX]["close"])
    except KeyError as e:
        print(f"[election] regime series unavailable ({e}); no mask")
        return None


def backtest_all(bars=None, limit=None, window_months=BT_LOOKBACK_MONTHS):
    # backtesting every registered strategy on the same bars and window
    from engine.backtest import run, summary_row
    from engine.strategies import REGISTRY
    from engine.market_data import load_universe, download_bars
    if bars is None:
        tickers = [t for t, _ in load_universe(limit=limit)]
        bars = download_bars(tickers + [REGIME_INDEX, REGIME_VOL_INDEX],
                             days=BARS_LOOKBACK_DAYS)
    mask = _regime_mask(bars)
    universe = {t: df for t, df in bars.items()
                if t not in (REGIME_VOL_INDEX,)}
    start = date.today() - timedelta(days=int(window_months * 30.5))
    from engine.backtest import walk_forward_score
    rows = []
    for name, strat in REGISTRY.items():
        # indicators warm up on all bars; metrics count from the window start
        res = run(universe, strat, regime_mask=mask, start=start)
        if res is None:
            print(f"[election] {name}: no result in window")
            continue
        if res["n_days"] < 60:
            print(f"[election] {name}: only {res['n_days']} days in window")
        row = summary_row(res)
        # walk-forward robustness: worst out-of-sample fold and the spread
        if BT_WALK_FORWARD:
            row.update(walk_forward_score(res["net_returns"]))
        rows.append(row)
    # rank on the robust metric when walk-forward is on: a strategy must hold
    # up in its weakest window, not just on average over the trailing year
    key = "worst_fold_utility" if BT_WALK_FORWARD else "utility"
    rows.sort(key=lambda r: r.get(key, r["utility"]), reverse=True)
    return rows


def save_backtests(rows):
    # storing the leaderboard for the site and the weekly report
    today = str(date.today())
    payload = [{"run_date": today, **r} for r in rows]
    if payload:
        get_client().table("strategy_backtests").upsert(payload).execute()


def _robust_utility(row):
    # the metric we elect on: worst out-of-sample fold when walk-forward is on,
    # otherwise the trailing-window utility
    return row.get("worst_fold_utility", row["utility"]) \
        if BT_WALK_FORWARD else row["utility"]


def elect(rows):
    # three-tier policy:
    #  1. blend the top-K strategies that are robustly positive and invested
    #  2. if none clear that bar, fall back to the least-risky strategy, but
    #     only if its drawdown is tolerable — a defensive trade beats sitting
    #     out when the safest option is genuinely safe
    #  3. otherwise cash: even the calmest strategy is too risky right now
    invested = [r for r in rows if r["exposure"] >= BT_MIN_EXPOSURE]
    positive = [r for r in invested if _robust_utility(r) > 0]
    if positive:
        positive.sort(key=_robust_utility, reverse=True)
        winners = positive[:max(1, STRATEGY_BLEND_K)]
        return "+".join(w["strategy"] for w in winners)
    # fallback: the least-risky invested strategy, if its drawdown is bearable
    if invested:
        safest = min(invested, key=lambda r: abs(r["max_drawdown"]))
        if abs(safest["max_drawdown"]) <= FALLBACK_MAX_DRAWDOWN:
            print(f"[election] no strategy robustly positive; falling back to "
                  f"least-risky {safest['strategy']} "
                  f"(maxdd {safest['max_drawdown']:+.1%})")
            return safest["strategy"]
        print(f"[election] safest strategy {safest['strategy']} maxdd "
              f"{safest['max_drawdown']:+.1%} exceeds "
              f"{FALLBACK_MAX_DRAWDOWN:.0%} — staying in cash")
    return STRATEGY_CASH


def run_election(limit=None):
    # weekly entry point: backtest, store, elect, report
    rows = backtest_all(limit=limit)
    for r in rows:
        wf = (f"  worst-fold {r['worst_fold_utility']:+.3f}  "
              f"spread {r.get('fold_spread', 0):.3f}"
              if BT_WALK_FORWARD and "worst_fold_utility" in r else "")
        print(f"  {r['strategy']:<20} utility {r['utility']:+.3f}  "
              f"sharpe {r['sharpe']:+.2f}  cagr {r['cagr']:+.1%}  "
              f"maxdd {r['max_drawdown']:+.1%}  exposure {r['exposure']:.0%}"
              f"{wf}")
    try:
        save_backtests(rows)
    except Exception as e:
        print(f"[election] could not save backtests: {e}")
    winner = elect(rows)
    current = get_strategy_champion()
    if winner != current:
        set_strategy_champion(winner)
        print(f"election: {current} -> {winner}")
    else:
        print(f"election: {current} keeps the title")
    return winner, rows
