"""electing the live strategy from ritter-utility backtests over the universe"""

from datetime import date, timedelta
from core.config import (STRATEGY_DEFAULT, STRATEGY_CASH, BT_LOOKBACK_MONTHS,
                         BT_MIN_EXPOSURE, BARS_LOOKBACK_DAYS, REGIME_INDEX,
                         REGIME_VOL_INDEX, REGIME_GATE)
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
    rows = []
    for name, strat in REGISTRY.items():
        # indicators warm up on all bars; metrics count from the window start
        res = run(universe, strat, regime_mask=mask, start=start)
        if res is None:
            print(f"[election] {name}: no result in window")
            continue
        if res["n_days"] < 60:
            print(f"[election] {name}: only {res['n_days']} days in window")
        rows.append(summary_row(res))
    rows.sort(key=lambda r: r["utility"], reverse=True)
    return rows


def save_backtests(rows):
    # storing the leaderboard for the site and the weekly report
    today = str(date.today())
    payload = [{"run_date": today, **r} for r in rows]
    if payload:
        get_client().table("strategy_backtests").upsert(payload).execute()


def elect(rows):
    # choosing the top strategy by ritter utility that is actually invested;
    # electing cash when nothing earns a positive risk-adjusted reward
    eligible = [r for r in rows if r["utility"] > 0
                and r["exposure"] >= BT_MIN_EXPOSURE]
    return eligible[0]["strategy"] if eligible else STRATEGY_CASH


def run_election(limit=None):
    # weekly entry point: backtest, store, elect, report
    rows = backtest_all(limit=limit)
    for r in rows:
        print(f"  {r['strategy']:<20} utility {r['utility']:+.3f}  "
              f"sharpe {r['sharpe']:+.2f}  cagr {r['cagr']:+.1%}  "
              f"maxdd {r['max_drawdown']:+.1%}  exposure {r['exposure']:.0%}")
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
