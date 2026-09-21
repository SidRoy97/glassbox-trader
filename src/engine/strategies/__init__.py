"""registering every rule strategy so screener, packet, and backtest agree"""

from engine.strategies import (momentum, mean_reversion, low_vol,
                               trend_follow, dual_momentum, index_regime)

# order matters only for display; election ranks on backtest utility
REGISTRY = {momentum.NAME: momentum,
            mean_reversion.NAME: mean_reversion,
            low_vol.NAME: low_vol,
            trend_follow.NAME: trend_follow,
            dual_momentum.NAME: dual_momentum,
            index_regime.NAME: index_regime}


def get_strategy(name):
    # resolving a strategy module by name, raising a clear error otherwise
    if name not in REGISTRY:
        raise KeyError(f"unknown strategy {name!r}; "
                       f"known: {sorted(REGISTRY)}")
    return REGISTRY[name]


def all_signals(df):
    # reading every strategy on one frame for the glass-box packet
    out = {}
    for name, mod in REGISTRY.items():
        try:
            sig = mod.signal(df)
            out[name] = {"direction": sig["direction"],
                         "score": sig["score"]}
        except Exception as e:
            out[name] = {"direction": "unavailable", "score": 0.0,
                         "error": str(e)[:120]}
    return out


def champion_members(champion):
    # splitting a possibly-blended champion string "a+b" into its modules
    return [get_strategy(n) for n in str(champion).split("+") if n]


def blended_signal(champion, df):
    # combining member strategies into one signal: BUY if any member buys,
    # score is the mean of member scores, reason lists the contributing members
    members = champion_members(champion)
    sigs = [m.signal(df) for m in members]
    buys = [s for s in sigs if s.get("direction") == "BUY"]
    if not buys:
        return {"model": champion, "direction": "NO_TRADE", "score": 0.0,
                "reason": "no member strategy signalled BUY"}
    score = round(sum(s["score"] for s in buys) / len(members), 4)
    reason = "; ".join(f"{s['model']}: {s['reason']}" for s in buys)
    return {"model": champion, "direction": "BUY", "score": score,
            "reason": reason, "members": [s["model"] for s in buys]}
