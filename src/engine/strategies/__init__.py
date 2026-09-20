"""registering every rule strategy so screener, packet, and backtest agree"""

from engine.strategies import momentum, mean_reversion, low_vol

# order matters only for display; election ranks on backtest utility
REGISTRY = {momentum.NAME: momentum,
            mean_reversion.NAME: mean_reversion,
            low_vol.NAME: low_vol}


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
