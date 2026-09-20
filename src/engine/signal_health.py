"""monitoring the elected strategy's live health and deriving a risk multiplier

This module is READ-ONLY over decisions (it never trades). It answers two
questions: is the live strategy's hit rate decaying toward random (drift),
and is the market in a directionless regime where longs lose (regime). It
cannot make the signal more accurate; it PROTECTS capital when the signal
weakens by shrinking risk per trade and the daily trade cap.
"""

import os
from engine.memory import get_client

# --- config (all env-tunable) ---
DRIFT_WINDOW = int(os.environ.get("DRIFT_WINDOW", "150"))          # recent scored reads to judge on
DRIFT_MIN_N = int(os.environ.get("DRIFT_MIN_N", "40"))            # need this many before judging
RANDOM_BASELINE = float(os.environ.get("RANDOM_BASELINE", "0.3333"))
# how far above random the strategy must be to be "healthy"; below this we derisk
DRIFT_HEALTHY_EDGE = float(os.environ.get("DRIFT_HEALTHY_EDGE", "0.05"))  # +5pp over random
# risk multipliers applied by execution when signal is degraded (never > 1.0)
DERISK_FLOOR = float(os.environ.get("DERISK_FLOOR", "0.25"))      # never cut below 25% risk


def _champion():
    from engine.strategy_election import get_strategy_champion
    return get_strategy_champion()


def _recent_scored(window=DRIFT_WINDOW):
    # scored decisions with a strategy read, newest first
    rows = get_client().table("decisions") \
        .select("strategy_name,strategy_direction,outcome_label,was_correct") \
        .not_.is_("scored_at", "null") \
        .order("decided_at", desc=True).limit(int(window) * 3) \
        .execute().data or []
    return rows


def drift_status():
    # is the elected strategy's recent BUY hit rate healthily above random?
    champ = _champion()
    rows = [r for r in _recent_scored()
            if r.get("strategy_name") == champ
            and r.get("strategy_direction") == "BUY"][:DRIFT_WINDOW]
    n = len(rows)
    if n < DRIFT_MIN_N:
        return {"champion": champ, "n": n, "hit_rate": None,
                "edge": None, "state": "insufficient_data",
                "risk_multiplier": 1.0}
    hits = sum(1 for r in rows if r.get("outcome_label") == "Up")
    hit_rate = hits / n
    edge = hit_rate - RANDOM_BASELINE
    # translate edge into a risk multiplier: full risk at healthy edge, scaling
    # down linearly to the floor as edge -> 0, and floor if at/below random.
    if edge >= DRIFT_HEALTHY_EDGE:
        mult, state = 1.0, "healthy"
    elif edge <= 0:
        mult, state = DERISK_FLOOR, "at_or_below_random"
    else:
        frac = edge / DRIFT_HEALTHY_EDGE
        mult = round(DERISK_FLOOR + (1.0 - DERISK_FLOOR) * frac, 3)
        state = "degrading"
    return {"champion": champ, "n": n, "hit_rate": round(hit_rate, 4),
            "edge": round(edge, 4), "state": state,
            "risk_multiplier": mult}


def regime_status(window=None):
    # reading the market regime from scored outcomes: when a large share of
    # recent days are Neutral the market is directionless and longs lose, so
    # the safe play is to trade less. stable over many decisions, not a
    # reaction to recent P&L. returns a label + cap multiplier in [floor, 1].
    win = int(window or os.environ.get("REGIME_WINDOW", "200"))
    try:
        rows = get_client().table("decisions") \
            .select("outcome_label") \
            .not_.is_("scored_at", "null") \
            .order("decided_at", desc=True).limit(win).execute().data or []
    except Exception as e:
        return {"regime": "unknown", "cap_multiplier": 1.0, "detail": str(e)}
    outcomes = [r.get("outcome_label") for r in rows
                if r.get("outcome_label") is not None]
    if len(outcomes) < 30:
        return {"regime": "insufficient_data", "cap_multiplier": 1.0,
                "neutral_share": None}
    neutral_share = sum(1 for o in outcomes if o == "Neutral") / len(outcomes)

    hi = float(os.environ.get("REGIME_NEUTRAL_HIGH", "0.55"))
    lo = float(os.environ.get("REGIME_NEUTRAL_LOW", "0.40"))
    floor = float(os.environ.get("REGIME_CAP_FLOOR", "0.25"))
    if neutral_share >= hi:
        mult, regime = floor, "neutral_dominated"
    elif neutral_share <= lo:
        mult, regime = 1.0, "directional"
    else:
        frac = (hi - neutral_share) / (hi - lo)
        mult = round(floor + (1.0 - floor) * frac, 3)
        regime = "mixed"
    return {"regime": regime, "neutral_share": round(neutral_share, 3),
            "cap_multiplier": mult, "n": len(outcomes)}


def regime_cap_multiplier():
    # the single number risk_gate multiplies the flexible daily cap by
    if os.environ.get("REGIME_CAP", "1") != "1":
        return 1.0
    return regime_status().get("cap_multiplier", 1.0)


def signal_risk_multiplier(ticker=None):
    # the single number execution multiplies RISK_PER_TRADE by. always in
    # [DERISK_FLOOR, 1.0]; defaults to 1.0 when data is thin, so it never
    # silently amplifies risk — it can only hold or reduce it.
    d = drift_status()
    mult = min(1.0, max(DERISK_FLOOR, d["risk_multiplier"]))
    return mult, {"drift": d, "risk_multiplier": mult}


def health_report():
    # one-line-per-item health summary for the weekly review / logs
    d = drift_status()
    print("signal health — drift:")
    if d["state"] == "insufficient_data":
        print(f"  strategy {d['champion']}: only {d['n']} scored BUY reads — "
              f"need {DRIFT_MIN_N} to judge drift")
    else:
        print(f"  strategy {d['champion']}: hit {d['hit_rate']:.0%} on "
              f"{d['n']} (edge {d['edge']:+.1%} vs random) — {d['state']}; "
              f"risk x{d['risk_multiplier']}")
    reg = regime_status()
    print("signal health — market regime (from outcomes):")
    if reg.get("neutral_share") is not None:
        print(f"  {reg['regime']}: {reg['neutral_share']:.0%} of recent days "
              f"Neutral — trade cap x{reg['cap_multiplier']}")
    return {"drift": d, "regime": reg}
