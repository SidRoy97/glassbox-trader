"""running the fixed three-round debate that terminates by construction"""

from collections import Counter
from engine.panels import (run_cases, run_rebuttal, run_judges,
                           BUY_PANEL, SELL_PANEL)


def majority_vote(votes, default="NO_TRADE"):
    # requiring a strict weighted majority of judges: every judge starts
    # at weight 1.0 and drifts only with their own scored record; a
    # directional verdict additionally needs at least two judges agreeing
    if not votes:
        return default
    try:
        from engine.judge_weights import provider_weights
        weights = provider_weights()
    except Exception:
        weights = {}
    total, tallies, heads = 0.0, {}, Counter()
    for v in votes:
        w = weights.get(v.get("provider"), 1.0)
        total += w
        tallies[v["vote"]] = tallies.get(v["vote"], 0.0) + w
        heads[v["vote"]] += 1
    top = max(tallies, key=tallies.get)
    if tallies[top] <= total / 2:
        return default
    if top in ("BUY", "SELL") and heads[top] < 2:
        return default
    return top


def decide(packet):
    # executing round 1 cases, round 2 rebuttals, round 3 votes — then stop
    bull = run_cases(packet, "bull", BUY_PANEL)
    bear = run_cases(packet, "bear", SELL_PANEL)

    bull_reb = run_rebuttal(packet, "bull", BUY_PANEL, bear)
    bear_reb = run_rebuttal(packet, "bear", SELL_PANEL, bull)

    votes = run_judges(packet, bull, bear, bull_reb, bear_reb)
    decision = majority_vote(votes)

    return {"decision": decision,
            "bull_case": {"opening": bull, "rebuttal": bull_reb},
            "bear_case": {"opening": bear, "rebuttal": bear_reb},
            "judge_votes": votes}
