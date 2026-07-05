"""running the fixed three-round debate that terminates by construction"""

from collections import Counter
from engine.panels import (run_cases, run_rebuttal, run_judges,
                           BUY_PANEL, SELL_PANEL)


def majority_vote(votes, default="NO_TRADE"):
    # requiring a strict majority of judges, defaulting to no trade
    if not votes:
        return default
    counts = Counter(v["vote"] for v in votes)
    top, n = counts.most_common(1)[0]
    return top if n > len(votes) / 2 else default


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
