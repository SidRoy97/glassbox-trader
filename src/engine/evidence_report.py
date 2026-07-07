"""grading each evidence concept by the scored outcomes of calls citing it"""

import json
from engine.memory import get_client

WINDOW = 500                 # judging over the most recent scored decisions
MIN_CITATIONS = 5            # hiding buckets with too little evidence

BUCKETS = [
    ("liquidity_sweep", ("sweep",)),
    ("fvg_order_block", ("fvg", "order_block", "zone")),
    ("market_structure", ("structure",)),
    ("ema_regime", ("ema", "regime", "ma50", "200")),
    ("divergence", ("divergence",)),
    ("vix_fix_capitulation", ("capitulation", "wvf", "vix")),
    ("first_pullback", ("pullback",)),
    ("range_or_squeeze", ("rang", "squeeze", "position_in_range")),
    ("market_stage", ("stage",)),
    ("connors_rsi2", ("rsi2", "connors")),
    ("volume", ("volume", "poc")),
    ("candles_momentum", ("candle", "engulf", "momentum", "wick",
                          "exhaustion", "asymmetry")),
    ("adx_trend", ("adx",)),
    ("overnight_gap", ("gap",)),
    ("insider_activity", ("insider",)),
    ("news_sentiment", ("news", "headline", "sentiment")),
    ("model_signal", ("cnn", "signal", "confidence", "direction")),
    ("track_record", ("track", "past_decision", "record")),
    ("earnings", ("earnings",)),
    ("thesis_lessons", ("thesis", "lesson")),
]


def _bucket(field):
    # mapping one cited packet field onto its strategy concept
    f = str(field).lower()
    for name, needles in BUCKETS:
        if any(n in f for n in needles):
            return name
    return "other"


def _cited_buckets(case_blob, side):
    # collecting the distinct concepts one side's cases cited
    if isinstance(case_blob, str):
        try:
            case_blob = json.loads(case_blob)
        except Exception:
            return set()
    if not isinstance(case_blob, dict):
        return set()
    out = set()
    for stage in ("opening", "rebuttal"):
        for case in case_blob.get(stage) or []:
            for kp in (case or {}).get("key_points") or []:
                field = (kp or {}).get("evidence_field")
                if field:
                    out.add(_bucket(field))
    return out


def evidence_report():
    # cross-tabulating cited concepts against real outcomes of directional calls
    rows = get_client().table("decisions") \
        .select("action,bull_case,bear_case,was_correct") \
        .not_.is_("was_correct", "null") \
        .in_("action", ["BUY", "SELL"]) \
        .order("decided_at", desc=True).limit(WINDOW).execute().data or []
    if len(rows) < MIN_CITATIONS:
        print(f"evidence report: only {len(rows)} scored directional "
              f"calls — need more history before grading concepts")
        return {}

    stats = {}
    for r in rows:
        # crediting the side that argued for the action actually taken
        side = "bull_case" if r["action"] == "BUY" else "bear_case"
        for bucket in _cited_buckets(r.get(side), side):
            s = stats.setdefault(bucket, [0, 0])
            s[1] += 1
            s[0] += 1 if r["was_correct"] else 0

    base_n = len(rows)
    base_hit = sum(1 for r in rows if r["was_correct"]) / base_n
    print(f"\nevidence effectiveness — {base_n} scored directional calls, "
          f"baseline hit rate {base_hit:.0%}")
    print(f"{'concept':<24} {'cited':>6} {'hit rate':>9}  vs baseline")
    shown = 0
    for name, (hits, n) in sorted(stats.items(), key=lambda kv: -kv[1][1]):
        if n < MIN_CITATIONS:
            continue
        rate = hits / n
        edge = rate - base_hit
        print(f"{name:<24} {n:>6} {rate:>8.0%}  {edge:+.0%}")
        shown += 1
    if not shown:
        print("no concept has enough citations yet — check back next week")
    return {k: (h / n, n) for k, (h, n) in stats.items() if n >= MIN_CITATIONS}
