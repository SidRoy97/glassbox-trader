"""desktop diagnostic: reports LLM/judge coverage per stock for today's run.

reads today's decisions from supabase and shows, per ticker: how many judges
voted, which providers participated, and the verdict — so you can see at a
glance whether every stock got a full debate or some were starved.

run from the repo root:
    PYTHONPATH=$PWD/src python src/engine/check_coverage.py
"""

import os
import json
from datetime import date
from collections import Counter

from engine.memory import get_client


def _providers_in_votes(votes):
    # pulling provider/model names out of the judge_votes json, whatever
    # shape it takes (list of dicts, dict keyed by provider, etc.)
    names = []
    if isinstance(votes, str):
        try:
            votes = json.loads(votes)
        except Exception:
            return names
    if isinstance(votes, list):
        for v in votes:
            if isinstance(v, dict):
                names.append(v.get("provider") or v.get("model")
                             or v.get("judge") or "?")
    elif isinstance(votes, dict):
        names.extend(votes.keys())
    return [n for n in names if n]


def main():
    client = get_client()
    today = str(date.today())
    rows = client.table("decisions").select("*") \
        .gte("decided_at", today).execute().data or []

    if not rows:
        print(f"no decisions found for {today}")
        return

    print(f"\n=== LLM / JUDGE COVERAGE for {today} ===")
    print(f"{'ticker':14s} {'judges':>7s} {'action':>10s}  providers")
    print("-" * 70)

    provider_totals = Counter()
    judge_counts = []
    for r in sorted(rows, key=lambda x: x.get("ticker", "")):
        votes = r.get("judge_votes")
        provs = _providers_in_votes(votes)
        n = len(provs)
        judge_counts.append(n)
        for p in provs:
            provider_totals[p] += 1
        flag = "" if n >= 3 else "  <-- STARVED"
        print(f"{r.get('ticker','?'):14s} {n:>7d} "
              f"{str(r.get('action','?')):>10s}  "
              f"{', '.join(provs) if provs else '(none)'}{flag}")

    print("-" * 70)
    total = len(rows)
    full = sum(1 for c in judge_counts if c >= 3)
    print(f"stocks: {total} | full panels (>=3 judges): {full} "
          f"| starved (<3): {total - full}")
    if judge_counts:
        print(f"avg judges per stock: {sum(judge_counts)/len(judge_counts):.1f}")
    print(f"\nprovider participation (judge seats across all stocks):")
    for prov, cnt in provider_totals.most_common():
        print(f"  {prov:16s} {cnt}")

    # also report the latest strategy leaderboard (replaces the tournament)
    try:
        bt = client.table("strategy_backtests") \
            .select("run_date,strategy,utility,sharpe") \
            .order("run_date", desc=True).limit(10).execute().data or []
        latest = bt[0]["run_date"] if bt else None
        print(f"\nstrategy backtests (run {latest}):")
        for b in bt:
            if b["run_date"] == latest:
                print(f"  {b['strategy']:20s} utility {b['utility']:+.3f} "
                      f"sharpe {b['sharpe']:+.2f}")
    except Exception as e:
        print(f"(strategy_backtests check skipped: {e})")

if __name__ == "__main__":
    main()
