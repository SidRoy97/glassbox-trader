"""writing bull cases, bear cases, rebuttals, and judge votes"""

from engine.llm_clients import ask, parse_json_reply
from engine.data_packet import packet_to_text

GROUNDING = ("RULES: the packet's evidence_reliability table grades each evidence category A (strongest) to D (weakest) — prefer higher-grade evidence when claims conflict; measured grades come from real scored outcomes and outrank priors. any case or vote that leans primarily on C- or D-grade evidence (check the table — this currently includes cnn_signal) without independent higher-grade corroboration should be treated as weak. "
             "Every claim must cite a field from the data packet by "
             "name (e.g. cnn_signal.rsi, technical_structure.adx, "
             "news[2].headline). claims citing facts not in the packet will "
             "be struck. respond with ONLY the json object, no prose before "
             "or after.")

CASE_SCHEMA = ["stance", "key_points", "confidence"]
VOTE_SCHEMA = ["vote", "reason", "confidence"]

import os

def _panel(env, default):
    # reading panel membership from env so providers swap without code edits
    names = os.environ.get(env, default).split(",")
    return [n.strip() for n in names if n.strip() in
            ("gemini", "groq", "mistral")] or default.split(",")

BUY_PANEL = _panel("BULL_PANEL", "gemini,groq")
SELL_PANEL = _panel("BEAR_PANEL", "mistral,groq")
JUDGE_PANEL = _panel("JUDGE_PANEL", "gemini,groq,mistral")
# substitution follows the explicit priority order: deepest free
# quotas first, quota-poor providers last
from engine.llm_clients import FALLBACK_ORDER as ALL_PROVIDERS


def _seat_reply(prompt, preferred, used, schema):
    # filling one panel seat, falling back to any healthy unused provider
    order = [preferred] + [p for p in ALL_PROVIDERS if p != preferred]
    for provider in order:
        if provider in used:
            continue
        reply = parse_json_reply(ask(provider, prompt), schema)
        if reply:
            if provider != preferred:
                print(f"  [panel] {preferred} seat filled by {provider}")
            reply["provider"] = provider
            used.add(provider)
            return reply
    return None


def _case_prompt(packet, stance):
    # building the opening-argument prompt for one side
    side = "strongest case FOR buying" if stance == "bull" \
        else "strongest case AGAINST buying (bearish case)"
    return (f"You are the {stance} analyst. Using ONLY the data packet, "
            f"write the {side} this stock today.\n{GROUNDING}\n"
            f'json schema: {{"stance": "{stance}", "key_points": '
            f'[{{"claim": str, "evidence_field": str}}], '
            f'"confidence": float 0-1}}\n\n' + packet_to_text(packet))


def _rebuttal_prompt(packet, stance, opposing_cases):
    # building the one-round rebuttal prompt against the opposing side
    import json
    return (f"You are the {stance} analyst. Rebut the opposing arguments "
            f"below using ONLY the data packet.\n{GROUNDING}\n"
            f'json schema: {{"stance": "{stance}", "key_points": '
            f'[{{"claim": str, "evidence_field": str}}], '
            f'"confidence": float 0-1}}\n\nOPPOSING CASES:\n'
            + json.dumps(opposing_cases, default=str)[:1500]
            + "\n\n" + packet_to_text(packet))


def _judge_prompt(packet, bull, bear, bull_reb, bear_reb):
    # building the judge prompt over the full debate
    import json
    return ("You are an impartial judge. Read the debate and vote. Strike "
            "any claim whose evidence_field is not actually in the packet. "
            "Vote NO_TRADE when evidence is weak or balanced.\n" + GROUNDING +
            '\njson schema: {"vote": "BUY"|"SELL"|"NO_TRADE", '
            '"reason": str, "confidence": float 0-1}. '
            'Vote a direction when the stronger case would survive its '
            'rebuttal; reserve NO_TRADE for genuinely balanced or '
            'evidence-thin debates and name which evidence failed. Your '
            'confidence must reflect evidence strength — 0.5 means you '
            'learned nothing from the debate.\n\nBULL CASES:\n'
            + json.dumps(bull, default=str)[:1200] + "\nBEAR CASES:\n"
            + json.dumps(bear, default=str)[:1200] + "\nBULL REBUTTAL:\n"
            + json.dumps(bull_reb, default=str)[:800] + "\nBEAR REBUTTAL:\n"
            + json.dumps(bear_reb, default=str)[:800]
            + "\n\n" + packet_to_text(packet))


def run_cases(packet, stance, panel):
    # collecting independent opening cases with per-seat provider fallback
    prompt = _case_prompt(packet, stance)
    cases, used = [], set()
    for provider in panel:
        reply = _seat_reply(prompt, provider, used, CASE_SCHEMA)
        if reply:
            cases.append(reply)
    if not cases:
        print(f"  [panel] WARNING: no {stance} cases produced — "
              f"all providers failed this stage")
    return cases


def run_rebuttal(packet, stance, panel, opposing_cases):
    # collecting one rebuttal, falling back past a dead preferred provider
    prompt = _rebuttal_prompt(packet, stance, opposing_cases)
    reply = _seat_reply(prompt, panel[0], set(), CASE_SCHEMA)
    if not reply:
        print(f"  [panel] no {stance} rebuttal produced — "
              f"continuing without one")
    return [reply] if reply else []


def run_judges(packet, bull, bear, bull_reb, bear_reb):
    # collecting independent votes, each judge seat from a distinct provider
    # dropping both rebuttals when either side's is missing, so an outage
    # can never hand one side the last word before the vote
    if bool(bull_reb) != bool(bear_reb):
        print("  [panel] asymmetric rebuttals — judging on opening "
              "cases only")
        bull_reb, bear_reb = [], []
    prompt = _judge_prompt(packet, bull, bear, bull_reb, bear_reb)
    votes, used = [], set()
    for provider in JUDGE_PANEL:
        reply = _seat_reply(prompt, provider, used, VOTE_SCHEMA)
        if reply and reply.get("vote") in ("BUY", "SELL", "NO_TRADE"):
            votes.append(reply)
    return votes
