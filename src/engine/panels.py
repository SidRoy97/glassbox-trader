"""writing bull cases, bear cases, rebuttals, and judge votes"""

from engine.llm_clients import ask, parse_json_reply
from engine.data_packet import packet_to_text

GROUNDING = ("RULES: every claim must cite a field from the data packet by "
             "name (e.g. cnn_signal.rsi, news[2].headline). claims citing "
             "facts not in the packet will be struck. respond with ONLY the "
             "json object, no prose before or after.")

CASE_SCHEMA = ["stance", "key_points", "confidence"]
VOTE_SCHEMA = ["vote", "reason", "confidence"]

BUY_PANEL = ["gemini", "groq"]
SELL_PANEL = ["mistral", "groq"]
JUDGE_PANEL = ["gemini", "groq", "mistral"]


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
            + json.dumps(opposing_cases, default=str)[:2500]
            + "\n\n" + packet_to_text(packet))


def _judge_prompt(packet, bull, bear, bull_reb, bear_reb):
    # building the judge prompt over the full debate
    import json
    return ("You are an impartial judge. Read the debate and vote. Strike "
            "any claim whose evidence_field is not actually in the packet. "
            "Vote NO_TRADE when evidence is weak or balanced.\n" + GROUNDING +
            '\njson schema: {"vote": "BUY"|"SELL"|"NO_TRADE", '
            '"reason": str, "confidence": float 0-1}\n\nBULL CASES:\n'
            + json.dumps(bull, default=str)[:2000] + "\nBEAR CASES:\n"
            + json.dumps(bear, default=str)[:2000] + "\nBULL REBUTTAL:\n"
            + json.dumps(bull_reb, default=str)[:1200] + "\nBEAR REBUTTAL:\n"
            + json.dumps(bear_reb, default=str)[:1200]
            + "\n\n" + packet_to_text(packet))


def run_cases(packet, stance, panel):
    # collecting independent opening cases from one panel
    prompt = _case_prompt(packet, stance)
    cases = []
    for provider in panel:
        reply = parse_json_reply(ask(provider, prompt), CASE_SCHEMA)
        if reply:
            reply["provider"] = provider
            cases.append(reply)
    return cases


def run_rebuttal(packet, stance, panel, opposing_cases):
    # collecting one rebuttal from the first responsive panel member
    prompt = _rebuttal_prompt(packet, stance, opposing_cases)
    for provider in panel[:1]:
        reply = parse_json_reply(ask(provider, prompt), CASE_SCHEMA)
        if reply:
            reply["provider"] = provider
            return [reply]
    return []


def run_judges(packet, bull, bear, bull_reb, bear_reb):
    # collecting independent votes from every judge
    prompt = _judge_prompt(packet, bull, bear, bull_reb, bear_reb)
    votes = []
    for provider in JUDGE_PANEL:
        reply = parse_json_reply(ask(provider, prompt), VOTE_SCHEMA)
        if reply and reply.get("vote") in ("BUY", "SELL", "NO_TRADE"):
            reply["provider"] = provider
            votes.append(reply)
    return votes
