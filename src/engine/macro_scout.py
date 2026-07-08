"""nominating tickers materially affected by today's macro headlines

one llm call per day reads the market-wide news and proposes at most two
s&p tickers for the debate queue — the debate itself, the judges, and the
risk gate still decide everything downstream
"""

import os
import json

MAX_MACRO_PINS = 2
_SCOUT_PROVIDERS = ["gemini", "mistral", "groq"]

PROMPT = (
    "You screen macro and geopolitical headlines for single-stock impact.\n"
    "Today's market-wide headlines (with sentiment):\n{headlines}\n\n"
    "Name at most {k} S&P 500 tickers whose business is DIRECTLY and "
    "MATERIALLY affected by these specific headlines today — supply, "
    "demand, costs, sanctions, regulation. Do not name broad-market "
    "proxies, do not stretch: if no single stock is clearly and "
    "materially affected, return an empty list.\n"
    'Respond with ONLY this JSON: {{"picks": [{{"ticker": "XOM", '
    '"why": "one short sentence tying it to a headline"}}]}}'
)


def _universe():
    # loading the current constituent list for validation
    for path in ("universe.csv",
                 os.path.join(os.environ.get("STOCK_LENS_BASE", ""),
                              "universe.csv")):
        try:
            with open(path) as f:
                return {line.strip().split(",")[0].upper()
                        for line in f if line.strip()}
        except OSError:
            continue
    return None


def macro_pins():
    # proposing validated macro-driven pins, or nothing at all
    from engine.news_fetcher import fetch_macro_news
    from engine.llm_clients import ask, parse_json_reply

    headlines = fetch_macro_news()
    if not headlines:
        return []
    text = "\n".join(f"[{i}] ({h.get('sentiment')}) {h['headline']}"
                     for i, h in enumerate(headlines))
    prompt = PROMPT.format(headlines=text, k=MAX_MACRO_PINS)

    reply = None
    for provider in _SCOUT_PROVIDERS:
        reply = ask(provider, prompt)
        if reply:
            break
    parsed = parse_json_reply(reply, ["picks"])
    if not parsed or not isinstance(parsed.get("picks"), list):
        return []

    universe = _universe()
    out = []
    for p in parsed["picks"]:
        if len(out) >= MAX_MACRO_PINS:
            break
        ticker = str((p or {}).get("ticker", "")).upper().strip()
        why = str((p or {}).get("why", ""))[:140]
        if not ticker.replace(".", "").replace("-", "").isalpha() \
                or len(ticker) > 6:
            continue
        if universe is not None and ticker not in universe:
            print(f"  [macro-scout] {ticker} not in universe — dropped")
            continue
        out.append((ticker, why))
    return out
