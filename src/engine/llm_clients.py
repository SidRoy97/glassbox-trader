"""calling the three free-tier llm providers through one uniform interface"""

import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

MAX_TOKENS = 600
TIMEOUT = 45

GEMINI_URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
              "{}:generateContent").format(os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"))
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"


def ask_gemini(prompt):
    # requesting one completion from the gemini free tier
    key = os.environ.get("GEMINI_API_KEY")
    r = requests.post(f"{GEMINI_URL}?key={key}",
                      json={"contents": [{"parts": [{"text": prompt}]}],
                            "generationConfig": {
                                "maxOutputTokens": MAX_TOKENS,
                                "temperature": 0.4,
                                "thinkingConfig": {"thinkingBudget": 0}}},
                      timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


def ask_groq(prompt):
    # requesting one completion from the groq free tier
    key = os.environ.get("GROQ_API_KEY")
    r = requests.post(GROQ_URL,
                      headers={"Authorization": f"Bearer {key}"},
                      json={"model": os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": MAX_TOKENS, "temperature": 0.4},
                      timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def ask_mistral(prompt):
    # requesting one completion from the mistral free tier
    key = os.environ.get("MISTRAL_API_KEY")
    r = requests.post(MISTRAL_URL,
                      headers={"Authorization": f"Bearer {key}"},
                      json={"model": os.environ.get("MISTRAL_MODEL", "mistral-small-latest"),
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": MAX_TOKENS, "temperature": 0.4},
                      timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


PROVIDERS = {"gemini": ask_gemini, "groq": ask_groq, "mistral": ask_mistral}


_consecutive_failures = {}
_circuit_opened_at = {}
CIRCUIT_THRESHOLD = 3        # opening the circuit after this many strikeouts
CIRCUIT_COOLDOWN = 120       # allowing a single probe call after this long


def ask(provider, prompt):
    # routing one prompt with 429-aware backoff and a half-open circuit
    # breaker that probes for recovery instead of staying dark all run
    import time
    probing = False
    if _consecutive_failures.get(provider, 0) >= CIRCUIT_THRESHOLD:
        if time.time() - _circuit_opened_at.get(provider, 0) < CIRCUIT_COOLDOWN:
            print(f"  [llm] {provider} circuit open — skipping")
            return None
        probing = True
        print(f"  [llm] {provider} circuit half-open — probing for recovery")

    attempts = (1,) if probing else (1, 2, 3)
    for attempt in attempts:
        try:
            reply = PROVIDERS[provider](prompt)
            if _consecutive_failures.get(provider, 0) >= CIRCUIT_THRESHOLD:
                print(f"  [llm] {provider} recovered — circuit closed")
            _consecutive_failures[provider] = 0
            return reply
        except Exception as e:
            status = getattr(getattr(e, "response", None),
                             "status_code", None)
            print(f"  [llm] {provider} attempt {attempt} failed: {e}")
            if attempt < attempts[-1]:
                # waiting out a rate-limit window instead of burning strikes
                time.sleep(20 if status == 429 else 4)
    _consecutive_failures[provider] = \
        _consecutive_failures.get(provider, 0) + 1
    _circuit_opened_at[provider] = time.time()
    return None
