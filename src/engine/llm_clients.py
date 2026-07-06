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


def ask(provider, prompt):
    # routing one prompt with one retry and visible failure logging
    import time
    for attempt in (1, 2):
        try:
            return PROVIDERS[provider](prompt)
        except Exception as e:
            print(f"  [llm] {provider} attempt {attempt} failed: {e}")
            time.sleep(3)
    return None


def parse_json_reply(text, required_keys):
    # extracting and validating strict json from an llm reply
    if not text:
        return None
    try:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            return None
        obj = json.loads(text[start:end + 1])
        if not all(k in obj for k in required_keys):
            return None
        return obj
    except (json.JSONDecodeError, TypeError):
        return None
