"""fetching per-ticker news from yahoo rss and finnhub, then archiving it"""

import os
import requests
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from engine.memory import insert_news, validate_ticker

load_dotenv()

YAHOO_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}"
FINNHUB_NEWS = "https://finnhub.io/api/v1/company-news"


def fetch_yahoo_rss(ticker):
    # pulling the latest headlines from the free yahoo rss feed
    import feedparser
    ticker = validate_ticker(ticker)
    feed = feedparser.parse(YAHOO_RSS.format(ticker=ticker))
    items = []
    for e in feed.entries[:15]:
        items.append({"ticker": ticker,
                      "published_at": getattr(e, "published", None),
                      "source": "yahoo_rss",
                      "headline": getattr(e, "title", "")[:500],
                      "summary": getattr(e, "summary", "")[:2000],
                      "url": getattr(e, "link", None)})
    return items


def fetch_finnhub(ticker, days_back=3):
    # pulling structured company news from the finnhub free tier
    ticker = validate_ticker(ticker)
    key = os.environ.get("FINNHUB_API_KEY")
    if not key:
        return []
    to_d = datetime.now(timezone.utc).date()
    from_d = to_d - timedelta(days=days_back)
    try:
        r = requests.get(FINNHUB_NEWS,
                         params={"symbol": ticker, "from": str(from_d),
                                 "to": str(to_d), "token": key},
                         timeout=15)
        r.raise_for_status()
        items = []
        for a in r.json()[:15]:
            items.append({"ticker": ticker,
                          "published_at": datetime.fromtimestamp(
                              a.get("datetime", 0),
                              tz=timezone.utc).isoformat(),
                          "source": str(a.get("source", "finnhub"))[:100],
                          "headline": str(a.get("headline", ""))[:500],
                          "summary": str(a.get("summary", ""))[:2000],
                          "url": a.get("url")})
        return items
    except Exception:
        return []


def dedupe(items):
    # dropping items whose headlines are near-duplicates of earlier ones
    seen, out = set(), []
    for it in items:
        key = it["headline"].lower().strip()[:80]
        if key and key not in seen:
            seen.add(key)
            out.append(it)
    return out


def fetch_and_archive(ticker, top_n=5):
    # combining both sources, archiving everything, returning the freshest few
    items = dedupe(fetch_yahoo_rss(ticker) + fetch_finnhub(ticker))
    for it in items:
        insert_news(**it)
    return items[:top_n]
