"""news gate — headlines veto or confirm a rule signal, never create one"""

from core.config import (NEWS_VETO_SENTIMENT, NEWS_CONFIRM_SENTIMENT,
                         NEWS_CONFIRM_BONUS)

SOURCE = ("Jansen ch3/ch13 (news sentiment as a risk overlay, not an alpha "
          "source on its own)")


def avg_sentiment(news_items, top=5):
    # averaging the sentiment of the freshest few headlines
    vals = [n.get("sentiment") for n in (news_items or [])[:top]
            if n.get("sentiment") is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def apply(signal, news_items):
    # returning an adjusted copy of the signal with a note explaining why
    sig = dict(signal)
    avg = avg_sentiment(news_items)
    sig["news_sentiment_avg"] = avg
    if sig.get("direction") != "BUY" or avg is None:
        sig["news_note"] = "no adjustment"
        return sig
    if avg <= NEWS_VETO_SENTIMENT:
        sig["direction"] = "NO_TRADE"
        sig["score"] = 0.0
        sig["news_note"] = (f"vetoed: sentiment {avg:+.2f} <= "
                            f"{NEWS_VETO_SENTIMENT}")
        return sig
    if avg >= NEWS_CONFIRM_SENTIMENT:
        sig["score"] = round(min(1.0, sig["score"] + NEWS_CONFIRM_BONUS), 4)
        sig["news_note"] = (f"confirmed: sentiment {avg:+.2f} >= "
                            f"{NEWS_CONFIRM_SENTIMENT}, +{NEWS_CONFIRM_BONUS}")
        return sig
    sig["news_note"] = f"neutral sentiment {avg:+.2f}"
    return sig
