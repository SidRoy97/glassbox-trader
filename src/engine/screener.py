"""scanning the whole universe with the elected strategy and ranking candidates"""

import os
import random
from core.helpers import log
from core.config import STRATEGY_CASH, BARS_LOOKBACK_DAYS
from engine.market_data import load_universe, download_bars
from engine.strategies import get_strategy
from engine.strategies.common import rsi
from engine.strategy_election import get_strategy_champion

FALLBACK = ["SPY", "QQQ", "AAPL", "MSFT", "JPM"]
EXPLORE_SLOTS_DEFAULT = 2


def _price_facts(df):
    # the small set of facts the scan page and packet expect per ticker
    close = df["close"]
    vol = df["volume"] if "volume" in df.columns else None
    ratio = 1.0
    if vol is not None and vol.rolling(20).mean().iloc[-1] > 0:
        ratio = float(vol.iloc[-1] / vol.rolling(20).mean().iloc[-1])
    return {"close": round(float(close.iloc[-1]), 2),
            "return_1d": round(float(close.iloc[-1] / close.iloc[-2] - 1), 4),
            "vol_ratio": round(ratio, 2),
            "rsi": round(float(rsi(close).iloc[-1]), 1)}


def scan_universe(limit=None, bars=None):
    # scoring every ticker with the elected strategy plus interest terms
    champion = get_strategy_champion()
    if champion == STRATEGY_CASH:
        log("screener: champion is cash — scanning with the default rule "
            "for the record only")
        from core.config import STRATEGY_DEFAULT
        champion = STRATEGY_DEFAULT
    # a blended champion "a+b" ranks the universe by its first member; the
    # blend combines at the portfolio level, not the scan level
    scan_name = str(champion).split("+")[0]
    strat = get_strategy(scan_name)
    if bars is None:
        tickers = [t for t, _ in load_universe(limit=limit)]
        bars = download_bars(tickers, days=BARS_LOOKBACK_DAYS)
    results = []
    for ticker, df in bars.items():
        if len(df) < 30:
            continue
        try:
            sig = strat.signal(df)
            facts = _price_facts(df)
        except Exception as e:
            log(f"screener: {ticker} skipped ({e})")
            continue
        # scoring interest from strategy strength, abnormal move, and volume
        directional = float(sig["score"]) if sig["direction"] == "BUY" else 0.0
        move = min(abs(facts["return_1d"]) * 10, 0.5)
        volume = min(max(facts["vol_ratio"] - 1, 0) * 0.2, 0.4)
        results.append({"ticker": ticker, **facts,
                        "direction": "Up" if sig["direction"] == "BUY"
                        else "Neutral",
                        "confidence": round(float(sig["score"]), 4),
                        "strategy": scan_name,
                        "reason": sig["reason"],
                        "score": round(directional + move + volume, 4)})
    results.sort(key=lambda r: r["score"], reverse=True)
    log(f"screener[{scan_name}]: scanned {len(results)} tickers, "
        f"top: {[r['ticker'] for r in results[:5]]}")
    return results


def select_watchlist(k=5, limit=None, exclude=None):
    # picking top names plus exploration wildcards from the quiet middle
    from datetime import date as _date
    exclude = exclude or set()
    results = scan_universe(limit=limit)
    if not results:
        log("screener: falling back to the core watchlist")
        return FALLBACK, []
    eligible = [r["ticker"] for r in results if r["ticker"] not in exclude]

    slots = max(0, min(int(os.environ.get("EXPLORE_SLOTS")
                           or EXPLORE_SLOTS_DEFAULT), int(k) - 1))
    top = eligible[:int(k) - slots]

    # sampling wildcards from mid-ranked names to counter momentum bias
    pool = [t for t in eligible[max(20, int(k)):250] if t not in top]
    random.seed(str(_date.today()))
    wild = random.sample(pool, min(slots, len(pool))) if pool else []
    if wild:
        log(f"screener: exploration wildcards {wild}")
    return top + wild, results
