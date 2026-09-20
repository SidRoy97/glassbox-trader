"""serving the live rule-strategy signal over HTTP"""

from fastapi import FastAPI, HTTPException
from engine.strategies import REGISTRY

app = FastAPI(title="glassbox-trader signal API",
              description="Rule-strategy reads on live yfinance data. "
                          "Educational only.")


@app.get("/health")
def health():
    # reporting the registered strategies and the elected one
    from engine.strategy_election import get_strategy_champion
    return {"status": "ok", "strategies": sorted(REGISTRY),
            "champion": get_strategy_champion()}


@app.get("/signal")
def signal(ticker: str):
    # returning the elected strategy's read plus every strategy's vote
    from engine.data_packet import get_strategy_signal
    ticker = ticker.upper().strip()
    sig = get_strategy_signal(ticker)
    if sig.get("direction") == "unavailable":
        raise HTTPException(404, f"no live data found for {ticker}")
    return {"ticker": ticker, **sig,
            "disclaimer": "educational output, not financial advice"}
