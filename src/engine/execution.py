"""executing gated decisions on alpaca paper with code-computed risk levels"""

import os
import math
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv
from engine.memory import get_client, validate_ticker

load_dotenv()

PAPER_URL = "https://paper-api.alpaca.markets/v2"
LIVE_URL = "https://api.alpaca.markets/v2"
LIVE_CONFIRM_SENTINEL = "I_UNDERSTAND_REAL_MONEY"
RISK_PER_TRADE = 0.01        # risking one percent of equity per position
STOP_ATR_MULT = 1.5          # placing the stop this many ATRs below entry
REWARD_RISK = 2.0            # requiring two units of reward per unit of risk
MAX_POSITION_FRACTION = 0.10  # capping any position at ten percent of equity
MAX_DRAWDOWN_HALT = 0.10     # halting new entries past this peak-to-now drawdown
MAX_HOLD_DAYS = 10           # closing stale positions unless a thesis backs them


def trading_mode():
    # resolving paper or live with a double interlock guarding real money
    mode = os.environ.get("TRADING_MODE", "").lower()
    if not mode:
        mode = "paper" if os.environ.get(
            "PAPER_TRADING", "").lower() == "true" else ""
    if mode == "live" and os.environ.get(
            "LIVE_TRADING_CONFIRM") != LIVE_CONFIRM_SENTINEL:
        print("[exec] TRADING_MODE=live set without LIVE_TRADING_CONFIRM "
              "sentinel — refusing live, staying disabled")
        return ""
    return mode if mode in ("paper", "live") else ""


def base_url():
    # selecting the endpoint strictly from the resolved mode
    return LIVE_URL if trading_mode() == "live" else PAPER_URL


def enabled():
    # trading only when a valid mode and both keys are explicitly present
    return bool(trading_mode()
                and os.environ.get("ALPACA_API_KEY")
                and os.environ.get("ALPACA_SECRET_KEY"))


def _headers():
    return {"APCA-API-KEY-ID": os.environ["ALPACA_API_KEY"],
            "APCA-API-SECRET-KEY": os.environ["ALPACA_SECRET_KEY"]}


def _get(path):
    r = requests.get(f"{base_url()}{path}", headers=_headers(), timeout=20)
    r.raise_for_status()
    return r.json()


def get_account():
    # reading paper equity and buying power
    return _get("/account")


def get_positions():
    # listing open paper positions
    return _get("/positions")


def compute_levels(ticker):
    # deriving entry, atr stop, and 2r target from recent daily bars
    import yfinance as yf
    hist = yf.download(ticker.replace(".", "-"), period="2mo",
                       auto_adjust=True, progress=False)
    if hist.empty or len(hist) < 15:
        return None
    high = hist["High"].squeeze()
    low = hist["Low"].squeeze()
    close = hist["Close"].squeeze()
    prev_close = close.shift(1)
    tr = (high - low).combine((high - prev_close).abs(), max) \
        .combine((low - prev_close).abs(), max)
    atr = float(tr.rolling(14).mean().iloc[-1])
    entry = float(close.iloc[-1])
    stop = round(entry - STOP_ATR_MULT * atr, 2)
    target = round(entry + REWARD_RISK * (entry - stop), 2)
    if stop <= 0 or stop >= entry:
        return None
    return {"entry": entry, "stop": stop, "target": target,
            "atr": round(atr, 2)}


def in_drawdown_halt():
    # refusing new risk when equity has fallen too far from its peak
    res = get_client().table("portfolio_history").select("equity") \
        .order("equity", desc=True).limit(1).execute().data
    if not res:
        return False
    peak = float(res[0]["equity"])
    current = float(get_account()["equity"])
    return peak > 0 and (current / peak - 1) < -MAX_DRAWDOWN_HALT


def maybe_enter(ticker):
    # opening a long bracket position sized to risk one percent of equity
    if not enabled():
        return "paper trading disabled"
    ticker = validate_ticker(ticker)
    if in_drawdown_halt():
        return f"{ticker}: drawdown halt active — no new entries"

    # skipping when a position already exists for this symbol
    if any(p["symbol"] == ticker.replace(".", "-") for p in get_positions()):
        return f"{ticker}: position already open"

    levels = compute_levels(ticker)
    if levels is None:
        return f"{ticker}: could not compute risk levels"

    equity = float(get_account()["equity"])
    risk_dollars = equity * RISK_PER_TRADE
    per_share_risk = levels["entry"] - levels["stop"]
    qty = math.floor(risk_dollars / per_share_risk)
    max_qty = math.floor(equity * MAX_POSITION_FRACTION / levels["entry"])
    qty = min(qty, max_qty)
    if qty < 1:
        return f"{ticker}: position size below one share"

    order = {"symbol": ticker.replace(".", "-"), "qty": str(qty),
             "side": "buy", "type": "market", "time_in_force": "day",
             "order_class": "bracket",
             "take_profit": {"limit_price": str(levels["target"])},
             "stop_loss": {"stop_price": str(levels["stop"])}}
    r = requests.post(f"{base_url()}/orders", headers=_headers(), json=order,
                      timeout=20)
    r.raise_for_status()
    note = (f"{ticker}: bought {qty} @ ~{levels['entry']} "
            f"stop {levels['stop']} target {levels['target']} "
            f"(atr {levels['atr']})")
    print(f"  [paper] {note}")
    return note


def maybe_exit(ticker):
    # closing any open position when the panels vote sell
    if not enabled():
        return "paper trading disabled"
    ticker = validate_ticker(ticker)
    sym = ticker.replace(".", "-")
    if not any(p["symbol"] == sym for p in get_positions()):
        return f"{ticker}: no open position to exit"
    r = requests.delete(f"{base_url()}/positions/{sym}", headers=_headers(),
                        timeout=20)
    r.raise_for_status()
    print(f"  [paper] {ticker}: position closed on SELL vote")
    return f"{ticker}: closed"


def sync_positions_table():
    # mirroring live alpaca positions while preserving first-seen entry dates
    if not enabled():
        return
    live = get_positions()
    client = get_client()
    existing = {r["ticker"]: r for r in
                (client.table("positions").select("ticker,entry_date")
                 .eq("status", "OPEN").execute().data or [])}
    client.table("positions").update({"status": "CLOSED"}) \
        .eq("status", "OPEN").execute()
    for p in live:
        ticker = p["symbol"].replace("-", ".")
        first_seen = (existing.get(ticker) or {}).get("entry_date") \
            or datetime.now(timezone.utc).isoformat()
        client.table("positions").upsert({
            "ticker": ticker,
            "qty": float(p["qty"]),
            "entry_price": float(p["avg_entry_price"]),
            "entry_date": first_seen,
            "status": "OPEN"}).execute()


def manage_positions():
    # closing positions past the hold limit unless a thesis backs them
    if not enabled():
        return
    from engine.memory import get_active_thesis
    rows = get_client().table("positions").select("ticker,entry_date") \
        .eq("status", "OPEN").execute().data or []
    now = datetime.now(timezone.utc)
    for r in rows:
        try:
            age = (now - datetime.fromisoformat(
                r["entry_date"].replace("Z", "+00:00"))).days
        except Exception:
            continue
        if age <= MAX_HOLD_DAYS:
            continue
        thesis = get_active_thesis(r["ticker"])
        if thesis and thesis["direction"] == "LONG":
            print(f"  [paper] {r['ticker']}: {age}d old, "
                  f"held on active thesis")
            continue
        print(f"  [paper] {r['ticker']}: {age}d exceeds "
              f"{MAX_HOLD_DAYS}d limit — closing")
        maybe_exit(r["ticker"])


def paper_report():
    # summarising paper account performance for the weekly log
    if not enabled():
        print("paper trading: disabled")
        return
    acct = get_account()
    positions = get_positions()
    print(f"paper account: equity ${float(acct['equity']):,.2f} "
          f"(last ${float(acct['last_equity']):,.2f})")
    for p in positions:
        print(f"  open: {p['symbol']} x{p['qty']} "
              f"entry {p['avg_entry_price']} "
              f"unrealized {float(p['unrealized_pl']):+,.2f}")
    if not positions:
        print("  no open positions")


def is_trading_day():
    # asking alpaca whether the market opens at all today
    if not (os.environ.get("ALPACA_API_KEY")
            and os.environ.get("ALPACA_SECRET_KEY")):
        return True
    try:
        from datetime import date
        today = str(date.today())
        cal = _get(f"/calendar?start={today}&end={today}")
        return bool(cal)
    except Exception:
        return True
