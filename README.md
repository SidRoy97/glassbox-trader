# glassbox trader

**Independent AI panels debate every stock decision — and show their work.**

Live site: **https://glassbox-trader.vercel.app** · Mode: **PAPER** (simulated money)

glassbox-trader is a fully autonomous trading research system. Every weekday morning it scans the
entire S&P universe, picks the most interesting names, makes three AI model families argue about
each one, lets a hard-coded risk gate have the final word, executes on a paper brokerage account,
and then grades itself against what the market actually did. Every decision, argument, vote,
mistake, and lesson is public. Nothing is hidden — that is the product.

> Educational project. Nothing here is financial advice.

---

## How a decision is made

![decision flow](docs/debate_flow.svg)

Every weekday at 12:30 UTC (pre-market ET), GitHub Actions runs the engine:

**1. Market check.** Alpaca's calendar confirms the market opens today — holidays are skipped.
A factual market snapshot (S&P 500, Nasdaq, VIX) is computed by code and stored.

**2. Screener — the whole universe.** The **elected champion model** (see the tournament below)
scans ~500 tickers in one batch. Each ticker gets an **interest score** = directional conviction
(non-Neutral confidence) + abnormal 1-day move + abnormal volume. The top 20 are recorded; most
debate slots go to the top-ranked names (minus anything debated in the last 2 days — cooldown
rotation), and **2 exploration wildcards** are sampled from the quiet mid-ranked names each day
to counter momentum bias — so unremarkable-looking stocks still get their day in court.
Roughly 30–40 unique names get debated per week.

**3. The data packet — the only permitted evidence.** For each debated ticker, code assembles
a sealed packet: the CNN signal (direction, confidence, RSI, 5/10-day returns, price vs 50-day
MA, volume ratio, sector-relative strength), days until earnings, the 5 freshest headlines with
finance-aware sentiment scores, the last 5 decisions on this ticker **with their real outcomes**,
the all-time scored record on this ticker, distilled lessons from past mistakes, any active
long-horizon thesis, current open position (if held), and the market snapshot.

**4. The debate — fixed three rounds, exactly 9 LLM calls, terminates by construction.**
- **Round 1:** the bull panel (Gemini 2.5 Flash + Llama 3.3 70B) and the bear panel
  (Mistral Small + Llama 3.3 70B) each write independent opening cases. Every claim must cite a
  packet field by name (`cnn_signal.rsi`, `news[2].headline`). Claims citing facts not in the
  packet get struck.
- **Round 2:** one rebuttal per side.
- **Round 3:** three judges (one per model family) read everything, strike ungrounded claims,
  and vote BUY / SELL / NO_TRADE independently. A **strict majority** is required; ties, missing
  votes, and malformed replies all default to NO_TRADE. There is no loop and no model-controlled
  flow — the code calls each stage once and stops.

Three different companies' models are used deliberately: their errors are decorrelated, so a
majority vote filters mistakes instead of amplifying shared bias. In live runs, judges have
struck exaggerations ("RSI 60.3 is high") and unverifiable claims made by other models.

**5. The risk gate — pure code, no LLM can override it.** The verdict passes through hard rules:
average judge confidence ≥ 0.5, max 3 trades/day, a 10% peak-to-trough **drawdown halt** that
blocks all new entries, and thesis-aware annotations (a SELL against an active LONG thesis is
flagged). The gate's word is final.

**6. Execution (paper).** A surviving BUY becomes one **bracket order** on Alpaca: entry +
stop-loss at 1.5× ATR(14) below entry + take-profit at 2× the risk distance (2:1 reward:risk).
Position size = account equity × 1% ÷ stop distance, capped at 10% of equity per position.
The broker enforces the exits 24/7. Positions also close on a SELL vote, or after 10 days
(time exit) — unless an active thesis justifies holding longer. Long-only; no shorting.

**7. Scoring — by code, never by LLM self-grading.** At 22:30 UTC the scorer compares each
decision against the **first market close after** the decision was made (never against the past;
incomplete days wait). NO_TRADE before a big move is counted as a "missed opportunity",
separately from real wrong calls — the track record page shows both, unedited.

---

## System architecture

![architecture](docs/architecture.svg)

| Component | What it does |
|---|---|
| `src/pipeline/` | The ML pipeline: features (RSI, MACD, Bollinger, lags, sector-relative), sequence models, and `retrain_cnn.py` (5-year trailing retrain behind a champion/challenger gate) |
| `src/inference/` | Live feature building from yfinance + model loading and prediction |
| `src/engine/screener.py` | Full-universe batch CNN scan and interest ranking |
| `src/engine/data_packet.py` | Assembles the sealed evidence packet |
| `src/engine/panels.py` / `protocol.py` | Prompts, grounding contract, and the fixed 3-round state machine |
| `src/engine/risk_gate.py` | Hard-coded limits — confidence floor, trade cap, thesis awareness |
| `src/engine/execution.py` | Alpaca bracket orders, ATR stops, 1%-risk sizing, drawdown halt, time exits, paper/live interlock |
| `src/engine/memory.py` | Validated Supabase layer — 11 tables, ticker regex on every entry point |
| `src/engine/shadow.py` | Records every model's daily prediction for the tournament |
| `src/engine/lessons.py` | Weekly distillation of systematic mistakes into reusable guidance |
| `src/engine/thesis.py` | Long-horizon theses with code-enforced honesty (10% adverse move auto-weakens) |
| `src/engine/performance.py` | Syncs equity curve and FIFO-matched closed trades from Alpaca |
| `src/engine/run_daily.py` | Orchestrates daily / score / weekly modes |
| `web/` | Next.js 15 dark dashboard, 9 pages, deployed on Vercel |
| `.github/workflows/` | `engine.yml` (three cron schedules) and `retrain.yml` (manual, commits winning models back) |

**Data sources and what each contributes:** yfinance (prices — the CNN's food), Finnhub
(structured news + earnings calendar), Yahoo RSS (headlines), VADER + finance lexicon
(sentiment), SPY/QQQ/VIX (regime), SPDR sector ETFs (relative strength), yfinance institutional
holders (thesis evidence — positions are facts, opinions are noise), Alpaca (execution, account
truth, market calendar).

---

## How it learns from mistakes

![learning loops](docs/learning_loops.svg)

- **Daily:** every packet shows judges the ticker's recent calls *with outcomes* and its all-time
  record. Mistakes are visible before every vote.
- **Weekly:** the lesson distiller collects wrong calls — with the judges' stated reasoning, the
  CNN signal, and the news that preceded each — and asks for at most 2 **systematic** patterns
  (not one-off bad luck), each citing its evidence cases. Surviving lessons are deduped, capped
  at 10 active, and injected into every future debate. This is the "why" layer.
- **Continuous:** a shadow tournament records cnn1d and random_forest predictions on identical
  tickers and days; code scores both; the weekly report shows the standings. Theses are
  re-examined weekly and auto-weakened if the market moves 10% against them.
- **Weekly election:** the tournament is not just a scoreboard — it **governs**. Every Saturday,
  whichever model has the better rolling hit rate (minimum 20 scored predictions, and a clear
  +5-point margin so the title cannot flip-flop) is elected champion, and from Monday the
  screener and every data packet run on the winner. Both models keep predicting in the shadow
  regardless, so a dethroned model can earn the seat back.
- **Quarterly:** the weekly report prints the CNN's live hit rate against the ~33% random
  baseline. When it sags, the retrain workflow trains a challenger on a 5-year trailing window;
  it deploys **only if it beats the champion** on untouched recent data. Old artifacts are
  archived, never destroyed.

The LLMs' weights never change. All LLM-layer learning is prompt-level — auditable (every lesson
is a readable sentence with cited evidence), reversible, and immune to the failure modes of
fine-tuning on noisy market feedback. The one component where weight-learning is appropriate —
the CNN, with clean supervised labels — is exactly the one that gets it.

---

## The models

| Role | Model | Provider | Swap via |
|---|---|---|---|
| Bull panel | Gemini 2.5 Flash + Llama 3.3 70B | Google, Groq | `BULL_PANEL`, `GEMINI_MODEL`, `GROQ_MODEL` |
| Bear panel | Mistral Small + Llama 3.3 70B | Mistral, Groq | `BEAR_PANEL`, `MISTRAL_MODEL` |
| Judges | all three families | — | `JUDGE_PANEL` |
| Thesis agent & lesson distiller | Gemini 2.5 Flash | Google | `GEMINI_MODEL` |
| Signal engine | 1D-CNN (classification head) | trained in-repo | quarterly retrain |
| Shadow challenger | Random Forest | trained in-repo | can be elected champion weekly |

The CNN was chosen empirically: a 20-configuration bake-off (LSTM, GRU, TCN, CNN, Transformer ×
regression/classification heads) where classification beat regression 0.44–0.47 vs 0.15–0.39
macro F1 and cnn1d won the held-out test at **0.4679** with balanced per-class scores. Tabular
models (RF, XGBoost, ensembles — 33 configurations) ceilinged at ~0.39–0.41. Out-of-sample the
CNN decays to ~0.35 from regime drift — which is exactly why retraining is built in and why the
LLM layer exists: prices alone don't carry event information.

---

## Operating modes

| Mode | What it means | How it's enabled |
|---|---|---|
| **RESEARCH** | Signals and debates only, no orders anywhere | default with no Alpaca keys |
| **PAPER** (current) | Simulated orders on an Alpaca paper account | `TRADING_MODE=paper` (or `PAPER_TRADING=true`) + paper keys |
| **LIVE** | Real money, own account only | double interlock: `TRADING_MODE=live` **and** `LIVE_TRADING_CONFIRM=I_UNDERSTAND_REAL_MONEY`, plus live keys |

A single stray variable can never reach real money — both live switches must be deliberately set,
and every run logs its mode and endpoint. Live trading is for the owner's account only;
executing for others is regulated investment-adviser territory.

---

## Automation schedule (GitHub Actions, UTC)

| When | Mode | What happens |
|---|---|---|
| Weekdays 12:30 | `daily` | holiday check → market snapshot → universe scan → top-10 debates → gate → paper orders → position & performance sync → stale-position management |
| Weekdays 22:30 | `score` | decisions and shadow predictions graded against the completed session |
| Saturday 14:00 | `weekly` | scoring sweep → performance report → model tournament → paper P&L → thesis review & proposals → lesson distillation → news pruning (5-year retention) → report row for the site |
| Manual | `retrain` | 5-year trailing retrain; challenger deploys and commits back only if it wins |

---

## The website

Nine pages, all reading Supabase live (public read-only under Row Level Security):
**Briefing** (market banner + today's verdict cards) · **Scan** (the full top-20 ranking, debated
names highlighted) · **Signals** (CNN calls and confidence over time) · **News** (every archived
headline with sentiment) · **Track record** (every scored call, rolling hit rate vs random
baseline, wrong vs missed counted separately) · **Performance** (paper equity vs SPY, closed
trades with realized P&L) · **Reports** (the engine's weekly self-audits) · **Insights** (theses
and lessons) · **Positions** (open paper holdings and gate interventions). Debate pages open
with a 6-month candlestick chart marking the decision moment.

---

## Setup

```
# .env at repo root (see .env.example)
GEMINI_API_KEY=          # aistudio.google.com
GROQ_API_KEY=            # console.groq.com
MISTRAL_API_KEY=         # console.mistral.ai
FINNHUB_API_KEY=         # finnhub.io
SUPABASE_URL=            # supabase project settings
SUPABASE_KEY=            # publishable key (site reads)
SUPABASE_SERVICE_KEY=    # secret key (engine writes)
ALPACA_API_KEY=          # alpaca.markets paper keys
ALPACA_SECRET_KEY=
PAPER_TRADING=true
```

1. Run the SQL files in `src/engine/` (schema, RLS, screen, model_predictions, performance,
   reports) in the Supabase SQL editor.
2. Mirror the keys as GitHub Actions secrets; add repo variables `PAPER_TRADING=true`
   (optionally `DEBATE_BUDGET`, `TRADING_MODE`).
3. Deploy `web/` on Vercel (root directory `web`) with `NEXT_PUBLIC_SUPABASE_URL`,
   `NEXT_PUBLIC_SUPABASE_ANON_KEY`, `NEXT_PUBLIC_SITE_MODE=PAPER`.
4. Local run: `export STOCK_LENS_BASE=$PWD/stock-lens-data PYTHONPATH=$PWD/src` then
   `python -m engine.run_daily --mode daily`.

Free-tier budget: ~9 LLM calls per debate; Groq's daily token cap is the binding constraint —
`DEBATE_BUDGET=10` is safe, ~15 is the edge. A per-run circuit breaker skips any provider after
three consecutive failures (the 2-of-3 judge majority still functions), and models, panels, and
budgets are all env-swappable, so provider limit changes are a variable edit, not a code change.

---

## Honest limitations (and what mitigates them)

- The signal models' out-of-sample edge is thin (CNN ~0.35 macro F1 vs 0.33 random; the RF
  ceilinged lower in testing). They are calibrated priors that discipline the debate, not
  oracles. **Mitigations:** the drift monitor, the quarterly champion/challenger retrain, and
  the weekly election that always seats the best-performing model — but note honestly that
  comparison and election *select among* models; they do not raise any model's ceiling. The
  ceiling of the price-only family is the founding argument for the news-reading LLM layer
  above it.
- The screener leans toward movers and conviction by construction. **Mitigation:** two daily
  exploration wildcards sampled from quiet mid-ranked names, so the debated population is no
  longer purely momentum-selected — though top slots still favor "interesting" days, and
  track-record stats should be read with that in mind.
- Direction accuracy is not profitability. **Mitigations:** the weekly report tracks the average
  next-day return following BUY and SELL calls (a profitability proxy available immediately),
  and the Performance page tracks real trade-level P&L against SPY — the number that ultimately
  matters. Time is the one input that cannot be engineered; months of paper trading are the test.
- Free-tier LLM limits shape the design. **Mitigations:** trimmed prompts sized to the binding
  token caps, env-swappable models/panels/budgets, graceful 2-of-3 judge degradation, and a
  per-provider circuit breaker — but provider limits change without notice, and the honest
  long-term fix is the first paid dollar.

**Disclaimer:** educational research output only — nothing here is financial advice. Every
decision shown was produced by AI models debating public data, gated by hard-coded risk rules,
executed (if at all) on a simulated account.
