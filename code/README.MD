# stock-lens — pipeline code

Single-file pipeline for the stock-lens project. All logic lives in `stock_lens.py`
as separate functions, one per stage. Every stage saves plots and printed
observations to an `observations/` folder.

## Setup

```bash
pip install -r requirements.txt
```

Set Kaggle credentials (only needed the first time data is downloaded):

```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_key
```

## Choosing where data / models / plots live

By default everything goes under `./stock-lens/`. Override with an env var:

```bash
export STOCK_LENS_BASE=/workspace/stock-lens     # e.g. on vast.ai
```

This creates:

```
stock-lens/
├── data/            # downloaded csvs + generated master/train/val/test
├── models/          # saved model pickles
└── observations/    # all plots (.png), results tables (.csv), run_log.txt
```

## Running

Run everything end to end:

```bash
python stock_lens.py --stage all
```

Or run a single stage:

```bash
python stock_lens.py --stage 1     # load + inspect raw data
python stock_lens.py --stage 2     # indicators, fundamentals, labels, split
python stock_lens.py --stage 2b    # lagged / return / relative features + horizons
python stock_lens.py --stage 3     # baseline pooled next-day classification
python stock_lens.py --stage 3b    # full experiment: 5 strategies x 3 horizons
```

Stages depend on earlier outputs (`master.csv`, `master_enhanced.csv`), so run
in order the first time, or just use `--stage all`.

## Stage 3b — the experiment

Stage 3b tests, for each of three prediction horizons (1-day, 5-day, 10-day):

- **Pooled** — one model over all tickers, five strategies:
  random forest, xgboost, logistic regression, stacked ensemble, one-vs-rest
- **Per-sector** — one model per GICS sector, macro F1 averaged across sectors
- **Per-stock** — one xgboost per ticker, macro F1 averaged across stocks

It writes `observations/s3b_experiment_results.csv` plus comparison plots, and
prints which configurations beat the next-day pooled xgboost baseline. Keep the
strategies/horizons that improve results; drop the ones that don't.

> Note: per-stock is ~1,500 model fits. Expect this to take a while — run it on
> a strong machine (vast.ai), not free Colab.