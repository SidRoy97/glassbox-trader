"""retraining the sequence model on recent data behind a challenger gate"""

import os
import shutil
import pickle
import argparse
from datetime import date, timedelta
import numpy as np
import pandas as pd
from core.config import (MODEL_PATH, BASE_FEATURE_COLS, FUND_FILL_COLS,
                         LAG_COLS, LAG_DAYS, RETURN_FEATURES,
                         RELATIVE_FEATURES, SEQ_WINDOW, SEQ_THRESHOLD)
from core.helpers import log, section
from pipeline.oos_evaluation import prepare_oos_frame
from pipeline.sequence_models import (build_sequences, train_eval_seq,
                                      score_seq_model)
from inference.predictors import load_seq_predictor

TRAIL_YEARS = 5          # training on this many trailing years of data
EVAL_DAYS = 60           # holding out this many recent days for the gate
WARMUP_DAYS = 120        # padding the download so indicators have history


def tech_feature_cols(df):
    # selecting live-available features, dropping dataset-only fundamentals
    cols = [c for c in BASE_FEATURE_COLS if c not in FUND_FILL_COLS] \
        + [f"{c}_lag{l}" for c in LAG_COLS for l in LAG_DAYS] \
        + RETURN_FEATURES + RELATIVE_FEATURES
    return [c for c in cols if c in df.columns]


def score_champion(champion, eval_df):
    # scoring the currently deployed model on the held-out recent window
    if champion is None:
        return None
    meta = champion["meta"]
    df = eval_df.copy()
    means = dict(zip(meta["feature_cols"], champion["scaler"].mean_))
    for c in meta["feature_cols"]:
        df[c] = df[c].fillna(means[c]) if c in df.columns else means[c]
    df[meta["feature_cols"]] = champion["scaler"].transform(
        df[meta["feature_cols"]])
    X, _, y = build_sequences(df, meta["feature_cols"],
                              window=meta["window"])
    if X is None:
        return None
    f1, _ = score_seq_model(champion["model"], meta["head"], X, y)
    return f1


def retrain(limit=None):
    section("RETRAINING SEQUENCE MODEL ON RECENT DATA")
    from sklearn.preprocessing import StandardScaler

    # downloading the trailing window plus indicator warmup
    end = date.today()
    start = end - timedelta(days=int(TRAIL_YEARS * 365 + WARMUP_DAYS))
    df = prepare_oos_frame(limit=limit, start=str(start), end=str(end))
    if df is None or df.empty:
        log("download failed — aborting retrain")
        return
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    feature_cols = tech_feature_cols(df)
    log(f"rows: {df.shape[0]:,} | live-available features: {len(feature_cols)}")

    # holding out the most recent trading days for the challenger gate
    dates = np.sort(df["date"].unique())
    eval_start = dates[-EVAL_DAYS]
    fit_df = df[df["date"] < eval_start]
    eval_df = df[df["date"] >= eval_start].copy()

    # splitting fit data chronologically for training and validation
    fit_dates = np.sort(fit_df["date"].unique())
    val_start = fit_dates[int(len(fit_dates) * 0.85)]
    train = fit_df[fit_df["date"] < val_start].copy()
    val = fit_df[fit_df["date"] >= val_start].copy()
    log(f"train days: {len(fit_dates[fit_dates < val_start])} | "
        f"val days: {len(fit_dates[fit_dates >= val_start])} | "
        f"eval days: {EVAL_DAYS}")

    # scaling on training rows only, then training the challenger
    scaler = StandardScaler().fit(train[feature_cols])
    for part in (train, val):
        part[feature_cols] = scaler.transform(part[feature_cols])
    Xtr, rtr, ctr = build_sequences(train, feature_cols)
    Xva, _, cva = build_sequences(val, feature_cols)
    log(f"training challenger cnn1d on {len(Xtr):,} sequences...")
    val_f1, challenger = train_eval_seq("cnn1d", "classification",
                                        Xtr, rtr, ctr, Xva, cva,
                                        return_model=True)
    log(f"challenger validation macro_f1 = {val_f1:.4f}")

    # scoring both models once on the untouched recent window
    eval_scaled = eval_df.copy()
    eval_scaled[feature_cols] = scaler.transform(eval_scaled[feature_cols])
    Xe, _, ye = build_sequences(eval_scaled, feature_cols)
    chal_f1, _ = score_seq_model(challenger, "classification", Xe, ye)
    champion = load_seq_predictor()
    champ_f1 = score_champion(champion, eval_df)
    log(f"[GATE] challenger eval macro_f1 = {chal_f1:.4f}")
    log(f"[GATE] champion   eval macro_f1 = "
        f"{champ_f1:.4f}" if champ_f1 is not None else
        "[GATE] no champion found — challenger deploys by default")

    # keeping the champion when it still wins on recent data
    if champ_f1 is not None and champ_f1 >= chal_f1:
        log("champion retained — no artifact changes made")
        return

    # archiving old artifacts before deploying the challenger
    import torch
    archive = os.path.join(MODEL_PATH, "archive", str(date.today()))
    os.makedirs(archive, exist_ok=True)
    for fn in ("seq_model.pt", "seq_meta.pkl", "seq_scaler.pkl"):
        src_path = os.path.join(MODEL_PATH, fn)
        if os.path.exists(src_path):
            shutil.copy2(src_path, os.path.join(archive, fn))
    log(f"old artifacts archived to {archive}")

    # refitting the challenger on all pre-eval data for deployment
    deploy_scaler = StandardScaler().fit(fit_df[feature_cols])
    fit_scaled = fit_df.copy()
    fit_scaled[feature_cols] = deploy_scaler.transform(fit_scaled[feature_cols])
    aX, ar, ac = build_sequences(fit_scaled, feature_cols)
    _, deploy_model = train_eval_seq("cnn1d", "classification",
                                     aX, ar, ac, aX, ac, return_model=True)
    torch.save(deploy_model.state_dict(),
               os.path.join(MODEL_PATH, "seq_model.pt"))
    with open(os.path.join(MODEL_PATH, "seq_meta.pkl"), "wb") as f:
        pickle.dump({"kind": "cnn1d", "head": "classification",
                     "feature_cols": feature_cols, "window": SEQ_WINDOW,
                     "threshold": SEQ_THRESHOLD,
                     "n_features": len(feature_cols),
                     "classes": ["Down", "Neutral", "Up"],
                     "trained_through": str(eval_start)[:10]}, f)
    with open(os.path.join(MODEL_PATH, "seq_scaler.pkl"), "wb") as f:
        pickle.dump(deploy_scaler, f)
    log("challenger deployed — seq_model.pt, seq_meta.pkl, seq_scaler.pkl replaced")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", type=int, default=None,
                        help="limit tickers for a faster run")
    args = parser.parse_args()
    retrain(limit=args.tickers)


if __name__ == "__main__":
    main()
