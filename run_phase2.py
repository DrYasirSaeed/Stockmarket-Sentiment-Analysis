"""
Project: Information Transmission and Sectoral Asymmetry in the KSE-100
Module : Phase 2 Orchestrator — Topic Routing + FinBERT + PSX Sign Rules
Author : Dr. Yasir Saeed
Affiliation: Kohat University of Science & Technology (KUST)
Description: Runs the full Phase 2 pipeline on the relevance-filtered corpus:

    relevant_articles.csv  (Phase 1a/1b output, 158k articles)
        |
        v
    1. topic_router.classify_corpus     -> Monetary | Fiscal | External | Energy
    2. FinBERT scoring (batched, resumable checkpoints)
    3. psx_sign_rules.apply_sign_rules  -> psx_score (direction for KSE-100)
    4. Daily value-weighted aggregation -> S_Monetary ... S_Energy
        |
        v
    Phase 2 Outputs/
        classified_articles.csv         (step 1 checkpoint)
        finbert_scores.csv              (step 2 checkpoint — appended in batches,
                                         safe to interrupt and re-run)
        scored_articles_psx.csv         (article-level, all columns)
        sign_rules_audit.xlsx           (rule frequency + validation samples)
        daily_sentiment.xlsx            (PSX-directional series -> Phase 3 input)
        daily_sentiment_finbert_raw.xlsx(raw FinBERT series — robustness check)
        sentiment_series.png

USAGE
    python run_phase2.py                # full corpus (resumes automatically)
    python run_phase2.py --sample 500   # smoke test on a random sample
    python run_phase2.py --status       # show checkpoint progress and exit

NOTES
    - FinBERT input = headline + first 2,000 chars of body (the headline is
      included because it carries the primary editorial signal).
    - Scoring uses GPU automatically if a CUDA build of torch is installed;
      on CPU expect roughly 4-8 hours for the full corpus. Interrupting is
      safe: already-scored articles are skipped on the next run.
"""

import matplotlib
matplotlib.use("Agg")          # save plots without opening windows

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
PROJECT_DIR   = Path(r"D:\Stockmarket-Sentiment-Analysis")
# Relevance-filtered corpus from keyword_relevance_filter.py (Phase 1b).
# Already carries topic_category + conf_* columns, so Step 1 is skipped.
INPUT_CSV     = PROJECT_DIR / "Phase 1 Outputs" / "relevant_articles_keyword.csv"
OUT_DIR       = PROJECT_DIR / "Phase 2 Outputs"

CLASSIFIED_CSV   = OUT_DIR / "classified_articles.csv"
FINBERT_CSV      = OUT_DIR / "finbert_scores.csv"        # resumable checkpoint
SCORED_CSV       = OUT_DIR / "scored_articles_psx.csv"
AUDIT_XLSX       = OUT_DIR / "sign_rules_audit.xlsx"
DAILY_XLSX       = OUT_DIR / "daily_sentiment.xlsx"
DAILY_RAW_XLSX   = OUT_DIR / "daily_sentiment_finbert_raw.xlsx"
PLOT_PNG         = OUT_DIR / "sentiment_series.png"

BATCH_SIZE       = 32       # FinBERT inference batch
CHECKPOINT_EVERY = 1000     # append scores to disk every N articles
MAX_LENGTH       = 256      # token window: headline + lead paragraph carries
                            # the news signal; 256 halves compute vs 512
BODY_CHARS       = 2000     # chars of body text passed to FinBERT
USE_FP16         = True     # half precision on GPU (Turing+); fp32 on CPU

sys.path.insert(0, str(PROJECT_DIR))
from topic_router import classify_corpus                     # noqa: E402
from psx_sign_rules import apply_sign_rules, export_audit    # noqa: E402
from finbert_scorer import (aggregate_daily_sentiment,       # noqa: E402
                            plot_sentiment_series,
                            save_sentiment_series)


def parse_args():
    p = argparse.ArgumentParser(description="Phase 2: FinBERT + PSX sign rules")
    p.add_argument("--sample", type=int, default=0,
                   help="Run on a random sample of N articles (smoke test; "
                        "outputs get a _sample suffix, checkpoints not reused)")
    p.add_argument("--status", action="store_true",
                   help="Show checkpoint progress and exit")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Step 2: FinBERT scoring (batched + resumable)
# ---------------------------------------------------------------------------
def load_finbert_pipeline():
    import torch
    from transformers import pipeline
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if (USE_FP16 and device != "cpu") else torch.float32
    print(f"\n--- Loading FinBERT (ProsusAI/finbert) on {device} ({dtype}) ---")
    return pipeline("text-classification",
                    model="ProsusAI/finbert",
                    tokenizer="ProsusAI/finbert",
                    device=device,
                    torch_dtype=dtype)


def signed(label: str, conf: float) -> float:
    label = label.lower()
    if label == "positive":
        return +conf
    if label == "negative":
        return -conf
    return 0.0


def score_finbert_resumable(df: pd.DataFrame, checkpoint_csv: Path) -> pd.DataFrame:
    """
    Score all articles, appending results to checkpoint_csv every
    CHECKPOINT_EVERY articles. On re-run, already-scored article_ids
    are skipped. Returns the full score table (article_id + 3 columns).
    """
    done = pd.DataFrame(columns=["article_id", "sentiment_label",
                                 "sentiment_conf", "sentiment_score"])
    if checkpoint_csv.exists():
        done = pd.read_csv(checkpoint_csv)
        print(f"  Checkpoint found: {len(done):,} articles already scored")

    todo = df[~df["article_id"].isin(set(done["article_id"]))]
    if todo.empty:
        print("  Nothing to score — checkpoint is complete.")
        return done

    print(f"  To score: {len(todo):,} articles "
          f"(batch={BATCH_SIZE}, checkpoint every {CHECKPOINT_EVERY})")

    finbert = load_finbert_pipeline()

    texts = (todo["headline"].fillna("").astype(str) + ". " +
             todo["body_text"].fillna("").astype(str).str.slice(0, BODY_CHARS)).tolist()
    ids = todo["article_id"].tolist()

    buffer, t0, n_done = [], time.time(), 0
    write_header = not checkpoint_csv.exists()

    for start in range(0, len(texts), CHECKPOINT_EVERY):
        chunk_texts = texts[start:start + CHECKPOINT_EVERY]
        chunk_ids   = ids[start:start + CHECKPOINT_EVERY]

        results = finbert(chunk_texts, batch_size=BATCH_SIZE,
                          truncation=True, max_length=MAX_LENGTH, top_k=1)

        for aid, res in zip(chunk_ids, results):
            r = res[0] if isinstance(res, list) else res
            label, conf = r["label"].lower(), float(r["score"])
            buffer.append({"article_id": aid,
                           "sentiment_label": label,
                           "sentiment_conf": round(conf, 4),
                           "sentiment_score": round(signed(label, conf), 4)})

        pd.DataFrame(buffer).to_csv(checkpoint_csv, mode="a",
                                    header=write_header, index=False)
        write_header = False
        n_done += len(buffer)
        buffer = []

        rate = n_done / (time.time() - t0)
        remaining = (len(texts) - n_done) / rate if rate > 0 else float("inf")
        print(f"  Scored {n_done:,} / {len(texts):,} new articles "
              f"({rate:.1f}/s, ~{remaining / 3600:.1f}h remaining)")

    return pd.read_csv(checkpoint_csv)


# ---------------------------------------------------------------------------
# STATUS
# ---------------------------------------------------------------------------
def status():
    print("\n=== Phase 2 status ===")
    if INPUT_CSV.exists():
        n_total = sum(1 for _ in open(INPUT_CSV, encoding="utf-8", errors="ignore")) - 1
        print(f"  [OK] input corpus            ~{n_total:,} rows   {INPUT_CSV}")
    else:
        print(f"  [MISSING] input corpus       {INPUT_CSV}")
    for label, path in [("classified_articles.csv", CLASSIFIED_CSV),
                        ("finbert_scores.csv     ", FINBERT_CSV),
                        ("scored_articles_psx.csv", SCORED_CSV),
                        ("daily_sentiment.xlsx   ", DAILY_XLSX)]:
        if path.exists():
            try:
                n = sum(1 for _ in open(path, encoding="utf-8", errors="ignore")) - 1
                print(f"  [OK] {label}  {n:,} rows")
            except Exception:
                print(f"  [OK] {label}")
        else:
            print(f"  [--] {label}  not yet created")
    print()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    if args.status:
        status()
        return

    OUT_DIR.mkdir(exist_ok=True)
    sample_mode = args.sample > 0
    suffix = "_sample" if sample_mode else ""

    classified_csv = OUT_DIR / f"classified_articles{suffix}.csv"
    finbert_csv    = OUT_DIR / f"finbert_scores{suffix}.csv"
    scored_csv     = OUT_DIR / f"scored_articles_psx{suffix}.csv"
    audit_xlsx     = OUT_DIR / f"sign_rules_audit{suffix}.xlsx"
    daily_xlsx     = OUT_DIR / f"daily_sentiment{suffix}.xlsx"
    daily_raw_xlsx = OUT_DIR / f"daily_sentiment_finbert_raw{suffix}.xlsx"
    plot_png       = OUT_DIR / f"sentiment_series{suffix}.png"

    # ---- Load corpus ------------------------------------------------------
    print(f"\n--- Loading corpus: {INPUT_CSV} ---")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    df = df.drop_duplicates(subset="article_id")
    print(f"  {len(df):,} unique articles")

    if sample_mode:
        df = df.sample(args.sample, random_state=42).reset_index(drop=True)
        print(f"  SAMPLE MODE: {len(df):,} articles (outputs suffixed '{suffix}')")
        for p in (classified_csv, finbert_csv):   # fresh sample checkpoints
            p.unlink(missing_ok=True)

    # ---- Step 1: topic routing --------------------------------------------
    if "topic_category" in df.columns:
        # Input already classified by keyword_relevance_filter.py
        print("\n--- Step 1: topic_category present in input — skipping ---")
        classified = df
        if not classified_csv.exists():
            classified.to_csv(classified_csv, index=False)
    elif classified_csv.exists():
        classified = pd.read_csv(classified_csv, low_memory=False)
        if set(df["article_id"]) <= set(classified["article_id"]):
            print(f"\n--- Step 1: reusing {classified_csv.name} "
                  f"({len(classified):,} rows) ---")
        else:
            classified = None
    else:
        classified = None

    if classified is None:
        print("\n--- Step 1: topic routing (keyword classifier) ---")
        classified = classify_corpus(df)
        classified.to_csv(classified_csv, index=False)
        print(f"  Saved -> {classified_csv}")

    # ---- Step 2: FinBERT scoring ------------------------------------------
    print("\n--- Step 2: FinBERT sentiment scoring ---")
    scores = score_finbert_resumable(classified, finbert_csv)

    merged = classified.merge(scores, on="article_id", how="inner")
    print(f"  Merged: {len(merged):,} classified + scored articles")

    # ---- Step 3: PSX sign rules -------------------------------------------
    print("\n--- Step 3: PSX directional sign rules ---")
    merged = apply_sign_rules(merged)
    merged.to_csv(scored_csv, index=False)
    print(f"  Saved -> {scored_csv}")
    export_audit(merged, str(audit_xlsx))

    # ---- Step 4: daily aggregation ----------------------------------------
    print("\n--- Step 4: daily aggregation ---")
    daily_psx = aggregate_daily_sentiment(merged.copy(), score_col="psx_score")
    save_sentiment_series(daily_psx, output_path=str(daily_xlsx))

    daily_raw = aggregate_daily_sentiment(merged.copy(), score_col="sentiment_score")
    save_sentiment_series(daily_raw, output_path=str(daily_raw_xlsx))

    if not daily_psx.empty:
        plot_sentiment_series(daily_psx, output_path=str(plot_png))

    print("\n=== Phase 2 complete ===")
    print(f"  Panel VAR input (PSX-directional): {daily_xlsx}")
    print(f"  Robustness series (raw FinBERT)  : {daily_raw_xlsx}")
    print(f"  Validate the rules via           : {audit_xlsx}")


if __name__ == "__main__":
    main()
