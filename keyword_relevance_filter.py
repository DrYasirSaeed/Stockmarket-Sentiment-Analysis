"""
Project: Information Transmission and Sectoral Asymmetry in the KSE-100
Module : Phase 1b — Keyword Relevance Filter (replaces BERTopic keep/discard)
Author : Dr. Yasir Saeed
Affiliation: Kohat University of Science & Technology (KUST)
Description: The BERTopic-based relevance split (bertopic_filter.py) proved
             unreliable: topic-level keep/discard labels were internally
             inconsistent, the 33k-article outlier bucket was hard-discarded,
             and the topic model covered only a 100k subsample of the corpus.

             This module replaces it with a transparent ARTICLE-LEVEL rule:
             an article is PSX-relevant if and only if it classifies into one
             of the four macroeconomic policy channels of topic_router.py
             (Monetary | Fiscal | External | Energy — or Mixed) with at least
             MIN_TOTAL_HITS weighted keyword hits.

             The classifier is a vectorised re-implementation of
             topic_router.classify_article (identical scoring: substring
             counts, headline weighted 3x body, 0.35 dominance threshold) and
             is verified against the original function on a random sample at
             every run.

INPUT   Extracted Data/*.csv   (deduplicated monthly scrape files)
OUTPUT  Phase 1 Outputs/
          relevant_articles_keyword.csv   -> input to run_phase2.py
          keyword_filter_audit.xlsx       -> category counts, hit distribution,
                                             validation samples
USAGE
  python keyword_relevance_filter.py
  python keyword_relevance_filter.py --min-hits 3     (default 3)
"""

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(r"D:\Stockmarket-Sentiment-Analysis")
DATA_DIR    = PROJECT_DIR / "Extracted Data"
OUT_DIR     = PROJECT_DIR / "Phase 1 Outputs"

RELEVANT_CSV = OUT_DIR / "relevant_articles_keyword.csv"
AUDIT_XLSX   = OUT_DIR / "keyword_filter_audit.xlsx"

# Minimum weighted keyword hits (headline hit = 3, body hit = 1) for relevance.
# 3 means: at least one headline keyword, or three body mentions.
MIN_TOTAL_HITS_DEFAULT = 3

CHUNK_ROWS = 25_000          # keyword counting chunk size (memory control)

sys.path.insert(0, str(PROJECT_DIR))
from topic_router import (MONETARY_KEYWORDS, FISCAL_KEYWORDS,      # noqa: E402
                          EXTERNAL_FINANCE_KEYWORDS, ENERGY_KEYWORDS,
                          classify_article)

CATEGORIES = {
    "Monetary": MONETARY_KEYWORDS,
    "Fiscal":   FISCAL_KEYWORDS,
    "External": EXTERNAL_FINANCE_KEYWORDS,
    "Energy":   ENERGY_KEYWORDS,
}


# ---------------------------------------------------------------------------
# CORPUS LOADING
# ---------------------------------------------------------------------------
def normalise_dates(dates: pd.Series) -> pd.Series:
    """
    The scraper wrote two date formats across monthly files:
    ISO 'YYYY-MM-DD' (majority) and day-first 'DD-MM-YYYY' (some 2020-21
    files; verified day-first because the first field exceeds 12 while the
    second never does). Normalise everything to ISO strings so downstream
    pd.to_datetime cannot silently swap day and month.
    """
    s = dates.astype(str)
    iso = s.str.match(r"^\d{4}-\d{2}-\d{2}$")
    dayfirst = s.str.match(r"^\d{2}-\d{2}-\d{4}$")
    out = s.copy()
    out[dayfirst] = pd.to_datetime(
        s[dayfirst], format="%d-%m-%Y", errors="coerce").dt.strftime("%Y-%m-%d")
    bad = (~iso & ~dayfirst) | out.isna()
    if bad.any():
        print(f"  WARNING: {bad.sum():,} unparseable dates left as-is")
        out[bad] = s[bad]
    return out


def load_corpus() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if not files:
        sys.exit(f"ERROR: no CSVs found in {DATA_DIR}")
    print(f"\n--- Loading {len(files)} monthly CSVs from {DATA_DIR} ---")

    usecols = ["article_id", "url", "headline", "body_text", "date", "http_status"]
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f, usecols=lambda c: c in usecols,
                                      low_memory=False, on_bad_lines="skip"))
        except Exception as e:
            print(f"  WARNING: could not read {f}: {e}")
    df = pd.concat(frames, ignore_index=True)

    if "http_status" in df.columns:
        df = df[df["http_status"] == 200]
    df = df.drop_duplicates(subset="article_id").reset_index(drop=True)
    df["headline"]  = df["headline"].fillna("").astype(str)
    df["body_text"] = df["body_text"].fillna("").astype(str)
    df["date"]      = normalise_dates(df["date"])
    print(f"  Corpus: {len(df):,} unique articles "
          f"({df['date'].min()} -> {df['date'].max()})")
    return df


# ---------------------------------------------------------------------------
# VECTORISED CLASSIFIER (identical semantics to topic_router.classify_article)
# ---------------------------------------------------------------------------
def category_hit_counts(series_lower: pd.Series, keywords: list) -> np.ndarray:
    """Sum of substring occurrence counts across all keywords (per row)."""
    total = np.zeros(len(series_lower), dtype=np.int64)
    for kw in keywords:
        total += series_lower.str.count(re.escape(kw)).to_numpy()
    return total


def classify_vectorised(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorised replica of topic_router.classify_article:
      score[cat] = headline_hits * 3 + body_hits   (substring, case-insensitive)
      Unclassified if top score == 0
      Mixed        if top confidence < 0.35
      else the top category
    Adds: topic_category, conf_monetary/.../conf_energy, total_keyword_hits.
    """
    n = len(df)
    scores = {cat: np.zeros(n, dtype=np.int64) for cat in CATEGORIES}

    print(f"\n--- Classifying {n:,} articles "
          f"(chunks of {CHUNK_ROWS:,}) ---")
    for start in range(0, n, CHUNK_ROWS):
        end = min(start + CHUNK_ROWS, n)
        h = df["headline"].iloc[start:end].str.lower()
        b = df["body_text"].iloc[start:end].str.lower()
        for cat, kws in CATEGORIES.items():
            scores[cat][start:end] = (category_hit_counts(h, kws) * 3 +
                                      category_hit_counts(b, kws))
        print(f"  {end:,} / {n:,} classified ...")

    score_mat = np.column_stack([scores[c] for c in CATEGORIES])
    total     = score_mat.sum(axis=1)
    top_idx   = score_mat.argmax(axis=1)
    top_score = score_mat.max(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        conf_mat = np.where(total[:, None] > 0,
                            score_mat / np.maximum(total[:, None], 1), 0.0)
    top_conf = conf_mat[np.arange(n), top_idx]

    cat_names = np.array(list(CATEGORIES.keys()))
    category  = cat_names[top_idx].astype(object)
    category[top_score == 0] = "Unclassified"
    category[(top_score > 0) & (top_conf < 0.35)] = "Mixed"

    out = df.copy()
    out["topic_category"]     = category
    out["conf_monetary"]      = conf_mat[:, 0].round(3)
    out["conf_fiscal"]        = conf_mat[:, 1].round(3)
    out["conf_external"]      = conf_mat[:, 2].round(3)
    out["conf_energy"]        = conf_mat[:, 3].round(3)
    out["total_keyword_hits"] = total
    return out


def verify_against_original(df: pd.DataFrame, n: int = 500, seed: int = 42):
    """Assert the vectorised classifier matches topic_router.classify_article."""
    print(f"\n--- Verifying vectorised classifier on {n} random articles ---")
    sample = df.sample(min(n, len(df)), random_state=seed)
    mismatches = 0
    for _, row in sample.iterrows():
        ref = classify_article(row["headline"], row["body_text"])
        if (ref["category"] != row["topic_category"] or
                ref["total_keyword_hits"] != row["total_keyword_hits"]):
            mismatches += 1
            if mismatches <= 5:
                print(f"  MISMATCH id={row['article_id']}: "
                      f"ref={ref['category']}/{ref['total_keyword_hits']} "
                      f"vec={row['topic_category']}/{row['total_keyword_hits']}")
    if mismatches:
        sys.exit(f"ERROR: {mismatches}/{len(sample)} mismatches vs "
                 "topic_router.classify_article — investigate before using.")
    print("  OK — identical to topic_router.classify_article.")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Keyword relevance filter")
    p.add_argument("--min-hits", type=int, default=MIN_TOTAL_HITS_DEFAULT,
                   help="Minimum weighted keyword hits for relevance "
                        f"(default {MIN_TOTAL_HITS_DEFAULT})")
    args = p.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    df = load_corpus()
    df = classify_vectorised(df)
    verify_against_original(df)

    # ---- Relevance decision ---------------------------------------------
    in_channel = df["topic_category"].isin(
        ["Monetary", "Fiscal", "External", "Energy", "Mixed"])
    df["relevant"] = in_channel & (df["total_keyword_hits"] >= args.min_hits)

    relevant = df[df["relevant"]].drop(columns=["relevant"])
    print(f"\n{'=' * 62}")
    print(f"  RELEVANCE SUMMARY  (min weighted hits = {args.min_hits})")
    print(f"{'-' * 62}")
    print(f"  Corpus                : {len(df):>8,}")
    print(f"  Relevant              : {len(relevant):>8,} "
          f"({len(relevant) / len(df) * 100:.1f}%)")
    print(f"  ...by channel:")
    print(relevant["topic_category"].value_counts().to_string())
    below = int((in_channel & ~df['relevant']).sum())
    print(f"  In-channel but below hit threshold: {below:,}")
    print(f"  Unclassified (no keywords)        : "
          f"{(df['topic_category'] == 'Unclassified').sum():,}")
    print(f"{'=' * 62}")

    relevant.to_csv(RELEVANT_CSV, index=False)
    print(f"\n  Saved -> {RELEVANT_CSV}  ({len(relevant):,} articles)")

    # ---- Audit workbook ---------------------------------------------------
    seed = 42
    summary = (df.assign(bucket=np.where(df["relevant"], "relevant",
                         np.where(df["topic_category"] == "Unclassified",
                                  "unclassified", "below_threshold")))
                 .groupby(["bucket", "topic_category"])
                 .agg(articles=("article_id", "count"),
                      median_hits=("total_keyword_hits", "median"))
                 .reset_index())

    hit_dist = (df[in_channel]
                .groupby("total_keyword_hits")["article_id"].count()
                .rename("articles").reset_index().head(50))

    audit_cols = ["article_id", "date", "headline", "topic_category",
                  "total_keyword_hits"]
    rel_sample = relevant[audit_cols].sample(min(300, len(relevant)),
                                             random_state=seed)
    borderline = df[in_channel & ~df["relevant"]]
    border_sample = borderline[audit_cols].sample(min(300, len(borderline)),
                                                  random_state=seed)
    excluded = df[df["topic_category"] == "Unclassified"]
    excl_sample = excluded[audit_cols].sample(min(300, len(excluded)),
                                              random_state=seed)

    with pd.ExcelWriter(AUDIT_XLSX) as xw:
        summary.to_excel(xw, sheet_name="summary", index=False)
        hit_dist.to_excel(xw, sheet_name="hit_distribution", index=False)
        rel_sample.to_excel(xw, sheet_name="relevant_sample", index=False)
        border_sample.to_excel(xw, sheet_name="borderline_sample", index=False)
        excl_sample.to_excel(xw, sheet_name="excluded_sample", index=False)
    print(f"  Audit workbook -> {AUDIT_XLSX}")

    print("\nNext step:  python run_phase2.py")


if __name__ == "__main__":
    main()
