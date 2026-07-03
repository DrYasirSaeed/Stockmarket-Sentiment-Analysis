"""
Project: Information Transmission and Sectoral Asymmetry in the KSE-100
Module : Phase 2 v2 — Multi-Label Channels, Combined Series, Weekend Shift
Author : Dr. Yasir Saeed (KUST)
Description:
  Rebuilds the daily sentiment series with three methodological upgrades
  (FinBERT article scores are reused from the cached Phase 2 checkpoint —
  no re-scoring):

  1. MULTI-LABEL CHANNEL ASSIGNMENT. An article joins EVERY channel in
     which its weighted keyword evidence (3 x headline + 1 x body hits)
     is >= 3 — the same evidence standard as the relevance filter. A
     genuinely multi-channel article (e.g. an IMF review discussing energy
     tariffs) now informs both channels instead of being parked in a
     'Mixed' bucket. Directional sign rules are applied PER CHANNEL, so
     each channel reads the article through its own event rules.

  2. COMBINED SERIES S_All. A single aggregate sentiment series over ALL
     relevant articles, scored against the union of all sign rules —
     used to measure the impact of overall news sentiment on the combined
     market index (r_Market).

  3. WEEKEND / HOLIDAY SHIFT. News published on non-trading days cannot
     move prices until the next session. Every article is assigned to the
     first trading day >= its publication date (PSX trading calendar taken
     from the price data). Previously such articles were lost in the merge.

  Aggregation (unchanged): word-count-weighted mean within each
  (effective trading day, channel) cell; forward-fill across days with no
  articles in a channel. Raw-FinBERT versions of every series are produced
  in parallel as the robustness benchmark.

OUTPUT (Phase 2 Outputs/)
  article_channel_scores.csv    article x channel level detail
  daily_sentiment_v2.xlsx       date, S_All, S_Monetary..S_Energy (directional)
                                + sheet 'raw_finbert' with the same columns
USAGE
  python rebuild_sentiment.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(r"D:\Stockmarket-Sentiment-Analysis")
sys.path.insert(0, str(PROJECT_DIR))

from keyword_relevance_filter import CATEGORIES, category_hit_counts   # noqa: E402
from psx_sign_rules import score_article_direction                     # noqa: E402

RELEVANT_CSV  = PROJECT_DIR / "Phase 1 Outputs" / "relevant_articles_keyword.csv"
FINBERT_CSV   = PROJECT_DIR / "Phase 2 Outputs" / "finbert_scores.csv"
RETURNS_CSV   = PROJECT_DIR / "Phase 3 Inputs" / "all_sector_returns.csv"
OUT_ARTICLES  = PROJECT_DIR / "Phase 2 Outputs" / "article_channel_scores.csv"
OUT_DAILY     = PROJECT_DIR / "Phase 2 Outputs" / "daily_sentiment_v2.xlsx"

CHANNEL_MIN_HITS = 3        # per-channel evidence threshold (multi-label)
CHUNK = 25_000


def main():
    t0 = time.time()

    # ---- load & merge -------------------------------------------------------
    df = pd.read_csv(RELEVANT_CSV, low_memory=False)
    fb = pd.read_csv(FINBERT_CSV)
    df = df.merge(fb, on="article_id", how="inner")
    df["headline"]  = df["headline"].fillna("").astype(str)
    df["body_text"] = df["body_text"].fillna("").astype(str)
    print(f"Articles with FinBERT scores: {len(df):,}")

    # ---- per-channel weighted hit scores (exact recompute) ------------------
    print("Recomputing per-channel keyword scores ...")
    scores = {c: np.zeros(len(df), dtype=np.int64) for c in CATEGORIES}
    for start in range(0, len(df), CHUNK):
        end = min(start + CHUNK, len(df))
        h = df["headline"].iloc[start:end].str.lower()
        b = df["body_text"].iloc[start:end].str.lower()
        for c, kws in CATEGORIES.items():
            scores[c][start:end] = (category_hit_counts(h, kws) * 3 +
                                    category_hit_counts(b, kws))
    for c in CATEGORIES:
        df[f"hits_{c}"] = scores[c]

    # ---- weekend / holiday shift --------------------------------------------
    cal = pd.read_csv(RETURNS_CSV, usecols=["DATE"])
    trading_days = pd.DatetimeIndex(sorted(pd.to_datetime(cal["DATE"]).unique()))
    df["pub_date"] = pd.to_datetime(df["date"])
    idx = trading_days.searchsorted(df["pub_date"].values, side="left")
    ok = idx < len(trading_days)
    df = df[ok].copy()
    df["eff_date"] = trading_days[idx[ok]]
    shifted = (df["eff_date"] != df["pub_date"]).sum()
    print(f"Weekend/holiday articles shifted to next trading day: "
          f"{shifted:,} ({shifted / len(df) * 100:.1f}%)")

    df["word_count"] = df["body_text"].str.split().str.len().clip(lower=1)

    # ---- per-channel directional scores (multi-label) -----------------------
    print("Applying sign rules per (article, channel) ...")
    long_rows = []
    hl = df["headline"].to_numpy()
    bd = df["body_text"].to_numpy()
    fbs = df["sentiment_score"].to_numpy()
    wc = df["word_count"].to_numpy()
    eff = df["eff_date"].to_numpy()
    aid = df["article_id"].to_numpy()

    for c in CATEGORIES:
        mask = df[f"hits_{c}"].to_numpy() >= CHANNEL_MIN_HITS
        n = int(mask.sum())
        print(f"  {c:<10} articles assigned: {n:,}")
        sel = np.flatnonzero(mask)
        for i in sel:
            r = score_article_direction(hl[i], bd[i], c, fbs[i])
            long_rows.append((aid[i], eff[i], c, wc[i], fbs[i],
                              r["psx_score"], r["psx_source"], r["psx_rules_fired"]))

    long = pd.DataFrame(long_rows, columns=[
        "article_id", "eff_date", "channel", "word_count",
        "finbert_score", "psx_score", "psx_source", "rules_fired"])
    n_multi = long.groupby("article_id")["channel"].size()
    print(f"Articles in >=1 channel: {len(n_multi):,} | "
          f"in >=2 channels: {(n_multi >= 2).sum():,} | "
          f"rule-directed rows: {(long['psx_source'] == 'rule').sum():,}")

    # ---- S_All: all articles, union of all rules ('Mixed' path) -------------
    print("Scoring S_All (all rules) ...")
    all_scores = np.empty(len(df))
    for i in range(len(df)):
        all_scores[i] = score_article_direction(hl[i], bd[i], "All", fbs[i])["psx_score"]
    df["psx_all"] = all_scores

    long.to_csv(OUT_ARTICLES, index=False)
    print(f"Saved -> {OUT_ARTICLES}")

    # ---- daily aggregation ---------------------------------------------------
    def daily_series(frame, date_col, score_col, w_col, name):
        g = frame.groupby(date_col).apply(
            lambda x: np.average(x[score_col], weights=x[w_col]),
            include_groups=False)
        return g.rename(name)

    out = pd.DataFrame(index=trading_days)
    out.index.name = "date"

    for score_col, suffix in [("psx_score", ""), ("finbert_score", "_raw")]:
        for c in CATEGORIES:
            sub = long[long["channel"] == c]
            out[f"S_{c}{suffix}"] = daily_series(sub, "eff_date", score_col,
                                                 "word_count", f"S_{c}{suffix}")
    out["S_All"] = daily_series(df, "eff_date", "psx_all", "word_count", "S_All")
    out["S_All_raw"] = daily_series(df, "eff_date", "sentiment_score",
                                    "word_count", "S_All_raw")

    out = out.ffill().dropna()
    directional = ["S_All"] + [f"S_{c}" for c in CATEGORIES]
    raw = ["S_All_raw"] + [f"S_{c}_raw" for c in CATEGORIES]

    with pd.ExcelWriter(OUT_DAILY) as xw:
        out[directional].reset_index().to_excel(xw, sheet_name="directional", index=False)
        out[raw].reset_index().to_excel(xw, sheet_name="raw_finbert", index=False)

    print(f"\nDaily series: {len(out):,} trading days "
          f"({out.index.min().date()} -> {out.index.max().date()})")
    print(out[directional].describe().round(4).to_string())
    print(f"Saved -> {OUT_DAILY}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
