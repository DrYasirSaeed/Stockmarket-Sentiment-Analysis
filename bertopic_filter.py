"""
bertopic_filter.py — Bridge between BERTopic Phase 1a and topic_router.py Phase 1b.

Reads your manual keep/discard decisions from bertopic_topics.xlsx, classifies
every article into one of four groups, and writes:

  filtered_corpus.csv   — ALL articles with a 'group' label (nothing discarded)
  relevant_articles.csv — only the two relevant groups, ready for second BERTopic run

GROUP LABELS
  relevant_topic     : assigned to a keep=1 topic by BERTopic
  relevant_rescued   : topic -1 outlier that contains keywords from keep=1 topics
  irrelevant_topic   : assigned to a keep=0 topic by BERTopic
  irrelevant_outlier : topic -1 outlier with no relevant keywords found

USAGE
  python bertopic_filter.py            # run filter
  python bertopic_filter.py --status   # check input/output file status
"""

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
import os

CONFIG = {
    # ── Inputs ───────────────────────────────────────────────────────────────
    # BERTopic outputs folder
    "bertopic_out_dir":  r"D:\Stockmarket-Sentiment-Analysis\BERTopic Outputs",

    # Your annotated topics file — must have a 'keep' column (1 = relevant, 0 = irrelevant)
    "topics_xlsx":       "bertopic_topics.xlsx",

    # Assignment file produced by bertopic_explore.py
    "assignments_csv":   "bertopic_assignments.csv",

    # Original scraped CSVs (needed for body_text)
    "data_dir":          r"D:\Stockmarket-Sentiment-Analysis\Extracted Data",

    # ── Outputs (written into bertopic_out_dir) ───────────────────────────
    "filtered_corpus":   "filtered_corpus.csv",       # all articles + group label
    "relevant_articles": "relevant_articles.csv",     # relevant_topic + relevant_rescued only

    # ── Keyword rescue settings (for topic -1 articles) ───────────────────
    # Minimum number of relevant keywords that must appear in headline+body
    # to rescue an outlier article into the relevant group.
    "rescue_min_keywords": 1,

    # Whether to match whole words only (True) or allow substring matches (False)
    # True is safer — prevents 'rate' matching 'enumerate', etc.
    "rescue_whole_word": True,
}

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
import argparse
import sys
import re
import glob

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="BERTopic corpus filter")
    p.add_argument("--status", action="store_true",
                   help="Show input/output file status and exit")
    return p.parse_args()


def b_out(fname):
    """Resolve a filename inside the BERTopic output directory."""
    return os.path.join(CONFIG["bertopic_out_dir"], fname)


def status_check():
    print("\n=== File status ===")
    inputs = [
        ("topics_xlsx (annotated)",  b_out(CONFIG["topics_xlsx"])),
        ("assignments_csv",          b_out(CONFIG["assignments_csv"])),
        ("data_dir (Extracted Data)",CONFIG["data_dir"]),
    ]
    outputs = [
        ("filtered_corpus.csv",   b_out(CONFIG["filtered_corpus"])),
        ("relevant_articles.csv", b_out(CONFIG["relevant_articles"])),
    ]
    print("\n  INPUTS")
    for label, path in inputs:
        exists = os.path.isfile(path) or os.path.isdir(path)
        print(f"  {'[OK]    ' if exists else '[MISSING]'} {label}")
        print(f"            {path}")
    print("\n  OUTPUTS")
    for label, path in outputs:
        exists = os.path.isfile(path)
        print(f"  {'[OK]    ' if exists else '[MISSING]'} {label}")
    print()


# ---------------------------------------------------------------------------
# KEYWORD HELPERS
# ---------------------------------------------------------------------------
def parse_keywords(topics_df):
    """
    Extract all unique keywords from keep=1 topics.
    Returns a list of keyword strings (lowercase, stripped).
    """
    keep_rows = topics_df[topics_df["keep"] == 1]
    keywords = set()
    for kw_string in keep_rows["top_10_keywords"].dropna():
        for kw in kw_string.split(","):
            kw = kw.strip().lower()
            if kw:
                keywords.add(kw)
    return sorted(keywords)


def build_pattern(keywords, whole_word):
    """Compile a single regex that matches any keyword in the list."""
    escaped = [re.escape(kw) for kw in keywords]
    joined  = "|".join(escaped)
    if whole_word:
        return re.compile(r"\b(?:" + joined + r")\b", re.IGNORECASE)
    else:
        return re.compile(joined, re.IGNORECASE)


def count_keyword_hits(text, pattern):
    """Return number of distinct keyword matches found in text."""
    return len(pattern.findall(str(text)))


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    if args.status:
        status_check()
        sys.exit(0)

    # ── 1. Load annotated topics file ────────────────────────────────────────
    topics_path = b_out(CONFIG["topics_xlsx"])
    if not os.path.isfile(topics_path):
        sys.exit(f"ERROR: topics file not found:\n  {topics_path}\n"
                 "Run bertopic_explore.py first, then add a 'keep' column (1/0).")

    topics_df = pd.read_excel(topics_path)

    if "keep" not in topics_df.columns:
        sys.exit("ERROR: 'keep' column not found in bertopic_topics.xlsx.\n"
                 "Add a column named 'keep' with 1 (relevant) or 0 (irrelevant) "
                 "for each topic, then re-run.")

    # Validate keep values
    valid_keep = {0, 1}
    bad = topics_df[~topics_df["keep"].isin(valid_keep) & topics_df["keep"].notna()]
    if len(bad):
        print(f"  WARNING: {len(bad)} rows have unexpected 'keep' values "
              f"(expected 0 or 1). They will be treated as irrelevant.")
        topics_df["keep"] = pd.to_numeric(topics_df["keep"], errors="coerce").fillna(0)

    keep_ids    = set(topics_df[topics_df["keep"] == 1]["topic_id"].tolist())
    discard_ids = set(topics_df[topics_df["keep"] == 0]["topic_id"].tolist())

    print(f"\n  Relevant topics  (keep=1): {sorted(keep_ids)}")
    print(f"  Irrelevant topics (keep=0): {sorted(discard_ids)}")

    # ── 2. Extract rescue keywords from keep=1 topics ────────────────────────
    keywords = parse_keywords(topics_df)
    pattern  = build_pattern(keywords, CONFIG["rescue_whole_word"])
    print(f"\n  Rescue keywords ({len(keywords)} total from relevant topics):")
    print(f"  {', '.join(keywords[:30])}{'…' if len(keywords) > 30 else ''}")

    # ── 3. Load assignments ──────────────────────────────────────────────────
    assign_path = b_out(CONFIG["assignments_csv"])
    if not os.path.isfile(assign_path):
        sys.exit(f"ERROR: assignments file not found:\n  {assign_path}\n"
                 "Run bertopic_explore.py first.")

    assignments = pd.read_csv(assign_path)
    print(f"\n  Assignments loaded: {len(assignments):,} articles")

    # ── 4. Load original CSVs to get body_text ──────────────────────────────
    print(f"\n  Loading original CSVs from {CONFIG['data_dir']} …")
    csv_files = glob.glob(os.path.join(CONFIG["data_dir"], "*.csv"))
    if not csv_files:
        sys.exit(f"ERROR: No CSV files found in {CONFIG['data_dir']}")

    frames = []
    for f in csv_files:
        try:
            frames.append(pd.read_csv(f, low_memory=False, on_bad_lines="skip"))
        except Exception as e:
            print(f"  WARNING: could not read {f}: {e}")

    raw_df = pd.concat(frames, ignore_index=True)

    if "http_status" in raw_df.columns:
        raw_df = raw_df[raw_df["http_status"] == 200]

    raw_df["headline"]  = raw_df["headline"].fillna("").astype(str)
    raw_df["body_text"] = raw_df["body_text"].fillna("").astype(str)

    # Keep only the columns we need for the merge
    body_cols = ["article_id", "body_text"]
    raw_df = raw_df[body_cols].drop_duplicates(subset="article_id")

    # ── 5. Merge assignments with body_text ──────────────────────────────────
    df = assignments.merge(raw_df, on="article_id", how="left")
    df["body_text"] = df["body_text"].fillna("")
    print(f"  Merged: {len(df):,} articles with body_text")

    # ── 6. Classify each article ─────────────────────────────────────────────
    print("\n  Classifying articles …")

    def classify(row):
        tid = row["topic_id"]
        if tid in keep_ids:
            return "relevant_topic"
        elif tid in discard_ids:
            return "irrelevant_topic"
        elif tid == -1:
            # Keyword rescue: search headline + body_text
            combined = row["headline"] + " " + row["body_text"]
            hits = count_keyword_hits(combined, pattern)
            if hits >= CONFIG["rescue_min_keywords"]:
                return "relevant_rescued"
            else:
                return "irrelevant_outlier"
        else:
            # topic_id not found in topics file (shouldn't happen, but safe fallback)
            return "irrelevant_outlier"

    df["group"] = df.apply(classify, axis=1)

    # ── 7. Summary ───────────────────────────────────────────────────────────
    counts = df["group"].value_counts()
    total  = len(df)

    print(f"\n{'=' * 60}")
    print(f"  CLASSIFICATION SUMMARY  ({total:,} total articles)")
    print(f"{'─' * 60}")
    for group in ["relevant_topic", "relevant_rescued",
                  "irrelevant_topic", "irrelevant_outlier"]:
        n   = counts.get(group, 0)
        pct = n / total * 100
        print(f"  {group:<22}  {n:>6,}  ({pct:.1f}%)")
    print(f"{'─' * 60}")
    relevant_n = counts.get("relevant_topic", 0) + counts.get("relevant_rescued", 0)
    print(f"  {'TOTAL RELEVANT':<22}  {relevant_n:>6,}  "
          f"({relevant_n / total * 100:.1f}%)")
    print(f"{'=' * 60}\n")

    # ── 8. Save outputs ──────────────────────────────────────────────────────
    # Column order for outputs
    out_cols = [
        "article_id", "date", "headline", "body_text",
        "topic_id", "topic_probability", "group",
    ]
    out_cols = [c for c in out_cols if c in df.columns]

    # Full corpus with group labels
    corpus_path = b_out(CONFIG["filtered_corpus"])
    df[out_cols].to_csv(corpus_path, index=False)
    print(f"  Saved → {corpus_path}")
    print(f"          ({total:,} articles, all groups)")

    # Relevant only — for second BERTopic run
    relevant_df   = df[df["group"].isin(["relevant_topic", "relevant_rescued"])]
    relevant_path = b_out(CONFIG["relevant_articles"])
    relevant_df[out_cols].to_csv(relevant_path, index=False)
    print(f"  Saved → {relevant_path}")
    print(f"          ({len(relevant_df):,} articles, relevant groups only)\n")

    print("Next step:")
    print("  python bertopic_explore.py --refit")
    print(f"  (update CONFIG['data_dir'] to point at {relevant_path})\n")


if __name__ == "__main__":
    main()
