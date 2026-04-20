"""
Project: Information Transmission and Sectoral Asymmetry in the KSE-100
Module : Phase 2 — FinBERT Sentiment Scoring Pipeline
Author : Dr. Yasir Saeed
Affiliation: Kohat University of Science & Technology (KUST)
Description: Scores each classified news article using FinBERT
             (Araci, 2019) — a BERT-variant fine-tuned on financial corpora.
             Produces signed sentiment scores in [−1, +1] per article,
             then aggregates to four daily topic-specific sentiment series:
             S_Monetary,t | S_Fiscal,t | S_External,t | S_Energy,t
             These four series constitute the primary explanatory variables
             in the DCC-GARCH and Panel VAR stages.

Reference: Araci, D. (2019). FinBERT: Financial sentiment analysis with
           pre-trained language models. arXiv:1908.10063.
"""

# ===============================================
# 📌 Step 0: Import Libraries
# ===============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Professional theme for academic publication
sns.set_theme(style="whitegrid", palette="magma")

print("\n--- FinBERT Sentiment Pipeline Initialized ---")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Model     : ProsusAI/finbert (Araci, 2019)")


# ===============================================
# 📌 Step 1: Load FinBERT Model
# ===============================================
def load_finbert():
    """
    Load FinBERT from HuggingFace Hub.
    Model: ProsusAI/finbert — fine-tuned on financial news and analyst reports.
    Outputs: positive / negative / neutral with probability scores.

    Installation: pip install transformers torch
    """
    try:
        from transformers import pipeline
        print("\n--- Loading FinBERT from HuggingFace ---")
        finbert = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            truncation=True,
            max_length=512,      # BERT max token limit
        )
        print("Decision: FinBERT loaded successfully")
        return finbert
    except ImportError:
        print("transformers not installed. Run: pip install transformers torch")
        raise


# ===============================================
# 📌 Step 2: Score Individual Articles
# ===============================================
def score_article(text: str, finbert_pipeline) -> dict:
    """
    Score a single article text using FinBERT.
    Converts categorical output to signed numeric score:
      positive → +confidence score
      negative → −confidence score
      neutral  →  0

    Truncates to 512 tokens — standard BERT constraint.
    For long articles, scoring is applied to the lead paragraph
    (first 512 tokens), which contains the primary news signal.
    """
    if not text or len(text.strip()) < 20:
        return {"label": "neutral", "score": 0.0, "signed_score": 0.0}

    result = finbert_pipeline(text[:2000])[0]  # Truncate input for safety
    label  = result["label"].lower()
    conf   = result["score"]

    # Convert to signed score for time-series analysis
    if label == "positive":
        signed = +conf
    elif label == "negative":
        signed = -conf
    else:
        signed = 0.0

    return {
        "label":        label,
        "score":        conf,
        "signed_score": round(signed, 4),
    }


# ===============================================
# 📌 Step 3: Batch Scoring Pipeline
# ===============================================
def score_corpus(df: pd.DataFrame,
                 text_col: str = "body_text",
                 finbert_pipeline=None) -> pd.DataFrame:
    """
    Score all articles in the classified corpus.
    Adds three columns: sentiment_label, sentiment_conf, sentiment_score.
    Saves progress every 100 rows — protects against Colab session timeouts.
    """
    if finbert_pipeline is None:
        finbert_pipeline = load_finbert()

    print(f"\n--- Scoring {len(df)} articles with FinBERT ---")
    labels, confs, scores = [], [], []

    for i, row in df.iterrows():
        text   = str(row.get(text_col, ""))
        result = score_article(text, finbert_pipeline)
        labels.append(result["label"])
        confs.append(result["score"])
        scores.append(result["signed_score"])

        if (i + 1) % 100 == 0:
            print(f"  Scored {i + 1} / {len(df)} articles ...")

    df["sentiment_label"] = labels
    df["sentiment_conf"]  = confs
    df["sentiment_score"] = scores  # Primary variable: signed ∈ [−1, +1]

    print("\n--- Sentiment Distribution ---")
    print(df["sentiment_label"].value_counts().to_string())
    print(f"\nMean signed score : {df['sentiment_score'].mean():.4f}")
    print(f"Std  signed score : {df['sentiment_score'].std():.4f}")

    return df


# ===============================================
# 📌 Step 4: Daily Aggregation to Four Sentiment Series
# ===============================================
def aggregate_daily_sentiment(df: pd.DataFrame,
                               date_col: str = "date",
                               category_col: str = "topic_category",
                               score_col: str = "sentiment_score") -> pd.DataFrame:
    """
    Aggregate article-level scores to daily topic-specific sentiment series.
    Method: value-weighted average within each date-category cell.
    Weight = word count (longer, more detailed articles receive higher weight).

    Output columns:
      S_Monetary | S_Fiscal | S_External | S_Energy
    One row per trading day — aligned to PSX trading calendar.
    Missing days (weekends, holidays) carry forward the last observed value.
    """
    print("\n--- Aggregating to Daily Sentiment Series ---")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df["date_only"] = df[date_col].dt.date

    # Value-weighted average: weight by word count
    if "word_count" not in df.columns:
        df["word_count"] = df.get("body_text", pd.Series([""] * len(df))).apply(
            lambda t: len(str(t).split())
        )
    df["word_count"] = df["word_count"].clip(lower=1)

    daily_scores = {}
    for category in ["Monetary", "Fiscal", "External", "Energy"]:
        cat_df = df[df[category_col] == category].copy()
        if cat_df.empty:
            print(f"  Warning: No {category} articles found")
            continue

        # Value-weighted mean per day
        cat_df["weighted_score"] = cat_df[score_col] * cat_df["word_count"]
        grouped = cat_df.groupby("date_only").apply(
            lambda g: g["weighted_score"].sum() / g["word_count"].sum()
        ).reset_index()
        grouped.columns = ["date_only", f"S_{category}"]
        daily_scores[category] = grouped

    # Merge all four series on date
    from functools import reduce
    if not daily_scores:
        print("Decision: No classified articles available — corpus required")
        return pd.DataFrame()

    merged = reduce(
        lambda a, b: pd.merge(a, b, on="date_only", how="outer"),
        daily_scores.values()
    )
    merged = merged.sort_values("date_only").reset_index(drop=True)

    # Forward-fill missing trading days (carry last observation)
    score_cols = [c for c in merged.columns if c.startswith("S_")]
    merged[score_cols] = merged[score_cols].fillna(method="ffill")

    print(f"\n--- Daily Sentiment Series: {len(merged)} trading days ---")
    print(merged[score_cols].describe().round(4).to_string())

    return merged


# ===============================================
# 📌 Step 5: Visualization — Sentiment Time Series
# ===============================================
def plot_sentiment_series(daily_df: pd.DataFrame,
                           output_path: str = "sentiment_series.png"):
    """
    Plot the four daily sentiment series for visual inspection.
    Key diagnostic: series should show no obvious structural breaks
    prior to known policy shock dates (used for eyeball validation).
    """
    score_cols  = [c for c in daily_df.columns if c.startswith("S_")]
    colors      = ["#1F3864", "#C00000", "#375623", "#7F3F98"]
    labels      = ["Monetary", "Fiscal", "External Finance", "Energy"]

    fig, axes = plt.subplots(len(score_cols), 1,
                              figsize=(14, 3 * len(score_cols)),
                              sharex=True)

    for ax, col, color, label in zip(axes, score_cols, colors, labels):
        ax.plot(daily_df["date_only"], daily_df[col],
                color=color, linewidth=0.8, alpha=0.85)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_ylabel(f"S_{label}", fontsize=10)
        ax.set_title(f"Daily {label} Sentiment — KSE-100 Study (2014–2025)",
                     fontsize=11)

    axes[-1].set_xlabel("Date", fontsize=10)
    plt.suptitle("Topic-Routed Daily Sentiment Series (FinBERT)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n  Sentiment plot saved → {output_path}")
    plt.show()


# ===============================================
# 📌 Step 6: Save Processed Sentiment Series
# ===============================================
def save_sentiment_series(daily_df: pd.DataFrame,
                           output_path: str = "../data/processed/daily_sentiment.xlsx"):
    """
    Save the four daily sentiment series to Excel.
    This file is the primary input to Phase 3 (DCC-GARCH) and
    Phase 4 (Panel VAR) estimation stages.
    """
    daily_df.to_excel(output_path, index=False)
    print(f"\n--- Sentiment Series Saved ---")
    print(f"Output    : {Path(output_path).resolve()}")
    print(f"Shape     : {daily_df.shape}")
    print(f"Date range: {daily_df['date_only'].min()} → {daily_df['date_only'].max()}")


# ===============================================
# 📌 Step 7: Entry Point
# ===============================================
if __name__ == "__main__":
    print("\n--- FinBERT Pipeline: Awaiting Corpus Input ---")
    print("Expected input : ../data/interim/classified_articles.xlsx")
    print("                 (output of 01_data_acquisition/topic_router.py)")

    input_path = Path("../data/interim/classified_articles.xlsx")

    if input_path.exists():
        df = pd.read_excel(input_path)
        print(f"\n--- Corpus Loaded: {len(df)} articles ---")
        df = score_corpus(df)
        daily_df = aggregate_daily_sentiment(df)
        plot_sentiment_series(daily_df)
        save_sentiment_series(daily_df)
    else:
        print("\nDecision: Corpus file not found — Phase 1 (data acquisition) must complete first.")
        print("See 01_data_acquisition/README.md for data acquisition status.")

    print("\n--- Analysis Complete ---")
