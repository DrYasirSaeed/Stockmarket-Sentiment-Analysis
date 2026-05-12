"""
Project: Information Transmission and Sectoral Asymmetry in the KSE-100
Module : Phase 1 — Topic Router (Keyword Classifier)
Author : Dr. Yasir Saeed
Affiliation: Kohat University of Science & Technology (KUST)
Description: Classifies each scraped financial news article into one of four
             macroeconomic topic categories — Monetary, Fiscal, External Finance,
             and Energy — using a domain-validated keyword classifier.
             Classification is validated against a manually labelled sample
             of 300 headlines with precision and recall reported per category.
"""

# ===============================================
# Step 0: Import Libraries
# ===============================================
import re
import pandas as pd
from pathlib import Path
from datetime import datetime

print("\n--- Topic Router Initialized ---")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ===============================================
# Step 1: Category Keyword Dictionaries
# ===============================================
# Each dictionary reflects the institutional sources and terminology
# specific to Pakistan's macroeconomic policy environment.
# Validated against domain literature and Business Recorder corpus.

MONETARY_KEYWORDS = [
    # SBP institutional references
    "state bank", "sbp", "mpc", "monetary policy committee",
    "policy rate", "interest rate", "repo rate", "discount rate",
    # Rate actions
    "rate hike", "rate cut", "rate unchanged", "basis points", "bps",
    # Monetary aggregates
    "reserve money", "money supply", "m2", "m1", "broad money",
    # Credit and liquidity
    "open market operation", "omo", "treasury bills", "t-bills",
    "foreign exchange reserve", "forex reserve", "exchange rate",
    "rupee depreciation", "rupee appreciation", "pkr",
    # Inflation channel
    "inflation target", "cpi", "core inflation", "headline inflation",
]

FISCAL_KEYWORDS = [
    # Tax authority
    "fbr", "federal board of revenue", "tax revenue", "tax collection",
    "income tax", "sales tax", "gst", "customs duty", "withholding tax",
    # Budget process
    "federal budget", "budget deficit", "fiscal deficit", "primary surplus",
    "psdp", "public sector development", "annual development plan", "adp",
    # Expenditure
    "government spending", "current expenditure", "defence budget",
    "subsidy", "pension", "public debt", "domestic debt", "external debt",
    # Ratings and fiscal risk
    "sovereign rating", "credit rating", "s&p", "moody", "fitch",
    "debt to gdp", "fiscal consolidation", "austerity",
]

EXTERNAL_FINANCE_KEYWORDS = [
    # IMF
    "imf", "international monetary fund", "programme review",
    "extended fund facility", "eff", "stand-by arrangement", "sba",
    "imf tranche", "imf disbursement", "conditionality",
    # World Bank and multilaterals
    "world bank", "asian development bank", "adb", "islamic development bank",
    "isdb", "bilateral loan", "paris club",
    # Balance of payments
    "current account", "current account deficit", "trade deficit",
    "remittances", "workers remittances", "foreign direct investment", "fdi",
    "portfolio investment", "capital account", "financial account",
    # External debt
    "external debt", "debt repayment", "debt rollover", "eurobond",
    "sukuk", "global bond", "foreign currency",
]

ENERGY_KEYWORDS = [
    # Regulators
    "nepra", "ogra", "national electric power regulatory authority",
    "oil and gas regulatory authority",
    # Utilities and fuel
    "electricity tariff", "power tariff", "gas tariff", "gas price",
    "petroleum price", "petrol price", "diesel price", "furnace oil",
    # Circular debt
    "circular debt", "power sector debt", "inter-corporate debt",
    "capacity payment", "cppa", "ntdc",
    # Generation and distribution
    "load shedding", "loadshedding", "power outage", "power cut",
    "electricity shortage", "power generation", "power plant",
    # Oil and gas producers (major KSE-100 E&P firms)
    "ogdcl", "ppl", "mari petroleum", "pso", "hascol",
    "sui northern", "sngpl", "sui southern", "ssgcl",
    # Global benchmarks
    "crude oil", "brent crude", "opec", "oil price", "lng", "rlng",
]

print("\n--- Keyword Dictionaries Loaded ---")
print(f"Monetary   : {len(MONETARY_KEYWORDS)} keywords")
print(f"Fiscal     : {len(FISCAL_KEYWORDS)} keywords")
print(f"External   : {len(EXTERNAL_FINANCE_KEYWORDS)} keywords")
print(f"Energy     : {len(ENERGY_KEYWORDS)} keywords")


# ===============================================
# Step 2: Classification Functions
# ===============================================
def count_keyword_hits(text: str, keywords: list) -> int:
    """
    Count total keyword occurrences in text (case-insensitive).
    Uses str.count() so that multiple occurrences of the same keyword
    accumulate — this is required for the 3x headline weighting in
    classify_article() to have any effect.
    Partial matches included (e.g. 'inflation' matches 'hyperinflation').
    """
    text_lower = text.lower()
    return sum(text_lower.count(kw) for kw in keywords)


def classify_article(headline: str, body_text: str = "") -> dict:
    """
    Classify a news article into one of four macroeconomic topic categories.
    Headline hits are weighted 3x body hits (editorial emphasis signal).
    Returns category label and normalised confidence scores.
    """
    headline_lower = headline.lower()
    body_lower     = body_text.lower()

    category_keywords = {
        "Monetary": MONETARY_KEYWORDS,
        "Fiscal":   FISCAL_KEYWORDS,
        "External": EXTERNAL_FINANCE_KEYWORDS,
        "Energy":   ENERGY_KEYWORDS,
    }

    scores = {
        cat: count_keyword_hits(headline_lower, kws) * 3 +
             count_keyword_hits(body_lower, kws)
        for cat, kws in category_keywords.items()
    }

    total = sum(scores.values())

    if total > 0:
        confidence = {k: round(v / total, 3) for k, v in scores.items()}
    else:
        confidence = {k: 0.0 for k in scores}

    top_category = max(scores, key=scores.get)
    top_score    = scores[top_category]

    if top_score == 0:
        category = "Unclassified"
    elif confidence[top_category] < 0.35:
        category = "Mixed"   # No dominant category — flag for manual review
    else:
        category = top_category

    return {
        "category":           category,
        "confidence_scores":  confidence,
        "total_keyword_hits": total,
    }


# ===============================================
# Step 3: Batch Classification Pipeline
# ===============================================
def classify_corpus(df: pd.DataFrame,
                    headline_col: str = "headline",
                    body_col: str = "body_text") -> pd.DataFrame:
    """
    Classify an entire article corpus.
    Expects a DataFrame with headline and body_text columns.
    Returns DataFrame with five new columns:
      - topic_category   : primary classification label
      - conf_monetary    : confidence score for Monetary category
      - conf_fiscal      : confidence score for Fiscal category
      - conf_external    : confidence score for External Finance category
      - conf_energy      : confidence score for Energy category
    """
    print(f"\n--- Classifying {len(df)} articles ---")
    results = []

    for i, (_, row) in enumerate(df.iterrows()):
        headline  = str(row.get(headline_col, ""))
        body_text = str(row.get(body_col, ""))
        result    = classify_article(headline, body_text)

        results.append({
            "topic_category": result["category"],
            "conf_monetary":  result["confidence_scores"].get("Monetary", 0),
            "conf_fiscal":    result["confidence_scores"].get("Fiscal", 0),
            "conf_external":  result["confidence_scores"].get("External", 0),
            "conf_energy":    result["confidence_scores"].get("Energy", 0),
        })

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1} / {len(df)} articles ...")

    result_df = pd.DataFrame(results, index=df.index)
    df_out    = pd.concat([df, result_df], axis=1)

    print("\n--- Classification Complete ---")
    print(df_out["topic_category"].value_counts().to_string())
    return df_out


# ===============================================
# Step 4: Validation Against Labelled Sample
# ===============================================
def validate_classifier(labelled_df: pd.DataFrame,
                         true_label_col: str = "true_category") -> dict:
    """
    Validate classifier against manually labelled headlines.
    Computes precision, recall, and F1 for each category.
    Target: minimum 300 labelled headlines per proposal specification.

    Decision rule for academic reporting:
      Precision >= 0.75 and Recall >= 0.70 -> Category accepted for analysis
      Below threshold -> Category keywords require refinement
    """
    from sklearn.metrics import classification_report

    print("\n--- Classifier Validation ---")
    print(f"Labelled sample size: {len(labelled_df)} articles")

    classified = classify_corpus(labelled_df)
    y_true = labelled_df[true_label_col].values
    y_pred = classified["topic_category"].values

    report = classification_report(y_true, y_pred, output_dict=True)

    print("\n--- Precision / Recall / F1 by Category ---")
    for category in ["Monetary", "Fiscal", "External", "Energy"]:
        if category in report:
            p = report[category]["precision"]
            r = report[category]["recall"]
            f = report[category]["f1-score"]
            decision = "Accepted" if p >= 0.75 and r >= 0.70 else "Refine keywords"
            print(f"  {category:<16} P={p:.3f}  R={r:.3f}  F1={f:.3f}  -> {decision}")

    print(f"\n  Overall Accuracy: {report['accuracy']:.3f}")
    return report


# ===============================================
# Step 5: Demo
# ===============================================
if __name__ == "__main__":
    print("\n--- Demo: Single Article Classification ---")

    test_articles = [
        {
            "headline":  "SBP cuts policy rate by 100bps to 15 percent amid falling inflation",
            "body_text": "The State Bank of Pakistan's Monetary Policy Committee reduced the "
                         "benchmark interest rate by 100 basis points to 15 percent on Friday, "
                         "citing a significant deceleration in headline CPI inflation."
        },
        {
            "headline":  "FBR misses revenue target by Rs180 billion in first quarter",
            "body_text": "The Federal Board of Revenue collected Rs1.7 trillion in the first "
                         "quarter against a target of Rs1.88 trillion, widening the fiscal "
                         "deficit ahead of the IMF programme review."
        },
        {
            "headline":  "IMF approves $1.1 billion tranche for Pakistan under EFF",
            "body_text": "The International Monetary Fund's executive board completed the second "
                         "review under the Extended Fund Facility, approving disbursement of "
                         "approximately $1.1 billion to Pakistan."
        },
        {
            "headline":  "NEPRA approves 18 percent electricity tariff increase",
            "body_text": "The National Electric Power Regulatory Authority approved an 18 percent "
                         "increase in base electricity tariff effective next month to address "
                         "mounting circular debt in the power sector."
        },
    ]

    for article in test_articles:
        result = classify_article(article["headline"], article["body_text"])
        print(f"\n  Headline  : {article['headline'][:70]}")
        print(f"  Category  : {result['category']}")
        print(f"  Confidence: {result['confidence_scores']}")

    print("\n--- Analysis Complete ---")
