"""
Project: Information Transmission and Sectoral Asymmetry in the KSE-100
Module : Phase 2b — PSX Directional Sign Rules
Author : Dr. Yasir Saeed
Affiliation: Kohat University of Science & Technology (KUST)
Description: FinBERT (Araci, 2019) measures the *tone* of financial text,
             not its direction for equity investors. "SBP hikes policy rate
             to fight inflation" reads as decisive/neutral tone but is
             unambiguously negative for KSE-100 equities.

             This module overlays channel-specific directional event rules
             on top of FinBERT scores:

               psx_score = rule_sign x |finbert_score|   if a rule fires
               psx_score = finbert_score                 otherwise

             The rule supplies the DIRECTION (sign for PSX equities);
             FinBERT's confidence supplies the MAGNITUDE. When FinBERT is
             neutral (score = 0) but a directional event fired, a default
             magnitude NEUTRAL_MAGNITUDE is used so the event is not lost.

             Rule matches in the headline are weighted 3x matches in the
             body lead (first 2,000 characters), consistent with the
             editorial-emphasis convention in topic_router.py.

DELIBERATE OMISSIONS (fall back to FinBERT):
  - Global crude oil price moves: ambiguous for the aggregate market
    (positive for E&P earnings, negative for the import bill and
    downstream sectors). Left to the Panel VAR to disentangle.
  - "Rate unchanged" decisions: direction depends on expectations.

USAGE
  from psx_sign_rules import apply_sign_rules, export_audit
  df = apply_sign_rules(df)           # needs headline, body_text,
                                      # topic_category, sentiment_score
  export_audit(df, "sign_rules_audit.xlsx")
"""

import re
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
HEADLINE_WEIGHT   = 3      # headline hits count 3x body-lead hits
BODY_LEAD_CHARS   = 2000   # same scoring window as finbert_scorer.py
NEUTRAL_MAGNITUDE = 0.5    # |psx_score| when a rule fires but FinBERT is neutral

# ---------------------------------------------------------------------------
# DIRECTIONAL EVENT RULES
# ---------------------------------------------------------------------------
# sign = +1 : positive for KSE-100 equities   |   sign = -1 : negative
# Patterns are case-insensitive regular expressions.
# Rules are grouped by the topic_router.py channel they belong to.
# Articles labelled "Mixed" or "Unclassified" are checked against ALL rules.

RULES = {
    "Monetary": [
        {"name": "rate_cut", "sign": +1, "patterns": [
            r"rate cut",
            r"cuts? (?:the )?(?:policy|interest|key|benchmark|discount) rate",
            r"(?:reduc|lower|slash|eas)\w* (?:the )?(?:policy|interest|key|benchmark|discount) rate",
            r"monetary easing",
        ]},
        {"name": "rate_hike", "sign": -1, "patterns": [
            r"rate hike",
            r"(?:hik|rais|increas)\w* (?:the )?(?:policy|interest|key|benchmark|discount) rate",
            r"monetary tightening",
            r"tightens? monetary policy",
        ]},
        {"name": "inflation_up", "sign": -1, "patterns": [
            r"inflation (?:ris|surg|accelerat|jump|climb|spik|soar)\w*",
            r"cpi (?:ris|surg|jump|climb|spik)\w*",
            r"(?:highest|record|multi-year high) inflation",
        ]},
        {"name": "inflation_down", "sign": +1, "patterns": [
            r"inflation (?:eas|fall|declin|decelerat|slow|drop|cool)\w*",
            r"cpi (?:eas|fall|declin|slow|drop)\w*",
            r"disinflation",
        ]},
        {"name": "rupee_down", "sign": -1, "patterns": [
            r"rupee (?:depreciat|fall|weaken|slid|slide|plung|declin|drop|tumbl)\w*",
            r"rupee (?:hits|at|touches) (?:a )?(?:record|all-time|historic) low",
            r"depreciation of (?:the )?rupee",
        ]},
        {"name": "rupee_up", "sign": +1, "patterns": [
            r"rupee (?:appreciat|gain|strengthen|recover|rebound|ris)\w*",
        ]},
        {"name": "reserves_up", "sign": +1, "patterns": [
            r"(?:forex|foreign exchange|fx) reserves? (?:ris|increas|improv|climb|jump|swell|cross)\w*",
        ]},
        {"name": "reserves_down", "sign": -1, "patterns": [
            r"(?:forex|foreign exchange|fx) reserves? (?:fall|declin|drop|deplet|dwindl|slip)\w*",
        ]},
    ],

    "Fiscal": [
        {"name": "revenue_beat", "sign": +1, "patterns": [
            r"(?:fbr|revenue|tax collection).{0,30}(?:exceed|surpass|beat)\w*",
            r"exceeds? (?:its )?(?:revenue|tax|collection) target",
        ]},
        {"name": "revenue_miss", "sign": -1, "patterns": [
            r"(?:fbr|revenue|tax collection).{0,30}(?:miss|shortfall|falls? short)",
            r"misses (?:its )?(?:revenue|tax|collection) target",
        ]},
        {"name": "deficit_widens", "sign": -1, "patterns": [
            r"(?:fiscal|budget) deficit (?:widen|ris|swell|balloon|grow)\w*",
        ]},
        {"name": "deficit_narrows", "sign": +1, "patterns": [
            r"(?:fiscal|budget) deficit (?:narrow|shrink|fall|declin)\w*",
            r"primary surplus",
        ]},
        {"name": "new_taxes", "sign": -1, "patterns": [
            r"mini-?budget",
            r"impos\w+ (?:new |additional )?tax",
            r"(?:hik|rais|increas)\w+.{0,15}(?:sales tax|gst|customs duty|withholding tax|excise)",
        ]},
        {"name": "rating_downgrade", "sign": -1, "patterns": [
            r"downgrad\w+",
            r"outlook.{0,20}negative",
        ]},
        {"name": "rating_upgrade", "sign": +1, "patterns": [
            r"rating upgrad\w*",
            r"upgrad\w+ (?:pakistan|(?:the )?(?:sovereign |credit )?rating|(?:the )?outlook)",
            r"outlook.{0,20}(?:revised (?:up|to positive)|positive|to stable)",
        ]},
    ],

    "External": [
        {"name": "imf_positive", "sign": +1, "patterns": [
            r"imf (?:approv|clear|complet|reach|disburs|releas)\w*",
            r"staff.level agreement",
            r"tranche (?:approv|disburs|releas|receiv)\w*",
            r"(?:secur|reach|clinch)\w+.{0,30}imf (?:deal|agreement|programme|program|bailout)",
        ]},
        {"name": "imf_negative", "sign": -1, "patterns": [
            r"imf (?:talks?|review|programme|program|deal).{0,40}(?:stall|delay|fail|suspend|derail|deadlock|collaps)",
            r"(?:stall|delay|fail|suspend|derail)\w*.{0,40}imf",
        ]},
        {"name": "cad_widens", "sign": -1, "patterns": [
            r"(?:current account|trade) deficit (?:widen|ris|swell|grow|balloon)\w*",
        ]},
        {"name": "cad_narrows", "sign": +1, "patterns": [
            r"(?:current account|trade) deficit (?:narrow|shrink|fall|declin|contract)\w*",
            r"current account (?:posts? a )?surplus",
        ]},
        {"name": "remittances_up", "sign": +1, "patterns": [
            r"remittances?.{0,30}(?:ris|increas|grow|surg|jump|climb|hit record|record high)\w*",
        ]},
        {"name": "remittances_down", "sign": -1, "patterns": [
            r"remittances?.{0,30}(?:fall|declin|drop|shrink|slip)\w*",
        ]},
        {"name": "fdi_up", "sign": +1, "patterns": [
            r"(?:fdi|foreign direct investment|foreign investment).{0,30}(?:ris|increas|grow|surg|jump|inflow)",
        ]},
        {"name": "fdi_down", "sign": -1, "patterns": [
            r"(?:fdi|foreign direct investment|foreign investment).{0,30}(?:fall|declin|drop|outflow)",
        ]},
        {"name": "default_risk", "sign": -1, "patterns": [
            r"default (?:risk|fears?|looms?)",
            r"cds spreads? (?:widen|spik|surg)\w*",
            r"sovereign default",
        ]},
        {"name": "bond_success", "sign": +1, "patterns": [
            r"(?:eurobond|sukuk).{0,40}(?:rais|issu|success|oversubscrib)\w*",
        ]},
    ],

    "Energy": [
        {"name": "tariff_hike", "sign": -1, "patterns": [
            r"(?:electricity|power|gas) tariff.{0,30}(?:hik|rais|increas)\w*",
            r"(?:hik|rais|increas)\w*.{0,30}(?:electricity|power|gas) tariff",
        ]},
        {"name": "tariff_cut", "sign": +1, "patterns": [
            r"(?:electricity|power|gas) tariff.{0,30}(?:cut|reduc|lower)\w*",
            r"(?:cut|reduc|lower)\w*.{0,30}(?:electricity|power|gas) tariff",
            r"tariff relief",
        ]},
        {"name": "fuel_price_up", "sign": -1, "patterns": [
            r"(?:petrol|diesel|petroleum|fuel|pol) prices?.{0,20}(?:hik|rais|increas|jump|surg)\w*",
            r"(?:hik|rais|increas)\w*.{0,25}(?:petrol|diesel|petroleum) prices?",
        ]},
        {"name": "fuel_price_down", "sign": +1, "patterns": [
            r"(?:petrol|diesel|petroleum|fuel|pol) prices?.{0,20}(?:cut|reduc|lower|slash|fall)\w*",
            r"(?:cut|reduc|slash)\w*.{0,25}(?:petrol|diesel|petroleum) prices?",
        ]},
        {"name": "circular_debt_up", "sign": -1, "patterns": [
            r"circular debt.{0,40}(?:ris|swell|grow|climb|mount|cross|surg)\w*",
            r"mounting circular debt",
        ]},
        {"name": "circular_debt_down", "sign": +1, "patterns": [
            r"circular debt.{0,40}(?:reduc|settl|retir|clear|fall|declin)\w*",
        ]},
        {"name": "outages", "sign": -1, "patterns": [
            r"load ?shedding",
            r"power (?:outage|cut|breakdown|crisis)",
            r"(?:electricity|gas) shortage",
            r"gas (?:load management|curtailment|suspension)",
        ]},
        {"name": "supply_restored", "sign": +1, "patterns": [
            r"(?:gas|power|electricity) supply (?:restor|resum)\w*",
            r"lng cargo\w*.{0,20}(?:arriv|secur)\w*",
        ]},
    ],
}

# Pre-compile: {channel: [(rule_name, sign, compiled_pattern), ...]}
_COMPILED = {
    channel: [
        (rule["name"], rule["sign"], re.compile(pat, re.IGNORECASE))
        for rule in rules
        for pat in rule["patterns"]
    ]
    for channel, rules in RULES.items()
}
_ALL_COMPILED = [t for pats in _COMPILED.values() for t in pats]


# ---------------------------------------------------------------------------
# CORE SCORING
# ---------------------------------------------------------------------------
def score_article_direction(headline: str, body_text: str,
                            category: str, finbert_score: float) -> dict:
    """
    Apply the channel's directional rules to one article.

    Returns dict with:
      psx_score   : signed score in [-1, +1] for PSX equities
      psx_source  : "rule" (a directional event fired) or "finbert" (fallback)
      psx_rule_sign : +1 / -1 / 0
      psx_rules_fired : semicolon-joined rule names (with hit counts)
    """
    headline = str(headline) if headline == headline else ""     # NaN guard
    body     = str(body_text) if body_text == body_text else ""
    body     = body[:BODY_LEAD_CHARS]

    if category in _COMPILED:
        patterns = _COMPILED[category]
    else:                                   # Mixed / Unclassified -> all rules
        patterns = _ALL_COMPILED

    net = 0.0
    fired = {}
    for name, sign, regex in patterns:
        h_hits = len(regex.findall(headline))
        b_hits = len(regex.findall(body))
        if h_hits or b_hits:
            weight = HEADLINE_WEIGHT * h_hits + b_hits
            net   += sign * weight
            fired[name] = fired.get(name, 0) + weight

    finbert_score = float(finbert_score) if finbert_score == finbert_score else 0.0

    if net != 0:
        rule_sign = 1 if net > 0 else -1
        magnitude = abs(finbert_score) if abs(finbert_score) > 0 else NEUTRAL_MAGNITUDE
        return {
            "psx_score": round(rule_sign * magnitude, 4),
            "psx_source": "rule",
            "psx_rule_sign": rule_sign,
            "psx_rules_fired": ";".join(f"{k}({v})" for k, v in sorted(fired.items())),
        }

    return {
        "psx_score": round(finbert_score, 4),
        "psx_source": "finbert",
        "psx_rule_sign": 0,
        "psx_rules_fired": "",
    }


def apply_sign_rules(df: pd.DataFrame,
                     headline_col: str = "headline",
                     body_col: str = "body_text",
                     category_col: str = "topic_category",
                     finbert_col: str = "sentiment_score") -> pd.DataFrame:
    """
    Apply directional sign rules to a FinBERT-scored, topic-classified corpus.
    Adds columns: psx_score, psx_source, psx_rule_sign, psx_rules_fired.
    """
    print(f"\n--- Applying PSX sign rules to {len(df):,} articles ---")

    records = [
        score_article_direction(h, b, c, s)
        for h, b, c, s in zip(df[headline_col], df[body_col],
                              df[category_col], df[finbert_col])
    ]
    result = pd.DataFrame(records, index=df.index)
    df_out = pd.concat([df, result], axis=1)

    n_rule = (df_out["psx_source"] == "rule").sum()
    flipped = ((df_out["psx_source"] == "rule") &
               (np.sign(df_out[finbert_col]) != 0) &
               (np.sign(df_out[finbert_col]) != df_out["psx_rule_sign"])).sum()
    print(f"  Rule-directed articles : {n_rule:,} ({n_rule / len(df_out) * 100:.1f}%)")
    print(f"  FinBERT fallback       : {len(df_out) - n_rule:,}")
    print(f"  Sign flipped vs FinBERT: {flipped:,}")
    return df_out


# ---------------------------------------------------------------------------
# AUDIT EXPORT (for manual validation / paper appendix)
# ---------------------------------------------------------------------------
def export_audit(df: pd.DataFrame, output_path: str,
                 finbert_col: str = "sentiment_score",
                 sample_size: int = 300, seed: int = 42):
    """
    Write an Excel audit workbook:
      rule_frequency : how often each rule fired, and its agreement with FinBERT
      flipped_sample : random sample of articles where the rule REVERSED FinBERT
      rule_sample    : random sample of all rule-directed articles for validation
    """
    rule_df = df[df["psx_source"] == "rule"].copy()

    # -- Sheet 1: rule frequency ------------------------------------------
    rows = []
    for channel, rules in RULES.items():
        for rule in rules:
            name = rule["name"]
            mask = rule_df["psx_rules_fired"].str.contains(rf"\b{name}\(", regex=True)
            sub  = rule_df[mask]
            if len(sub) == 0:
                rows.append({"channel": channel, "rule": name,
                             "sign": rule["sign"], "articles": 0,
                             "agrees_with_finbert_pct": np.nan})
                continue
            nonneutral = sub[np.sign(sub[finbert_col]) != 0]
            agree = (np.sign(nonneutral[finbert_col]) == nonneutral["psx_rule_sign"]).mean() \
                    if len(nonneutral) else np.nan
            rows.append({"channel": channel, "rule": name,
                         "sign": rule["sign"], "articles": len(sub),
                         "agrees_with_finbert_pct": round(agree * 100, 1)
                                                    if agree == agree else np.nan})
    freq = pd.DataFrame(rows).sort_values("articles", ascending=False)

    # -- Sheets 2-3: validation samples -----------------------------------
    audit_cols = [c for c in ["article_id", "date", "headline", "topic_category",
                              finbert_col, "psx_score", "psx_source",
                              "psx_rules_fired"] if c in df.columns]

    flipped = rule_df[(np.sign(rule_df[finbert_col]) != 0) &
                      (np.sign(rule_df[finbert_col]) != rule_df["psx_rule_sign"])]
    flipped_sample = flipped[audit_cols].sample(
        min(sample_size, len(flipped)), random_state=seed)
    rule_sample = rule_df[audit_cols].sample(
        min(sample_size, len(rule_df)), random_state=seed)

    with pd.ExcelWriter(output_path) as xw:
        freq.to_excel(xw, sheet_name="rule_frequency", index=False)
        flipped_sample.to_excel(xw, sheet_name="flipped_sample", index=False)
        rule_sample.to_excel(xw, sheet_name="rule_sample", index=False)

    print(f"\n  Sign-rule audit workbook saved -> {output_path}")
    print(f"  ({len(freq)} rules | {len(flipped_sample)} flipped-sample rows | "
          f"{len(rule_sample)} rule-sample rows)")


# ---------------------------------------------------------------------------
# DEMO
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        # (headline, category, mock finbert score)
        ("SBP hikes policy rate by 150bps to combat inflation", "Monetary", +0.62),
        ("SBP cuts policy rate by 100bps amid falling inflation", "Monetary", +0.55),
        ("IMF approves $1.1 billion tranche for Pakistan under EFF", "External", +0.80),
        ("NEPRA approves 18 percent electricity tariff increase", "Energy", +0.41),
        ("Rupee hits record low against dollar", "Monetary", -0.70),
        ("Cement dispatches rise in October", "Unclassified", +0.30),
    ]
    for headline, cat, fb in tests:
        r = score_article_direction(headline, "", cat, fb)
        print(f"\n  {headline}")
        print(f"    channel={cat}  finbert={fb:+.2f}  ->  "
              f"psx={r['psx_score']:+.2f}  [{r['psx_source']}] {r['psx_rules_fired']}")
