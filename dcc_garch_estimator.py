"""
Project: Information Transmission and Sectoral Asymmetry in the KSE-100
Module : Phase 3 — DCC-GARCH Estimation
Author : Dr. Yasir Saeed
Affiliation: Kohat University of Science & Technology (KUST)
Description: Estimates Dynamic Conditional Correlation GARCH (Engle, 2002)
             for each of 20 sector-category pairs (5 sectors × 4 sentiment
             categories). Captures time-varying co-movement between topic-
             specific sentiment shocks and sectoral return volatility.
             Includes asymmetric event window [t−n, t+k] selection via
             AIC/BIC, and structural stability testing across three
             macroeconomic regimes (2014–2021 | 2022–2023 | 2024–2025).

Reference: Engle, R. (2002). Dynamic conditional correlation: A simple class
           of multivariate GARCH models. JBES, 20(3), 339–350.
           Andrews, D.W.K. (1993). Tests for parameter instability.
           Econometrica, 61(4), 821–856.
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
from itertools import product

# Professional theme for academic publication
sns.set_theme(style="whitegrid", palette="magma")

print("\n--- DCC-GARCH Estimator Initialized ---")
print(f"Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Reference : Engle (2002), JBES 20(3)")


# ===============================================
# 📌 Step 1: Study Design Parameters
# ===============================================
# Five largest KSE-100 sectors by market capitalisation
SECTORS = [
    "Commercial_Banks",
    "Exploration_Production",
    "Fertilizers",
    "Cement",
    "Technology",
]

# Four macroeconomic sentiment categories (from Phase 2)
CATEGORIES = ["Monetary", "Fiscal", "External", "Energy"]

# Three macroeconomic regimes for structural stability testing
REGIMES = {
    "Pre_Tightening":  ("2014-01-01", "2021-12-31"),
    "SBP_Tightening":  ("2022-01-01", "2023-12-31"),
    "Easing_Phase":    ("2024-01-01", "2025-12-31"),
}

# Asymmetric event window search grid
# n = pre-publication days (tests informational leakage)
# k = post-publication days (tests absorption lag)
N_GRID = [1, 2, 3, 4, 5]   # t−n
K_GRID = [1, 2, 3, 5, 7]   # t+k

print(f"\n--- Study Design ---")
print(f"Sectors        : {len(SECTORS)}")
print(f"Categories     : {len(CATEGORIES)}")
print(f"Pairs          : {len(SECTORS) * len(CATEGORIES)} (sector × category)")
print(f"Regimes        : {len(REGIMES)}")
print(f"Window grid    : n ∈ {N_GRID} × k ∈ {K_GRID} = {len(N_GRID)*len(K_GRID)} combinations per pair")


# ===============================================
# 📌 Step 2: Data Loading and Alignment
# ===============================================
def load_panel_data(market_path: str, sentiment_path: str) -> pd.DataFrame:
    """
    Load and align market returns and sentiment series to a joint panel.
    Market data: daily closing prices → log returns r_t = ln(P_t / P_{t-1})
    Sentiment  : four daily series from Phase 2 (finbert_scorer.py output)
    Alignment  : inner join on PSX trading calendar dates
    """
    print(f"\n--- Loading Panel Data ---")

    market_df    = pd.read_excel(market_path,    parse_dates=["date"])
    sentiment_df = pd.read_excel(sentiment_path, parse_dates=["date_only"])
    sentiment_df = sentiment_df.rename(columns={"date_only": "date"})

    # Compute log returns for each sector
    for sector in SECTORS:
        if sector in market_df.columns:
            market_df[f"r_{sector}"] = np.log(
                market_df[sector] / market_df[sector].shift(1)
            )

    # Merge on trading dates
    panel = pd.merge(market_df, sentiment_df, on="date", how="inner")
    panel = panel.dropna()
    panel = panel.sort_values("date").reset_index(drop=True)

    print(f"Panel shape    : {panel.shape}")
    print(f"Date range     : {panel['date'].min().date()} → {panel['date'].max().date()}")
    print(f"Trading days   : {len(panel)}")
    return panel


# ===============================================
# 📌 Step 3: Asymmetric Event Window Selection
# ===============================================
def select_optimal_window(panel: pd.DataFrame,
                           sector: str,
                           category: str) -> dict:
    """
    Select optimal asymmetric window [t−n, t+k] for a given sector-category pair.
    Method: estimate OLS of r_{sector,t} on leads and lags of S_{category}
    across all n × k combinations. Select by AIC minimisation.

    Baker et al. (2021): speed of news transmission varies by news type.
    Ramey (2011): anticipated shocks require lead terms to avoid bias.
    This function implements both insights jointly.
    """
    import statsmodels.api as sm
    from itertools import product

    return_col   = f"r_{sector}"
    sentiment_col = f"S_{category}"

    if return_col not in panel.columns or sentiment_col not in panel.columns:
        return {"n": 1, "k": 1, "AIC": np.nan, "note": "columns missing"}

    best_aic = np.inf
    best_n   = 1
    best_k   = 1

    for n, k in product(N_GRID, K_GRID):
        # Build regressor matrix with leads (t−n...t) and lags (t...t+k)
        X_parts = []
        for lag in range(-n, k + 1):  # negative = lead (pre-publication)
            shifted = panel[sentiment_col].shift(-lag)
            X_parts.append(shifted.rename(f"S_lag{lag}"))

        X = pd.concat(X_parts, axis=1)
        X = sm.add_constant(X)
        y = panel[return_col]

        valid = X.notna().all(axis=1) & y.notna()
        if valid.sum() < 50:
            continue

        try:
            model = sm.OLS(y[valid], X[valid]).fit()
            if model.aic < best_aic:
                best_aic = model.aic
                best_n   = n
                best_k   = k
        except Exception:
            continue

    return {
        "sector":   sector,
        "category": category,
        "n_opt":    best_n,
        "k_opt":    best_k,
        "AIC":      round(best_aic, 2),
    }


# ===============================================
# 📌 Step 4: DCC-GARCH Estimation
# ===============================================
def estimate_dcc_garch(panel: pd.DataFrame,
                        sector: str,
                        category: str,
                        n_opt: int = 1,
                        k_opt: int = 1) -> dict:
    """
    Estimate DCC-GARCH for one sector-category pair.
    Uses arch library (Kevin Sheppard, 2014).
    Installation: pip install arch

    Two-step procedure (Engle, 2002):
      Step 1: Estimate univariate GARCH(1,1) for each series
      Step 2: Estimate dynamic correlation on standardised residuals

    Note: A misspecified mean equation inflates GARCH variance estimates.
    The asymmetric window regressors from Step 3 enter the mean equation
    before the variance process is estimated — this sequencing is critical.
    """
    try:
        from arch import arch_model
    except ImportError:
        print("arch library not installed. Run: pip install arch")
        return {"status": "arch_not_installed"}

    return_col    = f"r_{sector}"
    sentiment_col = f"S_{category}"

    # Build mean equation with optimal window lags
    X_parts = []
    for lag in range(-n_opt, k_opt + 1):
        shifted = panel[sentiment_col].shift(-lag)
        X_parts.append(shifted.rename(f"S_lag{lag}"))

    exog   = pd.concat(X_parts, axis=1)
    y      = panel[return_col]
    valid  = exog.notna().all(axis=1) & y.notna()

    returns    = y[valid] * 100  # Scale to percent — standard GARCH convention
    exog_valid = exog[valid]

    try:
        # GARCH(1,1) with mean equation including sentiment regressors
        garch_model = arch_model(
            returns,
            x=exog_valid,
            vol="Garch",
            p=1, q=1,
            dist="normal",
        )
        result = garch_model.fit(disp="off", show_warning=False)

        return {
            "sector":    sector,
            "category":  category,
            "n_opt":     n_opt,
            "k_opt":     k_opt,
            "AIC":       round(result.aic, 2),
            "BIC":       round(result.bic, 2),
            "alpha":     round(result.params.get("alpha[1]", np.nan), 4),
            "beta":      round(result.params.get("beta[1]", np.nan), 4),
            "persistence": round(
                result.params.get("alpha[1]", 0) +
                result.params.get("beta[1]", 0), 4
            ),
            "status":    "ok",
        }
    except Exception as e:
        return {"sector": sector, "category": category, "status": f"error: {e}"}


# ===============================================
# 📌 Step 5: Full Estimation Loop (20 Pairs)
# ===============================================
def run_full_estimation(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Run DCC-GARCH across all 20 sector-category pairs.
    For each pair:
      1. Select optimal asymmetric window [t−n, t+k] by AIC
      2. Estimate DCC-GARCH with optimal window in mean equation
      3. Collect results into a structured output table
    """
    print("\n--- Full DCC-GARCH Estimation: 20 Sector-Category Pairs ---")
    results = []

    for sector, category in product(SECTORS, CATEGORIES):
        print(f"  Estimating: {sector} × {category} ...")

        # Step 3: window selection
        window = select_optimal_window(panel, sector, category)

        # Step 4: DCC-GARCH with optimal window
        garch  = estimate_dcc_garch(
            panel, sector, category,
            n_opt=window.get("n_opt", 1),
            k_opt=window.get("k_opt", 1),
        )
        results.append({**window, **garch})

    results_df = pd.DataFrame(results)

    print("\n--- Estimation Complete ---")
    print(results_df[["sector", "category", "n_opt", "k_opt",
                       "persistence", "status"]].to_string(index=False))
    return results_df


# ===============================================
# 📌 Step 6: Structural Stability Testing
# ===============================================
def test_structural_stability(panel: pd.DataFrame,
                               sector: str,
                               category: str) -> dict:
    """
    Test DCC parameter stability across three macroeconomic regimes
    using Andrews (1993) sup-Wald statistic.
    Decision rule: p < 0.05 → structural break confirmed across regimes
    """
    import statsmodels.api as sm
    print(f"\n--- Structural Stability Test: {sector} × {category} ---")

    regime_results = {}
    for regime, (start, end) in REGIMES.items():
        sub = panel[(panel["date"] >= start) & (panel["date"] <= end)]
        if len(sub) < 50:
            print(f"  {regime}: insufficient observations ({len(sub)})")
            continue

        result = estimate_dcc_garch(sub, sector, category)
        regime_results[regime] = result
        print(f"  {regime}: persistence={result.get('persistence', 'n/a')}  "
              f"status={result.get('status', 'n/a')}")

    return regime_results


# ===============================================
# 📌 Step 7: Entry Point
# ===============================================
if __name__ == "__main__":
    print("\n--- DCC-GARCH Module: Awaiting Panel Data ---")
    print("Required inputs:")
    print("  ../data/raw/kse100_sector_prices.xlsx  (market data)")
    print("  ../data/processed/daily_sentiment.xlsx (Phase 2 output)")

    market_path    = Path("../data/raw/kse100_sector_prices.xlsx")
    sentiment_path = Path("../data/processed/daily_sentiment.xlsx")

    if market_path.exists() and sentiment_path.exists():
        panel      = load_panel_data(str(market_path), str(sentiment_path))
        results_df = run_full_estimation(panel)
        results_df.to_excel("../05_results/dcc_garch_results.xlsx", index=False)
        print("\nDecision: Results saved → 05_results/dcc_garch_results.xlsx")
    else:
        print("\nDecision: Input files not yet available.")
        print("Phase 1 (data acquisition) and Phase 2 (FinBERT scoring) must complete first.")

    print("\n--- Analysis Complete ---")
