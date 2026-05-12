"""
Project: Information Transmission and Sectoral Asymmetry in the KSE-100
Module : Phase 3 — Panel VAR, Granger Causality, IRF, FEVD
Author : Dr. Yasir Saeed
Affiliation: Kohat University of Science & Technology (KUST)
Description: Estimates Panel VAR (Holtz-Eakin, Newey & Rosen, 1988) on the
             pooled sectoral panel to identify directional transmission —
             which sentiment categories Granger-cause which sectors, at what
             lag, and with what persistence. Produces Impulse Response
             Functions (IRFs) and Forecast Error Variance Decomposition
             (FEVD) to quantify the share of sectoral return variance
             attributable to each macroeconomic sentiment category at
             horizons of 1, 5, and 22 trading days.

References: Holtz-Eakin, D., Newey, W., & Rosen, H.S. (1988). Estimating
            vector autoregressions with panel data. Econometrica, 56(6).
            Love, I., & Zicchino, L. (2006). Financial development and
            dynamic investment behavior. QREF, 46(2), 190-210.
            Ramey, V.A. (2011). Identifying government spending shocks.
            QJE, 126(1), 1-50.
"""

# ===============================================
# Step 0: Import Libraries
# ===============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from pathlib import Path
from datetime import datetime

sns.set_theme(style="whitegrid", palette="magma")

print("\n--- Panel VAR Module Initialized ---")
print(f"Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Reference : Holtz-Eakin, Newey & Rosen (1988); Love & Zicchino (2006)")


# ===============================================
# Step 1: Study Parameters
# ===============================================
SECTORS    = ["Commercial_Banks", "Exploration_Production",
              "Fertilizers", "Cement", "Technology"]
CATEGORIES = ["Monetary", "Fiscal", "External", "Energy"]

# FEVD horizons: 1-day (instantaneous), 5-day (weekly), 22-day (monthly)
FEVD_HORIZONS = [1, 5, 22]

print(f"\n--- VAR System Variables ---")
print(f"Endogenous: {len(SECTORS)} sector returns + {len(CATEGORIES)} sentiment series")
print(f"System dimension: {len(SECTORS) + len(CATEGORIES)} variables")


# ===============================================
# Step 2: Granger Causality Tests
# ===============================================
def run_granger_causality(panel: pd.DataFrame,
                           max_lag: int = 5) -> pd.DataFrame:
    """
    Test whether each sentiment category Granger-causes each sectoral return.
    H0: sentiment does NOT Granger-cause sector returns.
    Decision: Reject H0 at p < 0.05 -> sentiment leads sector (causal signal).

    Ramey (2011): anticipated news shocks must be accounted for to avoid
    downward-biased estimates — panel includes lead sentiment terms.
    """
    print(f"\n--- Granger Causality Tests (max lag = {max_lag}) ---")
    results = []

    for sector in SECTORS:
        return_col = f"r_{sector}"
        if return_col not in panel.columns:
            continue

        for category in CATEGORIES:
            sentiment_col = f"S_{category}"
            if sentiment_col not in panel.columns:
                continue

            test_data = panel[[return_col, sentiment_col]].dropna()
            if len(test_data) < 50:
                continue

            try:
                gc_result = grangercausalitytests(
                    test_data, maxlag=max_lag, verbose=False
                )
                for lag in range(1, max_lag + 1):
                    f_pval = gc_result[lag][0]["ssr_ftest"][1]
                    results.append({
                        "sector":      sector,
                        "category":    category,
                        "lag":         lag,
                        "F_p_value":   round(f_pval, 4),
                        "significant": "Yes" if f_pval < 0.05 else "No",
                    })
            except Exception as e:
                results.append({
                    "sector": sector, "category": category,
                    "lag": None, "F_p_value": None,
                    "significant": f"Error: {e}"
                })

    gc_df = pd.DataFrame(results)

    print("\n--- Granger Causality Summary (p < 0.05 = significant) ---")
    sig_pairs = gc_df[gc_df["significant"] == "Yes"][
        ["sector", "category", "lag", "F_p_value"]
    ].drop_duplicates(subset=["sector", "category"]).head(20)
    print(sig_pairs.to_string(index=False))

    return gc_df


# ===============================================
# Step 3: VAR Estimation
# ===============================================
def estimate_var(panel: pd.DataFrame, lag_order: int = None) -> object:
    """
    Estimate reduced-form VAR on the joint system of sectoral returns
    and topic-specific sentiment series.
    Lag order selected by AIC if not specified.
    Returns fitted VAR model object for IRF and FEVD computation.
    """
    from statsmodels.tsa.vector_ar.var_model import VAR

    return_cols    = [f"r_{s}" for s in SECTORS    if f"r_{s}"  in panel.columns]
    sentiment_cols = [f"S_{c}" for c in CATEGORIES if f"S_{c}"  in panel.columns]
    var_cols       = return_cols + sentiment_cols

    data = panel[var_cols].dropna()
    print(f"\n--- VAR System: {len(var_cols)} variables, {len(data)} observations ---")

    model = VAR(data)

    if lag_order is None:
        lag_results = model.select_order(maxlags=10)
        lag_order   = lag_results.aic
        print(f"AIC selects lag order = {lag_order}")
    else:
        print(f"Lag order: {lag_order} (user-specified)")

    result = model.fit(lag_order)
    print(result.summary())
    return result


# ===============================================
# Step 4: Impulse Response Functions
# ===============================================
def compute_irf(var_result, periods: int = 22,
                output_path: str = "irf_plots.png"):
    """
    Compute and plot IRFs — dynamic response of each sector's return
    to a one-standard-deviation shock in each sentiment category.
    Horizon: 22 trading days (~1 month) to capture full absorption.

    Key interpretation:
      IRF peak before t=0  -> informational leakage (pre-announcement)
      IRF peak at t=1-3    -> rapid absorption (efficient market)
      IRF peak at t=5-22   -> slow absorption (illiquid sector)
    """
    print(f"\n--- Computing Impulse Response Functions ({periods} periods) ---")

    irf = var_result.irf(periods=periods)

    fig, axes = plt.subplots(
        len(SECTORS), len(CATEGORIES),
        figsize=(4 * len(CATEGORIES), 3 * len(SECTORS)),
        sharex=True
    )

    return_cols    = [f"r_{s}" for s in SECTORS]
    sentiment_cols = [f"S_{c}" for c in CATEGORIES]

    for i, r_col in enumerate(return_cols):
        for j, s_col in enumerate(sentiment_cols):
            ax = axes[i][j]
            if r_col in var_result.names and s_col in var_result.names:
                r_idx    = list(var_result.names).index(r_col)
                s_idx    = list(var_result.names).index(s_col)
                irf_vals = irf.irfs[:, r_idx, s_idx]
                ax.plot(range(periods + 1), irf_vals,
                        color="#1F3864", linewidth=1.5)
                ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
                ax.fill_between(range(periods + 1),
                                irf.lower[:, r_idx, s_idx],
                                irf.upper[:, r_idx, s_idx],
                                alpha=0.15, color="#1F3864")
            ax.set_title(f"{SECTORS[i].replace('_',' ')} <- {CATEGORIES[j]}",
                         fontsize=8)

    plt.suptitle("Impulse Response Functions — KSE-100 Sectoral Returns to Sentiment Shocks",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  IRF plots saved -> {output_path}")
    plt.show()

    return irf


# ===============================================
# Step 5: Forecast Error Variance Decomposition
# ===============================================
def compute_fevd(var_result, horizons: list = FEVD_HORIZONS) -> pd.DataFrame:
    """
    Compute FEVD — share of sectoral return variance attributable to
    each sentiment category at horizons of 1, 5, and 22 trading days.
    Provides magnitude ranking of policy channel importance across sectors.
    Terminal output feeds directly into the Sectoral Sensitivity Matrix.
    """
    print(f"\n--- Forecast Error Variance Decomposition ---")
    print(f"Horizons: {horizons} trading days")

    fevd     = var_result.fevd(periods=max(horizons))
    fevd_tbl = fevd.decomp

    results        = []
    return_cols    = [f"r_{s}" for s in SECTORS]
    sentiment_cols = [f"S_{c}" for c in CATEGORIES]

    for r_col in return_cols:
        if r_col not in var_result.names:
            continue
        r_idx = list(var_result.names).index(r_col)

        for s_col in sentiment_cols:
            if s_col not in var_result.names:
                continue
            s_idx = list(var_result.names).index(s_col)

            row = {
                "sector":   r_col.replace("r_", ""),
                "category": s_col.replace("S_", ""),
            }
            for h in horizons:
                share     = fevd_tbl[r_idx, h - 1, s_idx]
                row[f"FEVD_h{h}"] = round(share * 100, 2)

            results.append(row)

    fevd_df = pd.DataFrame(results)

    print("\n--- FEVD Results (% of return variance explained by sentiment) ---")
    print(fevd_df.to_string(index=False))
    return fevd_df


# ===============================================
# Step 6: Entry Point
# ===============================================
if __name__ == "__main__":
    print("\n--- Panel VAR Module: Awaiting Panel Data ---")
    print("Required input: panel_data.xlsx")
    print("  (merged KSE-100 sector returns + daily_sentiment.xlsx from Phase 2)")

    panel_path = Path("panel_data.xlsx")

    if panel_path.exists():
        panel = pd.read_excel(panel_path, parse_dates=["date"])

        gc_df = run_granger_causality(panel)
        gc_df.to_excel("granger_causality.xlsx", index=False)

        var_result = estimate_var(panel)

        irf = compute_irf(var_result, periods=22, output_path="irf_plots.png")

        fevd_df = compute_fevd(var_result)
        fevd_df.to_excel("fevd_results.xlsx", index=False)

        print("\nPhase 3 outputs saved:")
        print("  granger_causality.xlsx")
        print("  irf_plots.png")
        print("  fevd_results.xlsx")
    else:
        print("\nPanel data not yet available.")
        print("Phases 1 and 2 must complete first.")

    print("\n--- Analysis Complete ---")
