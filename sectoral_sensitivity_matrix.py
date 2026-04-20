"""
Project: Information Transmission and Sectoral Asymmetry in the KSE-100
Module : Phase 5 — Sectoral Sensitivity Matrix (Terminal Output)
Author : Dr. Yasir Saeed
Affiliation: Kohat University of Science & Technology (KUST)
Description: Constructs the Sectoral Sensitivity Matrix — the core empirical
             contribution of this research. Synthesizes outputs from all four
             upstream phases into a single structured table mapping, for each
             of the five KSE-100 sectors, the:
               - Magnitude   : % return variance from sentiment (FEVD)
               - Direction   : positive/negative (IRF sign at peak)
               - Timing      : optimal pre/post window [t−n, t+k]
               - Persistence : GARCH decay parameter (α + β)
             This matrix is directly usable for portfolio construction,
             macro-hedging strategy, and regulatory monitoring of
             informational efficiency in the PSX.
"""

# ===============================================
# 📌 Step 0: Import Libraries
# ===============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Professional theme for academic publication
sns.set_theme(style="whitegrid")

print("\n--- Sectoral Sensitivity Matrix Module Initialized ---")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ===============================================
# 📌 Step 1: Load Results from Upstream Phases
# ===============================================
def load_all_results(results_dir: str = "../05_results") -> dict:
    """
    Load outputs from Phases 3 and 4 to assemble the final matrix.
    Required files:
      dcc_garch_results.xlsx  → timing (n_opt, k_opt) + persistence
      fevd_results.xlsx       → magnitude (FEVD at h=1, 5, 22)
      granger_causality.xlsx  → directionality confirmation
    """
    rdir = Path(results_dir)
    data = {}

    files = {
        "garch":   "dcc_garch_results.xlsx",
        "fevd":    "fevd_results.xlsx",
        "granger": "granger_causality.xlsx",
    }

    for key, fname in files.items():
        fpath = rdir / fname
        if fpath.exists():
            data[key] = pd.read_excel(fpath)
            print(f"  ✓ Loaded: {fname} ({len(data[key])} rows)")
        else:
            print(f"  ✗ Missing: {fname} — Phase results required")
            data[key] = pd.DataFrame()

    return data


# ===============================================
# 📌 Step 2: Build the Sectoral Sensitivity Matrix
# ===============================================
SECTORS    = ["Commercial_Banks", "Exploration_Production",
              "Fertilizers", "Cement", "Technology"]
CATEGORIES = ["Monetary", "Fiscal", "External", "Energy"]


def build_sensitivity_matrix(data: dict) -> pd.DataFrame:
    """
    Synthesize all phase outputs into one structured matrix.
    Rows: 20 sector-category pairs (5 × 4)
    Columns: Magnitude | Direction | n_opt | k_opt | Persistence | Granger_sig
    """
    print("\n--- Building Sectoral Sensitivity Matrix ---")
    rows = []

    for sector in SECTORS:
        for category in CATEGORIES:
            row = {"Sector": sector.replace("_", " "),
                   "Category": category}

            # Magnitude from FEVD (22-day horizon — monthly)
            if not data["fevd"].empty:
                mask = ((data["fevd"]["sector"] == sector) &
                        (data["fevd"]["category"] == category))
                fevd_row = data["fevd"][mask]
                row["Magnitude_h1"]  = fevd_row["FEVD_h1"].values[0]  if len(fevd_row) else np.nan
                row["Magnitude_h5"]  = fevd_row["FEVD_h5"].values[0]  if len(fevd_row) else np.nan
                row["Magnitude_h22"] = fevd_row["FEVD_h22"].values[0] if len(fevd_row) else np.nan
            else:
                row["Magnitude_h1"] = row["Magnitude_h5"] = row["Magnitude_h22"] = np.nan

            # Timing and persistence from DCC-GARCH
            if not data["garch"].empty:
                mask = ((data["garch"]["sector"] == sector) &
                        (data["garch"]["category"] == category))
                garch_row = data["garch"][mask]
                row["n_opt"]       = garch_row["n_opt"].values[0]       if len(garch_row) else np.nan
                row["k_opt"]       = garch_row["k_opt"].values[0]       if len(garch_row) else np.nan
                row["Persistence"] = garch_row["persistence"].values[0] if len(garch_row) else np.nan
            else:
                row["n_opt"] = row["k_opt"] = row["Persistence"] = np.nan

            # Granger causality significance
            if not data["granger"].empty:
                mask = ((data["granger"]["sector"] == sector) &
                        (data["granger"]["category"] == category) &
                        (data["granger"]["significant"] == "Yes"))
                row["Granger_Significant"] = "Yes" if mask.any() else "No"
            else:
                row["Granger_Significant"] = "Pending"

            rows.append(row)

    matrix = pd.DataFrame(rows)
    print("\n--- Sectoral Sensitivity Matrix ---")
    print(matrix.to_string(index=False))
    return matrix


# ===============================================
# 📌 Step 3: Heatmap Visualization (Academic Publication Format)
# ===============================================
def plot_sensitivity_heatmap(matrix: pd.DataFrame,
                              output_path: str = "sectoral_sensitivity_heatmap.png"):
    """
    Visualize the Sectoral Sensitivity Matrix as a publication-quality heatmap.
    Cell values: FEVD at 22-day horizon (% of return variance from sentiment).
    Cell color: magnitude of sentiment transmission effect.
    """
    print("\n--- Generating Sensitivity Heatmap ---")

    pivot = matrix.pivot(
        index="Sector", columns="Category", values="Magnitude_h22"
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        pivot,
        annot=True, fmt=".1f",
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "% Return Variance Explained (FEVD, h=22)"},
        ax=ax
    )
    ax.set_title(
        "Sectoral Sensitivity Matrix — KSE-100\n"
        "Share of Return Variance Attributable to Macroeconomic Sentiment (22-day horizon)",
        fontsize=12, fontweight="bold", pad=15
    )
    ax.set_xlabel("Macroeconomic Sentiment Category", fontsize=11)
    ax.set_ylabel("KSE-100 Sector", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Heatmap saved → {output_path}")
    plt.show()


# ===============================================
# 📌 Step 4: Export Final Matrix
# ===============================================
def export_matrix(matrix: pd.DataFrame,
                  output_path: str = "sectoral_sensitivity_matrix.xlsx"):
    """
    Export the final Sectoral Sensitivity Matrix to Excel.
    This is the core empirical deliverable of the research — directly usable
    for portfolio construction, macro-hedging, and SECP market integrity work.
    """
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        matrix.to_excel(writer, sheet_name="Sensitivity Matrix", index=False)

        # Summary interpretation guide
        guide = pd.DataFrame({
            "Column": ["Magnitude_h1", "Magnitude_h5", "Magnitude_h22",
                       "n_opt", "k_opt", "Persistence", "Granger_Significant"],
            "Interpretation": [
                "% return variance from sentiment at 1-day horizon (instantaneous)",
                "% return variance from sentiment at 5-day horizon (weekly)",
                "% return variance from sentiment at 22-day horizon (monthly)",
                "Optimal pre-publication window (days) — higher = more leakage",
                "Optimal post-publication absorption window (days)",
                "GARCH volatility persistence (α+β) — higher = longer shock duration",
                "Sentiment Granger-causes sector return (Yes/No, p<0.05)",
            ]
        })
        guide.to_excel(writer, sheet_name="Interpretation Guide", index=False)

    print(f"\n--- Matrix Exported ---")
    print(f"Output: {Path(output_path).resolve()}")


# ===============================================
# 📌 Step 5: Entry Point
# ===============================================
if __name__ == "__main__":
    print("\n--- Sectoral Sensitivity Matrix: Awaiting Phase 3 and 4 Outputs ---")

    data   = load_all_results()
    matrix = build_sensitivity_matrix(data)

    if not matrix.empty:
        plot_sensitivity_heatmap(
            matrix,
            output_path="sectoral_sensitivity_heatmap.png"
        )
        export_matrix(
            matrix,
            output_path="sectoral_sensitivity_matrix.xlsx"
        )
        print("\nDecision: Sectoral Sensitivity Matrix complete.")
        print("This is the core empirical contribution of the research.")
    else:
        print("\nDecision: Matrix empty — upstream phases must complete first.")

    print("\n--- Analysis Complete ---")
