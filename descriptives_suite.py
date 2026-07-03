"""
Project: Information Transmission and Sectoral Asymmetry in the KSE-100
Module : Phase 4a — Descriptive Statistics and Pre-Estimation Diagnostics
Author : Dr. Yasir Saeed (KUST)
Description: Produces every table and figure required before estimation:
  1. Descriptive statistics (returns and sentiment series)
  2. Time-series behaviour: monthly article flow by channel; sentiment
     series plots; cumulative sector return paths
  3. Correlation analysis: sentiment x sentiment, sentiment x returns,
     returns x returns (tables + heatmaps)
  4. Unit root tests: ADF and KPSS per series, plus Maddala-Wu (1999)
     Fisher-type panel unit root test combining individual ADF p-values
  5. VAR lag-order selection (AIC/BIC/FPE/HQIC) for the market system

OUTPUT   Results/tables/*.xlsx   Results/figures/*.png
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.vector_ar.var_model import VAR

PROJECT_DIR = Path(r"D:\Stockmarket-Sentiment-Analysis")
PANEL_XLSX  = PROJECT_DIR / "panel_data_all.xlsx"
ARTICLES    = PROJECT_DIR / "Phase 2 Outputs" / "article_channel_scores.csv"
TAB = PROJECT_DIR / "Results" / "tables"
FIG = PROJECT_DIR / "Results" / "figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

panel = pd.read_excel(PANEL_XLSX, sheet_name="directional", parse_dates=["date"])
r_cols = [c for c in panel.columns if c.startswith("r_")]
s_cols = [c for c in panel.columns if c.startswith("S_")]
print(f"Panel: {len(panel)} days | {len(r_cols)} return series | {len(s_cols)} sentiment series")

# ===================================================================
# 1. Descriptive statistics
# ===================================================================
def describe(cols, scale=1.0, unit=""):
    rows = []
    for c in cols:
        x = panel[c].dropna() * scale
        rows.append({"series": c, "N": len(x), f"mean{unit}": x.mean(),
                     f"std{unit}": x.std(), f"min{unit}": x.min(),
                     "p25": x.quantile(.25), "median": x.median(),
                     "p75": x.quantile(.75), f"max{unit}": x.max(),
                     "skewness": sps.skew(x), "excess_kurtosis": sps.kurtosis(x)})
    return pd.DataFrame(rows).round(4)

with pd.ExcelWriter(TAB / "05_descriptive_statistics.xlsx") as xw:
    describe(r_cols, 100, "_pct").to_excel(xw, sheet_name="returns_pct", index=False)
    describe(s_cols).to_excel(xw, sheet_name="sentiment", index=False)
print("descriptives saved")

# ===================================================================
# 2. Time-series behaviour
# ===================================================================
# 2a. monthly article flow by channel
art = pd.read_csv(ARTICLES, parse_dates=["eff_date"])
flow = (art.assign(month=art["eff_date"].dt.to_period("M").dt.to_timestamp())
           .groupby(["month", "channel"]).size().unstack(fill_value=0))
fig, ax = plt.subplots(figsize=(13, 4.5))
flow.plot(ax=ax, linewidth=1.2)
ax.set_title("Monthly article flow by sentiment channel (multi-label assignment)")
ax.set_ylabel("articles per month"); ax.set_xlabel("")
plt.tight_layout(); plt.savefig(FIG / "02_article_flow_by_channel.png", dpi=150)
plt.close()
flow.to_excel(TAB / "05b_monthly_article_flow.xlsx")

# 2b. sentiment series
fig, axes = plt.subplots(len(s_cols[:5]), 1, figsize=(13, 2.2 * 5), sharex=True)
for ax, c in zip(axes, ["S_All", "S_Monetary", "S_Fiscal", "S_External", "S_Energy"]):
    ax.plot(panel["date"], panel[c], linewidth=0.7, color="#1F3864")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_ylabel(c, fontsize=9)
axes[0].set_title("Daily PSX-directional sentiment series (weekend-shifted, multi-label)")
plt.tight_layout(); plt.savefig(FIG / "03_sentiment_series.png", dpi=150)
plt.close()

# 2c. cumulative return paths (market + original five for readability)
focus = ["r_Market", "r_Commercial_Banks", "r_Oil_Gas_Exploration_Companies",
         "r_Fertilizer", "r_Cement", "r_Technology_Communication"]
focus = [c for c in focus if c in panel.columns]
fig, ax = plt.subplots(figsize=(13, 5))
for c in focus:
    ax.plot(panel["date"], (1 + panel[c].fillna(0)).cumprod(),
            linewidth=1.2, label=c.replace("r_", "").replace("_", " "))
ax.legend(fontsize=8); ax.set_title("Cumulative equal-weighted return paths")
ax.set_ylabel("growth of 1 rupee")
plt.tight_layout(); plt.savefig(FIG / "04_cumulative_returns.png", dpi=150)
plt.close()
print("time-series figures saved")

# ===================================================================
# 3. Correlations
# ===================================================================
def heatmap(mat, title, path, size=(11, 9), fontsize=6):
    fig, ax = plt.subplots(figsize=size)
    im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels([c.replace("r_", "").replace("S_", "")[:18] for c in mat.columns],
                       rotation=90, fontsize=fontsize)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels([c.replace("r_", "").replace("S_", "")[:18] for c in mat.index],
                       fontsize=fontsize)
    plt.colorbar(im, shrink=0.8)
    ax.set_title(title)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

corr_ss = panel[s_cols].corr()
corr_rr = panel[r_cols].corr()
corr_sr = pd.DataFrame({s: [panel[r].corr(panel[s]) for r in r_cols] for s in s_cols},
                       index=r_cols)
with pd.ExcelWriter(TAB / "06_correlations.xlsx") as xw:
    corr_ss.round(3).to_excel(xw, sheet_name="sentiment_x_sentiment")
    corr_sr.round(3).to_excel(xw, sheet_name="returns_x_sentiment")
    corr_rr.round(3).to_excel(xw, sheet_name="returns_x_returns")
heatmap(corr_rr, "Sector return correlations", FIG / "05_corr_returns.png")
heatmap(corr_sr, "Contemporaneous return x sentiment correlations",
        FIG / "06_corr_returns_sentiment.png", size=(7, 9), fontsize=7)
heatmap(corr_ss, "Sentiment series correlations", FIG / "07_corr_sentiment.png",
        size=(6, 5), fontsize=8)
print("correlations saved")

# ===================================================================
# 4. Unit root tests
# ===================================================================
rows = []
for c in r_cols + s_cols:
    x = panel[c].dropna()
    adf_stat, adf_p, _, _, _, _ = adfuller(x, autolag="AIC")
    try:
        kp_stat, kp_p, _, _ = kpss(x, regression="c", nlags="auto")
    except Exception:
        kp_stat, kp_p = np.nan, np.nan
    rows.append({"series": c, "ADF_stat": round(adf_stat, 3), "ADF_p": round(adf_p, 4),
                 "KPSS_stat": round(kp_stat, 3), "KPSS_p": round(kp_p, 3),
                 "conclusion": "stationary" if (adf_p < 0.05 and (np.isnan(kp_p) or kp_p > 0.05))
                               else ("stationary (ADF)" if adf_p < 0.05 else "non-stationary?")})
ur = pd.DataFrame(rows)

# Maddala-Wu (1999) Fisher panel unit root: -2 sum(ln p_i) ~ chi2(2N)
def maddala_wu(cols, label):
    ps = ur[ur["series"].isin(cols)]["ADF_p"].clip(lower=1e-10)
    stat = float(-2 * np.log(ps).sum())
    dof = 2 * len(ps)
    return {"panel": label, "N_series": len(ps), "MW_chi2": round(stat, 1),
            "dof": dof, "p_value": round(1 - sps.chi2.cdf(stat, dof), 6)}

mw = pd.DataFrame([maddala_wu(r_cols, "sector returns"),
                   maddala_wu(s_cols, "sentiment series")])
with pd.ExcelWriter(TAB / "07_unit_root_tests.xlsx") as xw:
    ur.to_excel(xw, sheet_name="ADF_KPSS_by_series", index=False)
    mw.to_excel(xw, sheet_name="MaddalaWu_panel", index=False)
print("unit roots saved"); print(mw.to_string(index=False))

# ===================================================================
# 5. VAR lag selection (market system)
# ===================================================================
mkt_sys = panel[["r_Market", "S_All"]].dropna()
chan_sys = panel[["r_Market", "S_Monetary", "S_Fiscal", "S_External", "S_Energy"]].dropna()
rows = []
for name, data in [("r_Market + S_All", mkt_sys),
                   ("r_Market + 4 channel series", chan_sys)]:
    sel = VAR(data).select_order(10)
    for crit in ["aic", "bic", "fpe", "hqic"]:
        rows.append({"system": name, "criterion": crit.upper(),
                     "selected_lag": int(sel.selected_orders[crit])})
lag_sel = pd.DataFrame(rows)
lag_sel.to_excel(TAB / "08_lag_selection.xlsx", index=False)
print("lag selection saved"); print(lag_sel.to_string(index=False))
print("\nDiagnostics suite complete.")
