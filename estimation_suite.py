"""
Project: Information Transmission and Sectoral Asymmetry in the KSE-100
Module : Phase 4b — Full Estimation Suite (v2, all sectors)
Author : Dr. Yasir Saeed (KUST)
Description:
  1. MARKET LEVEL. VAR(A): [r_Market, S_All] — the combined index against
     combined sentiment. VAR(B): [r_Market, four channel series]. AIC lag
     selection, stability check, bidirectional Granger tests, IRFs with
     500-replication Monte Carlo 95% bands, FEVD.
  2. SECTORAL. For each eligible sector: VAR [r_sector, S_Monetary,
     S_Fiscal, S_External, S_Energy]; bivariate Granger tests in BOTH
     directions (sentiment -> return = transmission; return -> sentiment
     = reverse causality / anticipation); IRF peak timing and cumulative
     22-day response; FEVD shares at 1/5/22 days. Fused into the
     all-sector sensitivity matrix.
  3. LEAD-LAG / ANTICIPATION. Market-efficiency test: regress each return
     on 5 lags AND 5 leads of each sentiment series (Newey-West HAC).
     Jointly significant LEADS mean prices move before publication —
     news is anticipated (Ramey 2011); jointly significant LAGS mean
     delayed absorption. Cross-correlation profiles k = -10..+10.
  4. ROBUSTNESS. Everything re-estimated on raw FinBERT series (no
     directional correction); comparison table directional vs raw.

OUTPUT   Results/tables/09..12_*.xlsx   Results/figures/08..12_*.png
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR

PROJECT_DIR = Path(r"D:\Stockmarket-Sentiment-Analysis")
PANEL_XLSX  = PROJECT_DIR / "panel_data_all.xlsx"
TAB = PROJECT_DIR / "Results" / "tables"
FIG = PROJECT_DIR / "Results" / "figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

CHANNELS = ["S_Monetary", "S_Fiscal", "S_External", "S_Energy"]
MAX_GRANGER_LAG = 5
IRF_H = 22
FEVD_H = [1, 5, 22]

ORIGINAL_FIVE = ["r_Commercial_Banks", "r_Oil_Gas_Exploration_Companies",
                 "r_Fertilizer", "r_Cement", "r_Technology_Communication"]


def granger_both(df, x, y, maxlag=MAX_GRANGER_LAG):
    """Bivariate Granger tests x->y and y->x. Returns per-direction dict:
    p-value at each lag + min p + significant lags list."""
    out = {}
    for cause, effect, tag in [(x, y, f"{x}->{y}"), (y, x, f"{y}->{x}")]:
        d = df[[effect, cause]].dropna()
        try:
            r = grangercausalitytests(d, maxlag=maxlag, verbose=False)
            ps = {lag: r[lag][0]["ssr_ftest"][1] for lag in range(1, maxlag + 1)}
        except Exception:
            ps = {lag: np.nan for lag in range(1, maxlag + 1)}
        out[tag] = ps
    return out


def fit_var(data, maxlags=10):
    model = VAR(data.dropna())
    sel = model.select_order(maxlags)
    lag = max(int(sel.selected_orders["aic"]), 1)
    res = model.fit(lag)
    stable = bool(np.all(np.abs(res.roots) > 1))  # statsmodels roots: >1 = stable
    return res, lag, stable


def fevd_shares(res, target, sources, horizons=FEVD_H):
    f = res.fevd(max(horizons))
    names = list(res.names)
    ti = names.index(target)
    rows = []
    for s in sources:
        si = names.index(s)
        rows.append({"target": target, "source": s,
                     **{f"FEVD_h{h}_pct": round(f.decomp[ti, h - 1, si] * 100, 3)
                        for h in horizons}})
    return rows


def irf_summary(res, target, sources, h=IRF_H):
    irf = res.irf(h)
    names = list(res.names)
    ti = names.index(target)
    rows = []
    for s in sources:
        si = names.index(s)
        v = irf.irfs[:, ti, si] * 1e4  # bps per unit innovation
        peak = int(np.argmax(np.abs(v[1:])) + 1)
        rows.append({"target": target, "shock": s,
                     "day1_bps": round(v[1], 2), "day3_bps": round(v[3], 2),
                     "peak_day": peak, "peak_bps": round(v[peak], 2),
                     "cum22_bps": round(v[1:].sum(), 1)})
    return rows


def analyse_sheet(sheet, tag):
    """Run the full suite on one panel variant ('directional' or 'raw')."""
    panel = pd.read_excel(PANEL_XLSX, sheet_name=sheet, parse_dates=["date"])
    suffix = "_raw" if tag == "raw" else ""
    chans = [c + suffix for c in CHANNELS]
    s_all = "S_All" + suffix
    r_cols = [c for c in panel.columns if c.startswith("r_") and c != "r_Market"]

    results = {}

    # ---------- 1. market level ----------
    varA, lagA, stableA = fit_var(panel[["r_Market", s_all]])
    varB, lagB, stableB = fit_var(panel[["r_Market"] + chans])
    results["market_meta"] = pd.DataFrame([
        {"system": "VAR A: r_Market + S_All", "aic_lag": lagA, "stable": stableA,
         "nobs": varA.nobs},
        {"system": "VAR B: r_Market + 4 channels", "aic_lag": lagB, "stable": stableB,
         "nobs": varB.nobs}])

    g_rows = []
    for s in [s_all] + chans:
        gb = granger_both(panel, s, "r_Market")
        for tag2, ps in gb.items():
            g_rows.append({"pair": tag2,
                           **{f"p_lag{l}": round(p, 4) for l, p in ps.items()},
                           "min_p": round(np.nanmin(list(ps.values())), 4),
                           "sig_lags_5pct": ",".join(str(l) for l, p in ps.items()
                                                     if p < 0.05)})
    results["granger_market"] = pd.DataFrame(g_rows)
    results["fevd_market"] = pd.DataFrame(
        fevd_shares(varA, "r_Market", [s_all]) +
        fevd_shares(varB, "r_Market", chans))
    results["irf_market"] = pd.DataFrame(
        irf_summary(varA, "r_Market", [s_all]) +
        irf_summary(varB, "r_Market", chans))

    if tag == "directional":
        # IRF figures with MC bands (market systems only)
        for res_, srcs, fname, title in [
                (varA, [s_all], "08_irf_market_SAll.png",
                 "r_Market response to S_All innovation"),
                (varB, chans, "09_irf_market_channels.png",
                 "r_Market response to channel sentiment innovations")]:
            irf = res_.irf(IRF_H)
            lo, up = irf.errband_mc(orth=False, repl=500, signif=0.05)
            names = list(res_.names)
            ti = names.index("r_Market")
            fig, axes = plt.subplots(1, len(srcs), figsize=(4.2 * len(srcs), 3.4),
                                     squeeze=False)
            for ax, s in zip(axes[0], srcs):
                si = names.index(s)
                ax.plot(irf.irfs[:, ti, si] * 1e4, color="#1F3864", lw=1.5)
                ax.fill_between(range(IRF_H + 1), lo[:, ti, si] * 1e4,
                                up[:, ti, si] * 1e4, alpha=0.15, color="#1F3864")
                ax.axhline(0, color="k", lw=0.5, ls="--")
                ax.set_title(f"r_Market <- {s}", fontsize=9)
                ax.set_xlabel("days"); ax.set_ylabel("bps")
            plt.suptitle(title, fontsize=11)
            plt.tight_layout(); plt.savefig(FIG / fname, dpi=150); plt.close()

    # ---------- 2. sectoral ----------
    sec_granger, sec_fevd, sec_irf = [], [], []
    for r in r_cols:
        sub = panel[[r] + chans].dropna()
        if len(sub) < 300:
            continue
        try:
            res_, lag, stable = fit_var(sub)
        except Exception:
            continue
        for c in chans:
            gb = granger_both(panel, c, r)
            ps_fwd = gb[f"{c}->{r}"]; ps_rev = gb[f"{r}->{c}"]
            sec_granger.append({
                "sector": r, "channel": c, "var_lag": lag, "stable": stable,
                "fwd_min_p": round(np.nanmin(list(ps_fwd.values())), 4),
                "fwd_sig_lags": ",".join(str(l) for l, p in ps_fwd.items() if p < 0.05),
                "rev_min_p": round(np.nanmin(list(ps_rev.values())), 4),
                "rev_sig_lags": ",".join(str(l) for l, p in ps_rev.items() if p < 0.05)})
        sec_fevd += fevd_shares(res_, r, chans)
        sec_irf += irf_summary(res_, r, chans)

    results["granger_sectors"] = pd.DataFrame(sec_granger)
    results["fevd_sectors"] = pd.DataFrame(sec_fevd)
    results["irf_sectors"] = pd.DataFrame(sec_irf)

    # ---------- 3. lead-lag / anticipation ----------
    ll_rows = []
    K = 5
    for r in ["r_Market"] + r_cols:
        for s in [s_all] + chans:
            d = panel[[r, s]].dropna().reset_index(drop=True)
            X = pd.DataFrame(index=d.index)
            for k in range(1, K + 1):
                X[f"lag{k}"] = d[s].shift(k)      # past sentiment
                X[f"lead{k}"] = d[s].shift(-k)    # future sentiment (anticipation)
            dat = pd.concat([d[r], X], axis=1).dropna()
            y = dat[r]
            Xc = sm.add_constant(dat.drop(columns=[r]))
            m = sm.OLS(y, Xc).fit(cov_type="HAC", cov_kwds={"maxlags": K})
            lag_names = [f"lag{k}" for k in range(1, K + 1)]
            lead_names = [f"lead{k}" for k in range(1, K + 1)]
            f_lag = m.f_test([f"{n} = 0" for n in lag_names])
            f_lead = m.f_test([f"{n} = 0" for n in lead_names])
            ll_rows.append({
                "return": r, "sentiment": s,
                "lags_F": round(float(f_lag.fvalue), 3),
                "lags_p": round(float(f_lag.pvalue), 4),
                "lags_sum_coef": round(float(m.params[lag_names].sum()), 4),
                "leads_F": round(float(f_lead.fvalue), 3),
                "leads_p": round(float(f_lead.pvalue), 4),
                "leads_sum_coef": round(float(m.params[lead_names].sum()), 4),
                "interpretation": (
                    "lagged impact + anticipation" if f_lag.pvalue < .05 and f_lead.pvalue < .05
                    else "lagged impact only" if f_lag.pvalue < .05
                    else "anticipation only" if f_lead.pvalue < .05
                    else "neither")})
    results["lead_lag"] = pd.DataFrame(ll_rows)

    if tag == "directional":
        # cross-correlation profiles for the market
        fig, axes = plt.subplots(1, 5, figsize=(19, 3.4), sharey=True)
        for ax, s in zip(axes, [s_all] + chans):
            d = panel[["r_Market", s]].dropna()
            ks = range(-10, 11)
            cc = [d["r_Market"].corr(d[s].shift(-k)) for k in ks]
            ax.bar(list(ks), cc, color=["#C00000" if k < 0 else "#1F3864" for k in ks])
            ax.axhline(0, color="k", lw=0.5)
            ax.axhline(2 / np.sqrt(len(d)), color="grey", lw=0.6, ls="--")
            ax.axhline(-2 / np.sqrt(len(d)), color="grey", lw=0.6, ls="--")
            ax.set_title(f"corr(r_Market[t], {s}[t+k])", fontsize=8)
            ax.set_xlabel("k (days)")
        plt.suptitle("Cross-correlation profiles — k<0: sentiment lags returns "
                     "(anticipation); k>0: sentiment leads returns (transmission)",
                     fontsize=10)
        plt.tight_layout(); plt.savefig(FIG / "11_ccf_market.png", dpi=150); plt.close()

    return results


# =====================================================================
print("=== directional (main) ===")
main_r = analyse_sheet("directional", "directional")
print("=== raw FinBERT (robustness) ===")
raw_r = analyse_sheet("raw_finbert", "raw")

# ---- save main tables -------------------------------------------------------
with pd.ExcelWriter(TAB / "09_granger_causality.xlsx") as xw:
    main_r["granger_market"].to_excel(xw, sheet_name="market", index=False)
    main_r["granger_sectors"].to_excel(xw, sheet_name="sectors_bidirectional", index=False)

with pd.ExcelWriter(TAB / "10_var_irf_fevd.xlsx") as xw:
    main_r["market_meta"].to_excel(xw, sheet_name="var_specs", index=False)
    main_r["fevd_market"].to_excel(xw, sheet_name="fevd_market", index=False)
    main_r["irf_market"].to_excel(xw, sheet_name="irf_market", index=False)
    main_r["fevd_sectors"].to_excel(xw, sheet_name="fevd_sectors", index=False)
    main_r["irf_sectors"].to_excel(xw, sheet_name="irf_sectors", index=False)

main_r["lead_lag"].to_excel(TAB / "11_lead_lag_anticipation.xlsx", index=False)

# ---- all-sector sensitivity heatmap (FEVD h22 + Granger significance) --------
sens = main_r["fevd_sectors"].pivot(index="target", columns="source",
                                    values="FEVD_h22_pct")
gm = main_r["granger_sectors"].pivot(index="sector", columns="channel",
                                     values="fwd_min_p")
fig, ax = plt.subplots(figsize=(7.5, 10))
im = ax.imshow(sens.values, cmap="YlOrRd", aspect="auto")
ax.set_xticks(range(len(sens.columns)))
ax.set_xticklabels([c.replace("S_", "") for c in sens.columns], rotation=0, fontsize=8)
ax.set_yticks(range(len(sens.index)))
ax.set_yticklabels([c.replace("r_", "").replace("_", " ")[:28] for c in sens.index],
                   fontsize=7)
for i, sec in enumerate(sens.index):
    for j, ch in enumerate(sens.columns):
        star = "*" if gm.loc[sec, ch] < 0.05 else ""
        ax.text(j, i, f"{sens.iloc[i, j]:.2f}{star}", ha="center", va="center",
                fontsize=6.5)
plt.colorbar(im, shrink=0.7, label="FEVD share at 22 days (%)")
ax.set_title("All-sector sensitivity matrix\n(cell = FEVD %, * = Granger-significant "
             "at 5%)", fontsize=10)
plt.tight_layout(); plt.savefig(FIG / "10_sensitivity_heatmap_all.png", dpi=150)
plt.close()
sens_out = sens.copy()
sens_out.columns = [c + "_FEVD22pct" for c in sens_out.columns]
sens_out.to_excel(TAB / "10b_sensitivity_matrix_all.xlsx")

# ---- robustness: directional vs raw ------------------------------------------
def sig_count(gdf, col="fwd_min_p"):
    return int((gdf[col] < 0.05).sum())

comp = pd.DataFrame([
    {"metric": "market: S_All -> r_Market min p",
     "directional": main_r["granger_market"].set_index("pair").loc["S_All->r_Market", "min_p"],
     "raw_finbert": raw_r["granger_market"].set_index("pair").loc["S_All_raw->r_Market", "min_p"]},
    {"metric": "sector-channel pairs Granger-significant (fwd, 5%)",
     "directional": sig_count(main_r["granger_sectors"]),
     "raw_finbert": sig_count(raw_r["granger_sectors"])},
    {"metric": "pairs with significant anticipation (leads, 5%)",
     "directional": int((main_r["lead_lag"]["leads_p"] < 0.05).sum()),
     "raw_finbert": int((raw_r["lead_lag"]["leads_p"] < 0.05).sum())},
    {"metric": "pairs with significant lagged impact (5%)",
     "directional": int((main_r["lead_lag"]["lags_p"] < 0.05).sum()),
     "raw_finbert": int((raw_r["lead_lag"]["lags_p"] < 0.05).sum())},
])
with pd.ExcelWriter(TAB / "12_robustness.xlsx") as xw:
    comp.to_excel(xw, sheet_name="directional_vs_raw", index=False)
    raw_r["granger_market"].to_excel(xw, sheet_name="market_granger_raw", index=False)
    raw_r["granger_sectors"].to_excel(xw, sheet_name="sector_granger_raw", index=False)
    raw_r["lead_lag"].to_excel(xw, sheet_name="lead_lag_raw", index=False)
    o5 = main_r["granger_sectors"][main_r["granger_sectors"]["sector"].isin(ORIGINAL_FIVE)]
    o5.to_excel(xw, sheet_name="original_five_subset", index=False)

print("\n=== headline numbers ===")
print(comp.to_string(index=False))
print("\nMarket Granger (directional):")
print(main_r["granger_market"][["pair", "min_p", "sig_lags_5pct"]].to_string(index=False))
print("\nEstimation suite complete.")
