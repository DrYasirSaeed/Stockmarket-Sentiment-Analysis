"""
Project: Information Transmission and Sectoral Asymmetry in the KSE-100
Module : Phase 4b — Full Estimation Suite (v2.1, integration-order corrected)
Author : Dr. Yasir Saeed (KUST)
Description:
  v2.1 corrects the treatment of non-stationary sentiment series found by
  the unit-root tests (S_All, S_Monetary, S_External are I(1) in levels;
  S_Fiscal, S_Energy and all return series are I(0)):

  1. CAUSALITY — Toda-Yamamoto (1995) lag-augmented tests on LEVELS.
     For each pair, a bivariate system with p + dmax lags is estimated by
     OLS (dmax = highest integration order in the pair) and a Wald chi2
     test is applied to the first p lags of the cause variable only. The
     augmentation lag absorbs the unit root, making the test valid
     regardless of integration/cointegration. Reported for p = 1..5,
     both directions.

  2. VAR / IRF / FEVD / LEAD-LAG — estimated on a TRANSFORMED system in
     which every I(1) sentiment series enters in first differences
     (dS_* = daily sentiment CHANGE, the 'sentiment news' innovation);
     I(0) series and returns enter in levels. Determined per sheet by
     ADF at 5% (stationarity.transform_for_var).

  Everything else unchanged: market systems (r_Market with S_All and the
  four channels), 25 sectoral VARs, MC error bands, lead-lag anticipation
  regressions with Newey-West HAC, robustness on raw FinBERT series.

OUTPUT   Results/tables/09..12_*.xlsx   Results/figures/08..12_*.png
"""

import matplotlib
matplotlib.use("Agg")

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.var_model import VAR

PROJECT_DIR = Path(r"D:\Stockmarket-Sentiment-Analysis")
sys.path.insert(0, str(PROJECT_DIR))
from stationarity import integration_table, transform_for_var   # noqa: E402

PANEL_XLSX  = PROJECT_DIR / "panel_data_all.xlsx"
TAB = PROJECT_DIR / "Results" / "tables"
FIG = PROJECT_DIR / "Results" / "figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

CHANNELS = ["S_Monetary", "S_Fiscal", "S_External", "S_Energy"]
MAX_P = 5
IRF_H = 22
FEVD_H = [1, 5, 22]

ORIGINAL_FIVE = ["r_Commercial_Banks", "r_Oil_Gas_Exploration_Companies",
                 "r_Fertilizer", "r_Cement", "r_Technology_Communication"]


# ------------------------------------------------------------------ TY test --
def toda_yamamoto(df, cause, effect, p, dmax):
    """Lag-augmented Granger non-causality Wald test (levels).
    OLS: effect_t on const + lags 1..p+dmax of effect and cause;
    H0: coefficients on cause lags 1..p are zero (chi2)."""
    d = df[[effect, cause]].dropna().reset_index(drop=True)
    total = p + dmax
    X = pd.DataFrame(index=d.index)
    for k in range(1, total + 1):
        X[f"eff_l{k}"] = d[effect].shift(k)
        X[f"cau_l{k}"] = d[cause].shift(k)
    dat = pd.concat([d[effect].rename("y"), X], axis=1).dropna()
    m = sm.OLS(dat["y"], sm.add_constant(dat.drop(columns=["y"]))).fit()
    w = m.wald_test([f"cau_l{k} = 0" for k in range(1, p + 1)],
                    use_f=False, scalar=True)
    return float(w.pvalue)


def ty_both(df, x, y, dmax, max_p=MAX_P):
    """TY tests x->y and y->x at p = 1..max_p."""
    out = {}
    for cause, effect, tag in [(x, y, f"{x}->{y}"), (y, x, f"{y}->{x}")]:
        ps = {}
        for p in range(1, max_p + 1):
            try:
                ps[p] = toda_yamamoto(df, cause, effect, p, dmax)
            except Exception:
                ps[p] = np.nan
        out[tag] = ps
    return out


def fit_var(data, maxlags=10):
    model = VAR(data.dropna())
    sel = model.select_order(maxlags)
    lag = max(int(sel.selected_orders["aic"]), 1)
    res = model.fit(lag)
    stable = bool(np.all(np.abs(res.roots) > 1))
    return res, lag, stable


def fevd_shares(res, target, sources, horizons=FEVD_H):
    f = res.fevd(max(horizons))
    names = list(res.names)
    ti = names.index(target)
    return [{"target": target, "source": s,
             **{f"FEVD_h{h}_pct": round(f.decomp[ti, h - 1, names.index(s)] * 100, 3)
                for h in horizons}} for s in sources]


def irf_summary(res, target, sources, h=IRF_H):
    irf = res.irf(h)
    names = list(res.names)
    ti = names.index(target)
    rows = []
    for s in sources:
        v = irf.irfs[:, ti, names.index(s)] * 1e4
        peak = int(np.argmax(np.abs(v[1:])) + 1)
        rows.append({"target": target, "shock": s,
                     "day1_bps": round(v[1], 2), "day3_bps": round(v[3], 2),
                     "peak_day": peak, "peak_bps": round(v[peak], 2),
                     "cum22_bps": round(v[1:].sum(), 1)})
    return rows


def analyse_sheet(sheet, tag):
    panel = pd.read_excel(PANEL_XLSX, sheet_name=sheet, parse_dates=["date"])
    suffix = "_raw" if tag == "raw" else ""
    chans = [c + suffix for c in CHANNELS]
    s_all = "S_All" + suffix
    s_cols = [s_all] + chans
    r_cols = [c for c in panel.columns if c.startswith("r_") and c != "r_Market"]

    # ---- integration orders + transformed frame -----------------------------
    itab = integration_table(panel, s_cols)
    tpanel, nmap, orders = transform_for_var(panel, s_cols, itab)
    u_all, u_chans = nmap[s_all], [nmap[c] for c in chans]
    print(f"[{tag}] variables entering VAR: {u_all}, {u_chans}")

    results = {"integration": itab, "name_map": pd.DataFrame(
        [{"original": k, "used_in_VAR": v,
          "treatment": "first difference" if v != k else "level"}
         for k, v in nmap.items()])}

    # ---- 1. market level ------------------------------------------------------
    varA, lagA, stableA = fit_var(tpanel[["r_Market", u_all]])
    varB, lagB, stableB = fit_var(tpanel[["r_Market"] + u_chans])
    results["market_meta"] = pd.DataFrame([
        {"system": f"VAR A: r_Market + {u_all}", "aic_lag": lagA,
         "stable": stableA, "nobs": varA.nobs},
        {"system": f"VAR B: r_Market + {', '.join(u_chans)}", "aic_lag": lagB,
         "stable": stableB, "nobs": varB.nobs}])

    g_rows = []
    for s in s_cols:
        dmax = max(orders.get(s, 0), 0)   # returns are I(0)
        gb = ty_both(panel, s, "r_Market", dmax)
        for tag2, ps in gb.items():
            g_rows.append({"pair": tag2, "dmax": dmax,
                           **{f"TY_p_lag{l}": round(p, 4) for l, p in ps.items()},
                           "min_p": round(np.nanmin(list(ps.values())), 4),
                           "sig_lags_5pct": ",".join(str(l) for l, p in ps.items()
                                                     if p < 0.05)})
    results["granger_market"] = pd.DataFrame(g_rows)
    results["fevd_market"] = pd.DataFrame(
        fevd_shares(varA, "r_Market", [u_all]) +
        fevd_shares(varB, "r_Market", u_chans))
    results["irf_market"] = pd.DataFrame(
        irf_summary(varA, "r_Market", [u_all]) +
        irf_summary(varB, "r_Market", u_chans))

    if tag == "directional":
        for res_, srcs, fname, title in [
                (varA, [u_all], "08_irf_market_SAll.png",
                 f"r_Market response to {u_all} innovation"),
                (varB, u_chans, "09_irf_market_channels.png",
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
            plt.suptitle(title + "  (I(1) sentiment enters differenced)", fontsize=10)
            plt.tight_layout(); plt.savefig(FIG / fname, dpi=150); plt.close()

    # ---- 2. sectoral -----------------------------------------------------------
    sec_granger, sec_fevd, sec_irf = [], [], []
    for r in r_cols:
        sub = tpanel[[r] + u_chans].dropna()
        if len(sub) < 300:
            continue
        try:
            res_, lag, stable = fit_var(sub)
        except Exception:
            continue
        for c_orig, c_used in zip(chans, u_chans):
            dmax = max(orders.get(c_orig, 0), 0)
            gb = ty_both(panel, c_orig, r, dmax)
            ps_fwd = gb[f"{c_orig}->{r}"]; ps_rev = gb[f"{r}->{c_orig}"]
            sec_granger.append({
                "sector": r, "channel": c_orig, "dmax": dmax, "var_lag": lag,
                "stable": stable,
                "fwd_min_p": round(np.nanmin(list(ps_fwd.values())), 4),
                "fwd_sig_lags": ",".join(str(l) for l, p in ps_fwd.items() if p < 0.05),
                "rev_min_p": round(np.nanmin(list(ps_rev.values())), 4),
                "rev_sig_lags": ",".join(str(l) for l, p in ps_rev.items() if p < 0.05)})
        sec_fevd += fevd_shares(res_, r, u_chans)
        sec_irf += irf_summary(res_, r, u_chans)

    results["granger_sectors"] = pd.DataFrame(sec_granger)
    results["fevd_sectors"] = pd.DataFrame(sec_fevd)
    results["irf_sectors"] = pd.DataFrame(sec_irf)

    # ---- 3. lead-lag / anticipation (transformed sentiment) --------------------
    ll_rows = []
    K = 5
    for r in ["r_Market"] + r_cols:
        for s_orig in s_cols:
            s = nmap[s_orig]
            d = tpanel[[r, s]].dropna().reset_index(drop=True)
            X = pd.DataFrame(index=d.index)
            for k in range(1, K + 1):
                X[f"lag{k}"] = d[s].shift(k)
                X[f"lead{k}"] = d[s].shift(-k)
            dat = pd.concat([d[r], X], axis=1).dropna()
            m = sm.OLS(dat[r], sm.add_constant(dat.drop(columns=[r]))
                       ).fit(cov_type="HAC", cov_kwds={"maxlags": K})
            lag_n = [f"lag{k}" for k in range(1, K + 1)]
            lead_n = [f"lead{k}" for k in range(1, K + 1)]
            f_lag = m.f_test([f"{n} = 0" for n in lag_n])
            f_lead = m.f_test([f"{n} = 0" for n in lead_n])
            ll_rows.append({
                "return": r, "sentiment": s,
                "lags_F": round(float(f_lag.fvalue), 3),
                "lags_p": round(float(f_lag.pvalue), 4),
                "lags_sum_coef": round(float(m.params[lag_n].sum()), 4),
                "leads_F": round(float(f_lead.fvalue), 3),
                "leads_p": round(float(f_lead.pvalue), 4),
                "leads_sum_coef": round(float(m.params[lead_n].sum()), 4),
                "interpretation": (
                    "lagged impact + anticipation" if f_lag.pvalue < .05 and f_lead.pvalue < .05
                    else "lagged impact only" if f_lag.pvalue < .05
                    else "anticipation only" if f_lead.pvalue < .05
                    else "neither")})
    results["lead_lag"] = pd.DataFrame(ll_rows)

    if tag == "directional":
        fig, axes = plt.subplots(1, 5, figsize=(19, 3.4), sharey=True)
        for ax, s_orig in zip(axes, s_cols):
            s = nmap[s_orig]
            d = tpanel[["r_Market", s]].dropna()
            ks = range(-10, 11)
            cc = [d["r_Market"].corr(d[s].shift(-k)) for k in ks]
            ax.bar(list(ks), cc, color=["#C00000" if k < 0 else "#1F3864" for k in ks])
            ax.axhline(0, color="k", lw=0.5)
            ax.axhline(2 / np.sqrt(len(d)), color="grey", lw=0.6, ls="--")
            ax.axhline(-2 / np.sqrt(len(d)), color="grey", lw=0.6, ls="--")
            ax.set_title(f"corr(r_Market[t], {s}[t+k])", fontsize=8)
            ax.set_xlabel("k (days)")
        plt.suptitle("Cross-correlation profiles (stationary transforms) — k<0: "
                     "returns lead sentiment (anticipation); k>0: sentiment leads "
                     "returns (transmission)", fontsize=10)
        plt.tight_layout(); plt.savefig(FIG / "11_ccf_market.png", dpi=150); plt.close()

    return results


# =====================================================================
print("=== directional (main) ===")
main_r = analyse_sheet("directional", "directional")
print("=== raw FinBERT (robustness) ===")
raw_r = analyse_sheet("raw_finbert", "raw")

with pd.ExcelWriter(TAB / "09_granger_causality.xlsx") as xw:
    main_r["granger_market"].to_excel(xw, sheet_name="market_TY", index=False)
    main_r["granger_sectors"].to_excel(xw, sheet_name="sectors_TY_bidirectional", index=False)
    main_r["integration"].to_excel(xw, sheet_name="integration_orders", index=False)
    main_r["name_map"].to_excel(xw, sheet_name="var_treatment", index=False)

with pd.ExcelWriter(TAB / "10_var_irf_fevd.xlsx") as xw:
    main_r["market_meta"].to_excel(xw, sheet_name="var_specs", index=False)
    main_r["name_map"].to_excel(xw, sheet_name="var_treatment", index=False)
    main_r["fevd_market"].to_excel(xw, sheet_name="fevd_market", index=False)
    main_r["irf_market"].to_excel(xw, sheet_name="irf_market", index=False)
    main_r["fevd_sectors"].to_excel(xw, sheet_name="fevd_sectors", index=False)
    main_r["irf_sectors"].to_excel(xw, sheet_name="irf_sectors", index=False)

main_r["lead_lag"].to_excel(TAB / "11_lead_lag_anticipation.xlsx", index=False)

# ---- all-sector sensitivity heatmap ------------------------------------------
sens = main_r["fevd_sectors"].pivot(index="target", columns="source",
                                    values="FEVD_h22_pct")
gm = main_r["granger_sectors"].copy()
gm["used"] = gm["channel"].map(
    dict(zip(main_r["name_map"]["original"], main_r["name_map"]["used_in_VAR"])))
gm = gm.pivot(index="sector", columns="used", values="fwd_min_p")
sens = sens[[c for c in gm.columns if c in sens.columns]]
fig, ax = plt.subplots(figsize=(7.5, 10))
im = ax.imshow(sens.values, cmap="YlOrRd", aspect="auto")
ax.set_xticks(range(len(sens.columns)))
ax.set_xticklabels([c.replace("dS_", "d").replace("S_", "") for c in sens.columns],
                   rotation=0, fontsize=8)
ax.set_yticks(range(len(sens.index)))
ax.set_yticklabels([c.replace("r_", "").replace("_", " ")[:28] for c in sens.index],
                   fontsize=7)
for i, sec in enumerate(sens.index):
    for j, ch in enumerate(sens.columns):
        star = "*" if gm.loc[sec, ch] < 0.05 else ""
        ax.text(j, i, f"{sens.iloc[i, j]:.2f}{star}", ha="center", va="center",
                fontsize=6.5)
plt.colorbar(im, shrink=0.7, label="FEVD share at 22 days (%)")
ax.set_title("All-sector sensitivity matrix (I(1) sentiment differenced)\n"
             "cell = FEVD %, * = Toda-Yamamoto significant at 5%", fontsize=10)
plt.tight_layout(); plt.savefig(FIG / "10_sensitivity_heatmap_all.png", dpi=150)
plt.close()
sens_out = sens.copy(); sens_out.columns = [c + "_FEVD22pct" for c in sens_out.columns]
sens_out.to_excel(TAB / "10b_sensitivity_matrix_all.xlsx")

# ---- robustness ---------------------------------------------------------------
def sig_count(gdf, col="fwd_min_p"):
    return int((gdf[col] < 0.05).sum())

comp = pd.DataFrame([
    {"metric": "market: S_All -> r_Market TY min p",
     "directional": main_r["granger_market"].set_index("pair").loc["S_All->r_Market", "min_p"],
     "raw_finbert": raw_r["granger_market"].set_index("pair").loc["S_All_raw->r_Market", "min_p"]},
    {"metric": "sector-channel pairs TY-significant (fwd, 5%)",
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
    raw_r["integration"].to_excel(xw, sheet_name="integration_orders_raw", index=False)
    o5 = main_r["granger_sectors"][main_r["granger_sectors"]["sector"].isin(ORIGINAL_FIVE)]
    o5.to_excel(xw, sheet_name="original_five_subset", index=False)

print("\n=== headline numbers (v2.1, TY + differenced I(1) sentiment) ===")
print(comp.to_string(index=False))
print("\nMarket TY causality (directional):")
print(main_r["granger_market"][["pair", "dmax", "min_p", "sig_lags_5pct"]]
      .to_string(index=False))
print("\nEstimation suite complete.")
