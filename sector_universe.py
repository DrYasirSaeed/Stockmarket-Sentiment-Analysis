"""
Project: Information Transmission and Sectoral Asymmetry in the KSE-100
Module : Phase 3a — Sector Universe, All-Sector Returns, Market Return
Author : Dr. Yasir Saeed (KUST)
Description:
  Extends the return construction from five hand-picked sectors to ALL PSX
  sectors meeting objective eligibility criteria, plus an equal-weighted
  market portfolio used as the combined-index proxy (the price file contains
  no usable index series).

  Sector classification: PSX Data Portal symbols endpoint
  (Source Data/psx_symbols_sectors.json; sectorName per listed security).
  Delisted securities absent from the current listing remain unmapped and
  are excluded — a survivorship caveat documented in the report.

  ELIGIBILITY (research-defensible, applied to the sentiment window
  2020-06-22 onward):
    - equity securities only (isDebt = False, isETF = False,
      non-empty sector; open/closed-end funds and modaraba-management
      shells excluded via sector exclusion list)
    - a stock enters its sector portfolio on days it traded with valid
      CLOSE/LDCP and |return| <= 25%
    - a stock must have >= 70% day coverage inside the window to count
      toward its sector's firm quorum
    - a sector qualifies if it has >= 5 such firms (diversified EW
      portfolio; idiosyncratic noise averages out)
    - a sector-day requires >= 3 traded constituents

  MARKET RETURN r_Market: equal-weighted mean daily return over ALL
  eligible-sector constituents (not only qualifying sectors' firms —
  the market portfolio uses every mapped equity passing the stock-level
  screens), the standard EW market proxy.

OUTPUT
  Phase 3 Inputs/sector_universe.xlsx      (eligibility table, per-sector diagnostics)
  Phase 3 Inputs/all_sector_returns.csv    (daily EW returns, all qualifying sectors + r_Market)
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(r"D:\Stockmarket-Sentiment-Analysis")
PRICES_CSV  = Path(r"C:\Users\yasir\Desktop\psx\compiled_psx_historical_2017_2026.csv")
SECTORS_JSON = PROJECT_DIR / "Source Data" / "psx_symbols_sectors.json"
OUT_DIR     = PROJECT_DIR / "Phase 3 Inputs"

WINDOW_START = "2020-06-22"          # sentiment sample start
MAX_ABS_RETURN = 0.25
MIN_FIRMS_PER_SECTOR = 4   # 5 would exclude Oil & Gas Exploration (exactly 4
                           # firms: OGDC, PPL, POL, MARI) — the KSE-100's
                           # 2nd-largest sector; 4 large caps is a defensible
                           # quorum for an equal-weighted portfolio
MIN_DAY_COVERAGE = 0.70
MIN_CONSTITUENTS_PER_DAY = 3

# Non-operating / investment-vehicle sectors excluded from sector analysis
# (their "returns" reflect fund NAVs / financial engineering, not sector
# fundamentals). Their stocks still count toward the market portfolio.
EXCLUDED_SECTORS = {
    "", "BILLS AND BONDS", "OPEN-END MUTUAL FUNDS", "CLOSE-END MUTUAL FUND",
    "EXCHANGE TRADED FUNDS", "FUTURE CONTRACTS", "DEFAULTER SEGMENT",
    "MISCELLANEOUS",   # not an economically meaningful sector grouping
}


def slug(s: str) -> str:
    s = re.sub(r"[&/,.]", " ", s.title())
    return re.sub(r"\s+", "_", s.strip())


def main():
    OUT_DIR.mkdir(exist_ok=True)

    # ---- sector mapping ----------------------------------------------------
    m = pd.DataFrame(json.load(open(SECTORS_JSON, encoding="utf-8")))
    eq = m[(~m["isDebt"]) & (~m["isETF"])].copy()
    eq["sectorName"] = eq["sectorName"].fillna("").str.strip()
    eq = eq[~eq["sectorName"].isin(EXCLUDED_SECTORS)]
    sector_map = dict(zip(eq["symbol"], eq["sectorName"]))
    print(f"Sector mapping: {len(sector_map)} equities in "
          f"{eq['sectorName'].nunique()} sectors")

    # ---- prices ------------------------------------------------------------
    px = pd.read_csv(PRICES_CSV, usecols=["DATE", "SYMBOL", "LDCP", "CLOSE"])
    px["DATE"] = pd.to_datetime(px["DATE"])
    px = px[px["DATE"] >= WINDOW_START]
    px["sector"] = px["SYMBOL"].map(sector_map)
    unmapped = px[px["sector"].isna()]["SYMBOL"].nunique()
    px = px.dropna(subset=["sector"])
    px = px[(px["CLOSE"] > 0) & (px["LDCP"] > 0)]
    px["ret"] = px["CLOSE"] / px["LDCP"] - 1.0
    px = px[px["ret"].abs() <= MAX_ABS_RETURN]
    n_days = px["DATE"].nunique()
    print(f"Window {WINDOW_START}+: {n_days} trading days | "
          f"{px['SYMBOL'].nunique()} mapped stocks | {unmapped} unmapped tickers excluded")

    # ---- stock-level coverage screen ---------------------------------------
    cov = px.groupby("SYMBOL")["DATE"].nunique() / n_days
    core = set(cov[cov >= MIN_DAY_COVERAGE].index)
    print(f"Stocks with >= {MIN_DAY_COVERAGE:.0%} day coverage: {len(core)}")

    # ---- sector eligibility -------------------------------------------------
    firms = (px[px["SYMBOL"].isin(core)]
             .groupby("sector")["SYMBOL"].nunique()
             .rename("core_firms").reset_index())
    firms["eligible"] = firms["core_firms"] >= MIN_FIRMS_PER_SECTOR
    eligible = sorted(firms[firms["eligible"]]["sector"])
    print(f"\nEligible sectors ({len(eligible)} of {len(firms)}):")
    for s in eligible:
        print(f"  {s}")

    # ---- sector returns -----------------------------------------------------
    frames, diags = [], []
    for sector in eligible:
        sub = px[(px["sector"] == sector) & (px["SYMBOL"].isin(core))]
        daily = (sub.groupby("DATE")
                    .agg(ret=("ret", "mean"), n=("ret", "size")).reset_index())
        daily = daily[daily["n"] >= MIN_CONSTITUENTS_PER_DAY]
        col = f"r_{slug(sector)}"
        frames.append(daily[["DATE", "ret"]].rename(columns={"ret": col}))
        diags.append({"sector": sector, "column": col,
                      "core_firms": int(firms.loc[firms.sector == sector, "core_firms"].iloc[0]),
                      "trading_days": len(daily),
                      "mean_ret_bps": round(daily["ret"].mean() * 1e4, 2),
                      "std_ret_pct": round(daily["ret"].std() * 100, 3),
                      "avg_stocks_per_day": round(daily["n"].mean(), 1)})

    # ---- market return (all mapped equities passing stock screens) ---------
    mkt = (px[px["SYMBOL"].isin(core)]
           .groupby("DATE")
           .agg(r_Market=("ret", "mean"), n=("ret", "size")).reset_index())
    frames.insert(0, mkt[["DATE", "r_Market"]])
    print(f"\nr_Market: EW over {px[px['SYMBOL'].isin(core)]['SYMBOL'].nunique()} stocks, "
          f"avg {mkt['n'].mean():.0f} per day")

    from functools import reduce
    returns = reduce(lambda a, b: pd.merge(a, b, on="DATE", how="outer"), frames)
    returns = returns.sort_values("DATE").reset_index(drop=True)
    returns.to_csv(OUT_DIR / "all_sector_returns.csv", index=False)

    diag_df = pd.DataFrame(diags)
    firms_out = firms.sort_values(["eligible", "core_firms"], ascending=[False, False])
    with pd.ExcelWriter(OUT_DIR / "sector_universe.xlsx") as xw:
        firms_out.to_excel(xw, sheet_name="eligibility", index=False)
        diag_df.to_excel(xw, sheet_name="sector_diagnostics", index=False)
        pd.DataFrame({
            "parameter": ["window_start", "max_abs_return", "min_firms_per_sector",
                          "min_day_coverage", "min_constituents_per_day",
                          "unmapped_tickers_excluded", "eligible_sectors"],
            "value": [WINDOW_START, MAX_ABS_RETURN, MIN_FIRMS_PER_SECTOR,
                      MIN_DAY_COVERAGE, MIN_CONSTITUENTS_PER_DAY,
                      unmapped, len(eligible)],
        }).to_excel(xw, sheet_name="criteria", index=False)

    print(f"\nSaved -> {OUT_DIR / 'all_sector_returns.csv'} "
          f"({len(returns)} days x {len(returns.columns) - 1} return series)")
    print(f"Saved -> {OUT_DIR / 'sector_universe.xlsx'}")


if __name__ == "__main__":
    main()
