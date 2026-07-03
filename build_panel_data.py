"""
Project: Information Transmission and Sectoral Asymmetry in the KSE-100
Module : Phase 3 Prep — Sector Returns + Panel Assembly
Author : Dr. Yasir Saeed
Affiliation: Kohat University of Science & Technology (KUST)
Description: Builds the Panel VAR input (panel_data.xlsx) from:
               1. Ticker-level PSX daily data (compiled_psx_historical CSV) —
                  the file contains no usable index series, so each sector's
                  return is the EQUAL-WEIGHTED mean of daily returns of its
                  major KSE-100 constituents (standard when value weights
                  are unavailable).
               2. Phase 2 daily sentiment series (daily_sentiment.xlsx).

             Daily stock return = CLOSE / LDCP - 1. LDCP is the exchange's
             own last-day closing price, already adjusted for corporate
             actions, which makes this cleaner than chaining CLOSE across
             days with self-computed adjustments. Returns with |r| > 25%
             are treated as data errors and dropped (PSX circuit breakers
             cap genuine single-day moves well below this).

OUTPUT
  panel_data.xlsx                     (project root — input to
                                       panel_var_estimator.py)
  Phase 3 Inputs/sector_returns.xlsx  (full-history sector returns +
                                       per-sector diagnostics)
USAGE
  python build_panel_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(r"D:\Stockmarket-Sentiment-Analysis")
PRICES_CSV  = Path(r"C:\Users\yasir\Desktop\psx\compiled_psx_historical_2017_2026.csv")
SENTIMENT   = PROJECT_DIR / "Phase 2 Outputs" / "daily_sentiment.xlsx"

OUT_DIR     = PROJECT_DIR / "Phase 3 Inputs"
RETURNS_OUT = OUT_DIR / "sector_returns.xlsx"
PANEL_OUT   = PROJECT_DIR / "panel_data.xlsx"       # where the estimator looks

MAX_ABS_RETURN = 0.25    # |daily return| above this = data error, dropped
MIN_CONSTITUENTS = 2     # min stocks trading for a sector-day to count

# Major KSE-100 constituents per target sector (README Table 1).
# Equal-weighted; a stock contributes only on days it actually traded,
# so listings (AIRLINK 2021) and delistings (FFBL merged into FFC 2024)
# enter and exit the average naturally.
SECTOR_CONSTITUENTS = {
    "Commercial_Banks": ["HBL", "UBL", "MCB", "NBP", "ABL", "BAHL",
                         "BAFL", "MEBL", "AKBL", "FABL", "BOP", "HMB"],
    "Exploration_Production": ["OGDC", "PPL", "POL", "MARI"],
    "Fertilizers": ["FFC", "EFERT", "FATIMA", "FFBL"],
    "Cement": ["LUCK", "DGKC", "MLCF", "FCCL", "KOHC",
               "CHCC", "ACPL", "PIOC", "BWCL"],
    "Technology": ["SYS", "TRG", "NETSOL", "AVN", "AIRLINK"],
}


def build_sector_returns() -> pd.DataFrame:
    print(f"\n--- Loading PSX prices: {PRICES_CSV.name} ---")
    all_tickers = sorted({t for ts in SECTOR_CONSTITUENTS.values() for t in ts})
    px = pd.read_csv(PRICES_CSV, usecols=["DATE", "SYMBOL", "LDCP", "CLOSE"])
    px = px[px["SYMBOL"].isin(all_tickers)].copy()
    px["DATE"] = pd.to_datetime(px["DATE"])
    print(f"  {len(px):,} ticker-day rows for {px['SYMBOL'].nunique()} constituents "
          f"({px['DATE'].min().date()} -> {px['DATE'].max().date()})")

    # Daily return on the exchange's own adjusted previous close
    px = px[(px["CLOSE"] > 0) & (px["LDCP"] > 0)]
    px["ret"] = px["CLOSE"] / px["LDCP"] - 1.0
    n_err = (px["ret"].abs() > MAX_ABS_RETURN).sum()
    px = px[px["ret"].abs() <= MAX_ABS_RETURN]
    print(f"  Dropped {n_err} returns with |r| > {MAX_ABS_RETURN:.0%} (data errors)")

    frames, diags = [], []
    for sector, tickers in SECTOR_CONSTITUENTS.items():
        sub = px[px["SYMBOL"].isin(tickers)]
        daily = (sub.groupby("DATE")
                    .agg(ret=("ret", "mean"), n=("ret", "size"))
                    .reset_index())
        daily = daily[daily["n"] >= MIN_CONSTITUENTS]
        frames.append(daily[["DATE", "ret"]].rename(
            columns={"ret": f"r_{sector}"}))
        diags.append({
            "sector": sector,
            "constituents": len(tickers),
            "found_in_data": sub["SYMBOL"].nunique(),
            "trading_days": len(daily),
            "mean_daily_ret_bps": round(daily["ret"].mean() * 1e4, 2),
            "std_daily_ret_pct": round(daily["ret"].std() * 100, 3),
            "avg_stocks_per_day": round(daily["n"].mean(), 1),
        })

    from functools import reduce
    returns = reduce(lambda a, b: pd.merge(a, b, on="DATE", how="outer"), frames)
    returns = returns.sort_values("DATE").reset_index(drop=True)

    diag_df = pd.DataFrame(diags)
    print("\n--- Sector return diagnostics ---")
    print(diag_df.to_string(index=False))

    OUT_DIR.mkdir(exist_ok=True)
    with pd.ExcelWriter(RETURNS_OUT) as xw:
        returns.to_excel(xw, sheet_name="daily_returns", index=False)
        diag_df.to_excel(xw, sheet_name="diagnostics", index=False)
    print(f"\n  Saved -> {RETURNS_OUT}")
    return returns


def build_panel(returns: pd.DataFrame) -> pd.DataFrame:
    print(f"\n--- Merging with sentiment: {SENTIMENT.name} ---")
    sent = pd.read_excel(SENTIMENT)
    sent["date"] = pd.to_datetime(sent["date_only"])
    sent = sent.drop(columns=["date_only"])

    panel = returns.rename(columns={"DATE": "date"}).merge(
        sent, on="date", how="inner")

    # Sentiment series were forward-filled at build time; any residual gaps
    # (sector traded, no article that day in some channel) are forward-filled
    # here, then rows with missing returns are dropped.
    s_cols = [c for c in panel.columns if c.startswith("S_")]
    r_cols = [c for c in panel.columns if c.startswith("r_")]
    panel[s_cols] = panel[s_cols].ffill()
    panel = panel.dropna(subset=r_cols + s_cols)
    panel = panel.sort_values("date").reset_index(drop=True)

    print(f"  Panel: {len(panel):,} trading days "
          f"({panel['date'].min().date()} -> {panel['date'].max().date()})")
    print(f"  Columns: {list(panel.columns)}")

    panel.to_excel(PANEL_OUT, index=False)
    print(f"  Saved -> {PANEL_OUT}")
    return panel


if __name__ == "__main__":
    returns = build_sector_returns()
    panel = build_panel(returns)
    print("\nNext step:  python panel_var_estimator.py")
