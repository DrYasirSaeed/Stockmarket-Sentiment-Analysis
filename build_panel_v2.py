"""
Project: Information Transmission and Sectoral Asymmetry in the KSE-100
Module : Phase 3b v2 — Expanded Panel Assembly
Author : Dr. Yasir Saeed (KUST)
Description: Merges the all-sector return set (25 eligible sectors + the
             equal-weighted market portfolio) with the v2 sentiment series
             (S_All + four channel series, multi-label, weekend-shifted),
             in both directional and raw-FinBERT variants.
OUTPUT
  panel_data_all.xlsx   sheet 'directional' (main) and 'raw_finbert' (robustness)
"""

from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(r"D:\Stockmarket-Sentiment-Analysis")
RETURNS_CSV = PROJECT_DIR / "Phase 3 Inputs" / "all_sector_returns.csv"
SENT_XLSX   = PROJECT_DIR / "Phase 2 Outputs" / "daily_sentiment_v2.xlsx"
OUT         = PROJECT_DIR / "panel_data_all.xlsx"

ret = pd.read_csv(RETURNS_CSV, parse_dates=["DATE"]).rename(columns={"DATE": "date"})

with pd.ExcelWriter(OUT) as xw:
    for sheet in ["directional", "raw_finbert"]:
        sent = pd.read_excel(SENT_XLSX, sheet_name=sheet, parse_dates=["date"])
        panel = ret.merge(sent, on="date", how="inner").sort_values("date")
        s_cols = [c for c in panel.columns if c.startswith("S_")]
        r_cols = [c for c in panel.columns if c.startswith("r_")]
        panel[s_cols] = panel[s_cols].ffill()
        panel = panel.dropna(subset=["r_Market"] + s_cols).reset_index(drop=True)
        panel.to_excel(xw, sheet_name=sheet, index=False)
        print(f"{sheet}: {len(panel):,} days x {len(r_cols)} returns + {len(s_cols)} sentiment "
              f"({panel['date'].min().date()} -> {panel['date'].max().date()})")

print(f"Saved -> {OUT}")
