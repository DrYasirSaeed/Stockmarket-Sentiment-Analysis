"""
Project: Information Transmission and Sectoral Asymmetry in the KSE-100
Module : Stationarity utilities — integration order and transformation
Author : Dr. Yasir Saeed (KUST)
Description:
  Shared by descriptives_suite.py and estimation_suite.py.

  integration_table(): ADF and KPSS in levels and first differences for
  every series, with an integration-order verdict:
      I(0) if levels ADF rejects the unit root at 5%
      I(1) if levels ADF fails but first-difference ADF rejects
      I(2)? otherwise (flagged; none expected)

  transform_for_var(): returns a working DataFrame in which every I(1)
  sentiment series is replaced by its first difference (renamed with a
  'd' prefix, e.g. dS_All = daily CHANGE in aggregate sentiment — the
  'sentiment news' interpretation). Return series are I(0) by
  construction and always enter in levels. Used for VAR estimation,
  IRF, FEVD, lead-lag regressions and cross-correlations.

  Causality tests do NOT use the transformed data: they use
  Toda-Yamamoto lag augmentation on LEVELS (see estimation_suite.py),
  which is valid regardless of integration order.
"""

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


def _adf_p(x):
    return adfuller(x.dropna(), autolag="AIC")[1]


def _kpss_p(x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return kpss(x.dropna(), regression="c", nlags="auto")[1]
        except Exception:
            return np.nan


def integration_table(df: pd.DataFrame, cols) -> pd.DataFrame:
    rows = []
    for c in cols:
        x = df[c].dropna()
        d = x.diff().dropna()
        adf_l, kp_l = _adf_p(x), _kpss_p(x)
        adf_d, kp_d = _adf_p(d), _kpss_p(d)
        if adf_l < 0.05:
            order = "I(0)"
        elif adf_d < 0.05:
            order = "I(1)"
        else:
            order = "I(2)?"
        rows.append({"series": c,
                     "ADF_p_level": round(adf_l, 4), "KPSS_p_level": round(kp_l, 3),
                     "ADF_p_diff": round(adf_d, 4), "KPSS_p_diff": round(kp_d, 3),
                     "integration_order": order})
    return pd.DataFrame(rows)


def transform_for_var(df: pd.DataFrame, s_cols, itab: pd.DataFrame):
    """Replace I(1) sentiment columns by first differences (dS_*).
    Returns (transformed_df, name_map {original: used_name}, orders {col: 0/1})."""
    orders = {r["series"]: (0 if r["integration_order"] == "I(0)" else 1)
              for _, r in itab.iterrows()}
    out = df.copy()
    name_map = {}
    for c in s_cols:
        if orders.get(c, 0) >= 1:
            new = "d" + c
            out[new] = out[c].diff()
            name_map[c] = new
        else:
            name_map[c] = c
    return out, name_map, orders
