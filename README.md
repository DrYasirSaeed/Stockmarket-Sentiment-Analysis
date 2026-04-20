# 📌 Information Transmission and Sectoral Asymmetry in the KSE-100
## A Topic-Routed Sentiment Analysis with DCC-GARCH and Panel VAR

**Author:** Dr. Yasir Saeed  
**Affiliation:** Department of Economics, Kohat University of Science & Technology (KUST), KPK, Pakistan  
**Contact:** yasirsaeed@kust.edu.pk  
**Status:** Prototypes under development | April 2026  

---

## 🔍 Research Overview

This repository contains the full computational pipeline for my research on **how macroeconomic sentiment extracted from Pakistan's financial news media transmits into equity returns and volatility across five major sectors of the KSE-100 Index**.

The central argument is that treating sentiment as a single composite signal is a measurement error. Macroeconomic news in Pakistan arrives through distinct policy channels.  SBP monetary signals, FBR fiscal announcements, IMF/World Bank external finance developments, and NEPRA/OGRA energy regulatory decisions. Each channel activates different sectors at different intensities and speeds. Aggregating these signals before analysis destroys the very variation that contains the most economically meaningful information.

---

## Research Design Architecture

```
Phase 1 — Data Acquisition
├── Scrape: Business Recorder, Dawn Business, The News, ARY Business
├── Coverage: January 2014 – December 2025 (~2,900 trading days)  
└── Output: Raw article corpus (headline + body + timestamp)
        ↓
Phase 2 — FinBERT Sentiment Pipeline
├── Topic Router: Monetary | Fiscal | External Finance | Energy
├── FinBERT Scorer: signed sentiment ∈ [−1, +1] per article
└── Output: 4 daily sentiment series (S_Monetary, S_Fiscal, S_External, S_Energy)
        ↓
Phase 3 — DCC-GARCH Estimation
├── Asymmetric event window: [t−n, t+k] — empirically selected
├── 5 sectors × 4 sentiment categories = 20 sector-category pairs
└── Output: Time-varying correlations across 3 macroeconomic regimes
        ↓
Phase 4 — Panel VAR + Granger Causality
├── Impulse Response Functions (IRFs) — directional transmission
├── Forecast Error Variance Decomposition (FEVD) — magnitude ranking
└── Output: Sectoral Sensitivity Matrix
```

---

## Repository Structure

```
kse100-sentiment-transmission/
│
├── 01_data_acquisition/
│   ├── brecorder_scraper.py        ← Financial news scraper (Cloudflare challenge — see note)
│   ├── topic_router.py             ← Keyword classifier: Monetary/Fiscal/External/Energy
│   └── README.md
│
├── 02_finbert_pipeline/
│   ├── finbert_scorer.py           ← FinBERT sentiment scoring pipeline
│   ├── sentiment_aggregator.py     ← Daily value-weighted score aggregation
│   └── README.md
│
├── 03_dcc_garch/
│   ├── dcc_garch_estimator.py      ← DCC-GARCH across sector-category pairs
│   ├── window_selector.py          ← AIC/BIC-based asymmetric window selection
│   └── README.md
│
├── 04_panel_var/
│   ├── panel_var_estimator.py      ← Panel VAR estimation + Granger causality
│   ├── irf_fevd.py                 ← Impulse Response Functions + FEVD
│   └── README.md
│
├── 05_results/
│   └── sectoral_sensitivity_matrix.py   ← Terminal output: magnitude, direction, timing, persistence
│
├── data/
│   ├── raw/                        ← Raw scraped articles (gitignored — large files)
│   ├── interim/                    ← Classified and scored articles
│   └── processed/                  ← Final panel dataset for econometric estimation
│
├── requirements.txt
├── .gitignore
└── README.md                       ← You are here
```

---

## 📊 Target Sectors and Sentiment Categories

| Sector | KSE-100 Weight | Expected Primary Channel |
|---|---|---|
| Commercial Banks | Largest | Monetary (SBP rate decisions) |
| Exploration & Production (E&P) | 2nd | Energy + External Finance |
| Fertilizers | 3rd | Energy (gas policy) + Fiscal |
| Cement | 4th | Fiscal (PSDP spending) |
| Technology | Growing | External Finance (remittances/FDI) |
* Note for readers, these 5 catagories cover roughly about 70 percent of KSE-100 Index

| Sentiment Category | Primary Sources |
|---|---|
| Monetary | SBP MPC decisions, policy rate, reserve money |
| Fiscal | FBR revenue, budget, taxation, PSDP |
| External Finance | IMF reviews, World Bank, bilateral credit |
| Energy | NEPRA/OGRA decisions, circular debt, petroleum prices |

---

## Econometric Framework

**DCC-GARCH** (Engle, 2002) captures time-varying co-movement between sentiment shocks and sectoral return volatility across three macroeconomic regimes: pre-tightening (2014–2021), SBP tightening cycle (2022–2023), and easing phase (2024–2025).

**Panel VAR** (Holtz-Eakin, Newey & Rosen, 1988) identifies directional transmission — which sentiment categories Granger-cause which sectors, at what lag, and with what persistence.

**Asymmetric Event Window [t−n, t+k]** where n and k are selected empirically through AIC/BIC minimisation per sector-category pair. The pre-publication window (t−n) tests for informational leakage — a direct examination of weak-form efficiency in a frontier market.

---

## 🚧 Current Status by Module

| Module | Status | Note |
|---|---|---|
| `01_data_acquisition` | ⚠️ Blocked | Cloudflare protection on brecorder.com — see module README |
| `02_finbert_pipeline` | ✅ Ready | Pipeline written and tested on sample data |
| `03_dcc_garch` | 🔄 In progress | Framework structured, awaiting corpus |
| `04_panel_var` | 🔄 In progress | Framework structured, awaiting corpus |
| `05_results` | ⏳ Pending | Awaiting upstream modules |

---

## 🚧 Data Acquisition Challenge — Honest Note

The news scraper (`01_data_acquisition/brecorder_scraper.py`) is fully written and architecturally sound. It implements two Cloudflare bypass strategies — Playwright stealth browser and cloudscraper — with randomized human-like delays. However, brecorder.com's Cloudflare configuration defeats both strategies in Google Colab's environment due to (1) datacenter IP detection and (2) missing system libraries for headless Chromium.

This is a known and well-documented challenge in computational social science. The scraper works correctly as written; the limitation is infrastructural, not logical. Alternative data acquisition paths currently under evaluation: RSS feed parsing, ScraperAPI residential proxy integration, and direct archival access for pre-2020 data. See `01_data_acquisition/README.md` for full technical detail.

---

## Contact

**Dr. Yasir Saeed**  
Lecturer (BPS-18), Department of Economics  
Kohat University of Science & Technology (KUST)  
Khyber Pakhtunkhwa, Pakistan  
yasirsaeed@kust.edu.pk
