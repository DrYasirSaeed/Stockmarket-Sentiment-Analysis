# 📌 Information Transmission and Sectoral Asymmetry in the KSE-100
## A Topic-Routed Sentiment Analysis with DCC-GARCH and Panel VAR

**Author:** Dr. Yasir Saeed  
**Affiliation:** Department of Economics, Kohat University of Science & Technology (KUST), KPK, Pakistan  
**Contact:** yasirsaeed@kust.edu.pk  
**Status:** Active development | May 2026  

---

## 🔍 Research Overview

This repository contains the full computational pipeline for research on **how macroeconomic sentiment extracted from Pakistan's financial news media transmits into equity returns and volatility across five major sectors of the KSE-100 Index**.

The central argument is that treating sentiment as a single composite signal is a measurement error. Macroeconomic news in Pakistan arrives through distinct policy channels — SBP monetary signals, FBR fiscal announcements, IMF/World Bank external finance developments, and NEPRA/OGRA energy regulatory decisions. Each channel activates different sectors at different intensities and speeds. Aggregating these signals before analysis destroys the very variation that contains the most economically meaningful information.

---

## Pipeline Architecture

```
Phase 0 — Data Acquisition (news_extractor.py)
├── Source  : Business Recorder (brecorder.com)
├── Method  : Sequential article ID scan + curl_cffi Chrome TLS impersonation
├── Coverage: January 2023 – December 2024
└── Output  : brecorder_articles.csv  (article_id, date, headline, body_text, author)
        ↓
Phase 1 — Topic Router (topic_router.py)
├── Keyword classifier: Monetary | Fiscal | External Finance | Energy
├── Headline weighted 3× body text (editorial emphasis signal)
├── Validation: precision/recall against 300-article labelled sample
└── Output  : Classified corpus with per-category confidence scores
        ↓
Phase 2 — FinBERT Sentiment Pipeline (finbert_scorer.py)
├── ProsusAI/finbert — finance-domain BERT model
├── Signed sentiment score ∈ [−1, +1] per article
└── Output  : 4 daily sentiment series (S_Monetary, S_Fiscal, S_External, S_Energy)
        ↓
Phase 3 — DCC-GARCH Estimation
├── Asymmetric event window: [t−n, t+k] — empirically selected via AIC/BIC
├── 5 sectors × 4 sentiment categories = 20 sector-category pairs
└── Output  : Time-varying correlations across 3 macroeconomic regimes
        ↓
Phase 4 — Panel VAR + Granger Causality (panel_var_estimator.py)
├── Impulse Response Functions (IRFs) — directional transmission
├── Forecast Error Variance Decomposition (FEVD) — magnitude ranking
└── Output  : Sectoral Sensitivity Matrix (sectoral_sensitivity_matrix.py)
```

---

## Repository Structure

```
Stockmarket-Sentiment-Analysis/
│
├── news_extractor.py            ← Phase 0: BRecorder article scraper
├── topic_router.py              ← Phase 1: Keyword topic classifier
├── finbert_scorer.py            ← Phase 2: FinBERT sentiment scoring
├── panel_var_estimator.py       ← Phase 4: Panel VAR + Granger causality
├── sectoral_sensitivity_matrix.py  ← Phase 5: Results output
│
├── requirements.txt             ← All Python dependencies
├── LICENSE
└── README.md
```

---

## 📊 Target Sectors and Sentiment Categories

| Sector | KSE-100 Weight | Expected Primary Channel |
|---|---|---|
| Commercial Banks | Largest | Monetary (SBP rate decisions) |
| Exploration & Production (E&P) | 2nd | Energy + External Finance |
| Fertilizers | 3rd | Energy (gas policy) + Fiscal |
| Cement | 4th | Fiscal (PSDP spending) |
| Technology | Growing | External Finance (remittances / FDI) |

> These 5 sectors cover approximately 70% of the KSE-100 Index by weight.

| Sentiment Category | Primary Sources |
|---|---|
| Monetary | SBP MPC decisions, policy rate, reserve money, CPI |
| Fiscal | FBR revenue, federal budget, taxation, PSDP |
| External Finance | IMF programme reviews, World Bank, bilateral credit |
| Energy | NEPRA/OGRA decisions, circular debt, petroleum prices |

---

## Econometric Framework

**DCC-GARCH** (Engle, 2002) captures time-varying co-movement between sentiment shocks and sectoral return volatility across three macroeconomic regimes: pre-tightening (2014–2021), SBP tightening cycle (2022–2023), and easing phase (2024–2025).

**Panel VAR** (Holtz-Eakin, Newey & Rosen, 1988) identifies directional transmission — which sentiment categories Granger-cause which sectors, at what lag, and with what persistence.

**Asymmetric Event Window [t−n, t+k]** where n and k are selected empirically through AIC/BIC minimisation per sector-category pair. The pre-publication window (t−n) tests for informational leakage — a direct examination of weak-form efficiency in a frontier market.

---

## 🚧 Current Status

| Module | File | Status | Note |
|---|---|---|---|
| Phase 0 — Data Acquisition | `news_extractor.py` | ✅ Working | ID-scan pipeline; curl_cffi TLS bypass confirmed working locally |
| Phase 1 — Topic Router | `topic_router.py` | ✅ Ready | Keyword classifier validated; headline 3× weighting fixed |
| Phase 2 — FinBERT Scoring | `finbert_scorer.py` | ✅ Ready | Tested on sample data |
| Phase 3 — DCC-GARCH | — | 🔄 In progress | Framework structured, awaiting full corpus |
| Phase 4 — Panel VAR | `panel_var_estimator.py` | 🔄 In progress | Framework structured, awaiting corpus |
| Phase 5 — Results | `sectoral_sensitivity_matrix.py` | ⏳ Pending | Awaiting upstream modules |

---

## ⚙️ Setup and Usage

### Requirements

```bash
pip install -r requirements.txt
```

### Phase 0 — Extract news articles

```python
# Edit date range and ID window in news_extractor.py, then:
python news_extractor.py
# Output: brecorder_articles.csv
```

> **Important:** Run from a home or university network, not from Google Colab or any cloud
> server. Cloudflare blocks datacenter IPs regardless of scraping library used.
> `curl_cffi` impersonates Chrome's TLS fingerprint and works reliably from residential IPs.

### Phase 1 — Classify articles by topic

```python
from topic_router import classify_corpus
import pandas as pd

df = pd.read_csv("brecorder_articles.csv")
classified = classify_corpus(df)
classified.to_csv("classified_articles.csv", index=False)
```

### Phase 2 — Score sentiment

```python
from finbert_scorer import score_corpus
import pandas as pd

df = pd.read_csv("classified_articles.csv")
scored = score_corpus(df)
scored.to_csv("scored_articles.csv", index=False)
```

---

## Notes on Data Acquisition

Early versions of the scraper used `cloudscraper` and `playwright-stealth` to bypass
Cloudflare, but both failed when run from Google Colab because Cloudflare's Bot Management
blocklists Google Cloud datacenter IP ranges at the network level — no library-level bypass
can overcome an IP-level block.

The current `news_extractor.py` uses `curl_cffi` which impersonates Chrome's exact TLS
handshake (JA3 fingerprint). When run from a residential or university IP this passes
Cloudflare's checks reliably. The scraper scans sequential BRecorder article IDs
(`brecorder.com/news/<id>`), applies a date filter, and saves results with periodic
checkpoints so a crash does not lose progress.

---

## Contact

**Dr. Yasir Saeed**  
Lecturer (BPS-18), Department of Economics  
Kohat University of Science & Technology (KUST)  
Khyber Pakhtunkhwa, Pakistan  
yasirsaeed@kust.edu.pk
