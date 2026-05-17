# Information Transmission and Sectoral Asymmetry in the KSE-100
## A Topic-Routed Sentiment Analysis with Panel VAR

**Author:** Dr. Yasir Saeed  
**Affiliation:** Department of Economics, Kohat University of Science & Technology (KUST), KPK, Pakistan  
**Contact:** yasirsaeed@kust.edu.pk  
**Status:** Active development | May 2026  

---

## Research Overview

This repository contains the full computational pipeline for research on **how macroeconomic sentiment extracted from Pakistan's financial news media transmits into equity returns and volatility across five major sectors of the KSE-100 Index**.

The central argument is that treating sentiment as a single composite signal is a measurement error. Macroeconomic news in Pakistan arrives through distinct policy channels — SBP monetary signals, FBR fiscal announcements, IMF/World Bank external finance developments, and NEPRA/OGRA energy regulatory decisions. Each channel activates different sectors at different intensities and speeds. Aggregating these signals before analysis destroys the very variation that contains the most economically meaningful information.

---

## Pipeline Architecture

```
Phase 0 — Data Acquisition (news_extractor.py)
├── Source  : Business Recorder (brecorder.com)
├── Method  : Sequential article ID scan + curl_cffi Chrome TLS impersonation
├── Coverage: June 2020 to present (~40,000,000 onwards)
└── Output  : brecorder_YYYY_MM.csv  (article_id, date, headline, body_text, author)
        |
        v
Phase 1a — Corpus Exploration (bertopic_explore.py)
├── Model   : BERTopic with paraphrase-multilingual-MiniLM-L12-v2
├── Purpose : Unsupervised topic discovery to validate and refine
│             the keyword categories used in Phase 1b, and to
│             identify and filter off-topic articles before scoring
├── Outputs : topic assignments, visualisations, topics-over-time
└── Artefact: bertopic_model.pkl  (cached; refit with --refit flag)
        |
        v
Phase 1b — Topic Router (topic_router.py)
├── Keyword classifier: Monetary | Fiscal | External Finance | Energy
├── Headline weighted 3x body text (editorial emphasis signal)
├── BERTopic exploration used to validate category boundaries
├── Validation: precision/recall against 300-article labelled sample
└── Output  : classified_articles.csv  (+ topic_category, confidence scores)
        |
        v
Phase 2 — FinBERT Sentiment Scoring (finbert_scorer.py)
├── Model   : ProsusAI/finbert (Araci, 2019)
├── Signed sentiment score in [-1, +1] per article
├── Aggregated to value-weighted daily series
└── Output  : daily_sentiment.xlsx  (S_Monetary, S_Fiscal, S_External, S_Energy)
        |
        v
Phase 3 — Panel VAR + Granger Causality (panel_var_estimator.py)
├── Lag order selected by AIC
├── Granger causality: which sentiment categories lead which sectors
├── Impulse Response Functions (IRFs): directional transmission
├── Forecast Error Variance Decomposition (FEVD): magnitude ranking
└── Output  : granger_causality.xlsx | irf_plots.png | fevd_results.xlsx
        |
        v
Phase 4 — Sectoral Sensitivity Matrix (sectoral_sensitivity_matrix.py)
├── Synthesizes Phase 3 outputs into one structured table
├── 20 sector-category pairs (5 sectors x 4 sentiment categories)
└── Output  : sectoral_sensitivity_matrix.xlsx | heatmap.png
```

---

## Repository Structure

```
Stockmarket-Sentiment-Analysis/
│
├── news_extractor.py               <- Phase 0:  BRecorder article scraper
├── bertopic_explore.py             <- Phase 1a: BERTopic corpus exploration
├── topic_router.py                 <- Phase 1b: Keyword topic classifier
├── finbert_scorer.py               <- Phase 2:  FinBERT sentiment scoring
├── panel_var_estimator.py          <- Phase 3:  Panel VAR + Granger causality
├── sectoral_sensitivity_matrix.py  <- Phase 4:  Results synthesis
│
├── requirements.txt                <- All Python dependencies
├── LICENSE
└── README.md
```

> **Not tracked in git** (generated/data artefacts):  
> `Extracted Data/` · `BERTopic Outputs/` · `bertopic_model.pkl` · `embeddings_cache.npy`

---

## Target Sectors and Sentiment Categories

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

**Panel VAR** (Holtz-Eakin, Newey & Rosen, 1988) identifies directional transmission — which sentiment categories Granger-cause which sectors, at what lag, and with what persistence. Lag order is selected empirically via AIC minimisation.

**Impulse Response Functions (IRFs)** trace the dynamic response of each sector's return to a one-standard-deviation shock in each sentiment category over a 22-trading-day horizon.

**Forecast Error Variance Decomposition (FEVD)** quantifies the share of sectoral return variance attributable to each macroeconomic sentiment category at horizons of 1, 5, and 22 trading days — providing a magnitude ranking of policy channel importance across sectors.

**Asymmetric Event Window** analysis uses IRF peak timing to identify whether sentiment absorption is instantaneous (t=1–3, efficient) or delayed (t=5–22, illiquid), and whether pre-announcement leakage is detectable (IRF peak before t=0).

---

## Current Status

| Module | File | Status | Note |
|---|---|---|---|
| Phase 0 — Data Acquisition | `news_extractor.py` | Running | Parallel ID-scan; resumable; writing to monthly CSVs |
| Phase 1a — Corpus Exploration | `bertopic_explore.py` | Ready | BERTopic; embedding cache; all visualisations |
| Phase 1b — Topic Router | `topic_router.py` | Ready | Keyword classifier; 3x headline weighting |
| Phase 2 — FinBERT Scoring | `finbert_scorer.py` | Ready | Tested on sample data |
| Phase 3 — Panel VAR | `panel_var_estimator.py` | Ready | Awaiting completed corpus |
| Phase 4 — Sensitivity Matrix | `sectoral_sensitivity_matrix.py` | Ready | Awaiting Phase 3 outputs |

---

## Setup and Usage

### Install dependencies

```bash
pip install -r requirements.txt
```

### Phase 0 — Extract articles (run from home/university network)

```bash
python news_extractor.py              # resumes automatically from last position
python news_extractor.py --status     # check progress without scraping
python news_extractor.py --instance 2 # run second parallel instance
```

> Cloudflare blocks datacenter IPs (Colab, AWS, etc.) regardless of library used.  
> `curl_cffi` impersonates Chrome's TLS fingerprint and works from residential IPs.

### Phase 1a — Explore corpus with BERTopic

```bash
python bertopic_explore.py            # fit model and generate all outputs
python bertopic_explore.py --status   # check which output files exist
python bertopic_explore.py --refit    # force re-encode + refit from scratch
```

Embeddings are cached to `embeddings_cache.npy` after the first run — subsequent
runs skip the encoding step entirely. The fitted model is saved as `bertopic_model.pkl`.
All visualisations (interactive HTML + static PNG) are written to `BERTopic Outputs/`.

### Phase 1b — Classify articles

```python
from topic_router import classify_corpus
import pandas as pd

df = pd.read_csv("brecorder_articles.csv")
classified = classify_corpus(df)
classified.to_csv("classified_articles.csv", index=False)
```

### Phase 2 — Score sentiment

```bash
python finbert_scorer.py
# Input : classified_articles.csv
# Output: daily_sentiment.xlsx
```

### Phase 3 — Panel VAR

```bash
python panel_var_estimator.py
# Input : panel_data.xlsx  (KSE-100 returns merged with daily_sentiment.xlsx)
# Output: granger_causality.xlsx, irf_plots.png, fevd_results.xlsx
```

### Phase 4 — Sensitivity Matrix

```bash
python sectoral_sensitivity_matrix.py
# Input : fevd_results.xlsx, granger_causality.xlsx
# Output: sectoral_sensitivity_matrix.xlsx, sectoral_sensitivity_heatmap.png
```

---

## Contact

**Dr. Yasir Saeed**  
Lecturer (BPS-18), Department of Economics  
Kohat University of Science & Technology (KUST)  
Khyber Pakhtunkhwa, Pakistan  
yasirsaeed@kust.edu.pk
