"""
Project: Information Transmission and Sectoral Asymmetry in the KSE-100
Module : Phase 0 — News Gap Filler
Author : Dr. Yasir Saeed
Affiliation: Kohat University of Science & Technology (KUST)

Description:
  Scans the extracted CSV files from instance 1 and instance 2,
  identifies every article ID in the full range (40,000,000–40,422,619)
  that was not captured, then re-fetches only those missing IDs.

  Gaps arise from:
    - 403 blocks that exhausted all retries
    - Network errors that exhausted all retries
    - Gap-skip jumps (50 consecutive 404s → skip 200 IDs) that
      accidentally jumped over real articles

  Already-downloaded IDs are skipped instantly (no HTTP request).
  Recovered articles are saved to the 'filler' sub-folder in the
  same monthly-CSV format as the main extraction.

Usage:
    python news_filler.py           # analyse gaps then fill them
    python news_filler.py --status  # show gap report without fetching
"""

import re
import time
import json
import random
import logging
import argparse
import sys
import gc
from collections import deque
from datetime import datetime, date, timezone
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

try:
    from curl_cffi import requests as cffi_requests
except ImportError:
    print("ERROR: curl_cffi not installed.  Run:  pip install curl_cffi")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# =====================================================================
# Configuration
# =====================================================================

ID_SCAN_START = 40_000_000
ID_SCAN_END   = 40_422_619

BASE_OUTPUT_DIR = Path(r"D:\Stockmarket-Sentiment-Analysis\Extracted Data")
INSTANCE_DIRS   = {
    1: BASE_OUTPUT_DIR / "instance_1",
    2: BASE_OUTPUT_DIR / "instance_2",
}
FILLER_DIR    = BASE_OUTPUT_DIR / "filler"
PROGRESS_FILE = FILLER_DIR / "filler_progress.json"

# Delays — fixed, no adaptive logic
DELAY_HIT  = 2.3   # fixed delay after every successful article
DELAY_MISS_MIN  = 0.1
DELAY_MISS_MAX  = 0.3

MAX_RETRIES         = 2
RETRY_BACKOFF_403   = [2, 2]
RETRY_BACKOFF_OTHER = [2, 2]

BASE    = "https://www.brecorder.com"
ART_URL = BASE + "/news/{article_id}"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://www.google.com/",
}


# =====================================================================
# Load Downloaded IDs from Both Instances
# =====================================================================

def load_downloaded_ids() -> set:
    """
    Read all monthly CSV files from both instance folders and
    return a set of every article_id already successfully downloaded.
    """
    downloaded = set()
    for inst, folder in INSTANCE_DIRS.items():
        csv_files = list(folder.glob("brecorder_*.csv"))
        if not csv_files:
            log.warning(f"  Instance {inst}: no CSV files found in {folder}")
            continue
        for csv_path in sorted(csv_files):
            try:
                df = pd.read_csv(
                    csv_path, usecols=["article_id"],
                    dtype={"article_id": int}
                )
                downloaded.update(df["article_id"].tolist())
                log.info(f"  Loaded {len(df):>6,} IDs from {csv_path.name}")
            except Exception as e:
                log.warning(f"  Could not read {csv_path.name}: {e}")
    return downloaded


# =====================================================================
# Gap Analysis
# =====================================================================

def analyse_gaps(downloaded_ids: set) -> list:
    """
    Return a list of every ID in [ID_SCAN_START, ID_SCAN_END]
    not present in downloaded_ids.
    """
    return [i for i in range(ID_SCAN_START, ID_SCAN_END + 1)
            if i not in downloaded_ids]


def print_gap_report(downloaded_ids: set, missing_ids: list) -> None:
    total_range = ID_SCAN_END - ID_SCAN_START + 1
    coverage    = len(downloaded_ids) / total_range * 100 if total_range else 0

    # Cluster consecutive missing IDs into gap ranges
    clusters = []
    if missing_ids:
        start = end = missing_ids[0]
        for id_ in missing_ids[1:]:
            if id_ == end + 1:
                end = id_
            else:
                clusters.append((start, end, end - start + 1))
                start = end = id_
        clusters.append((start, end, end - start + 1))

    print(f"\n{'='*55}")
    print(f"  Gap Analysis Report")
    print(f"  Full ID range    : {ID_SCAN_START:,} → {ID_SCAN_END:,}  ({total_range:,} IDs)")
    print(f"  Downloaded       : {len(downloaded_ids):,} articles  ({coverage:.1f}% of range)")
    print(f"  IDs to check     : {len(missing_ids):,}")
    print(f"  Gap clusters     : {len(clusters):,}")
    if clusters:
        print(f"\n  10 largest gaps:")
        for s, e, sz in sorted(clusters, key=lambda x: -x[2])[:10]:
            print(f"    {s:,} → {e:,}  ({sz:,} IDs)")
    print(f"{'='*55}\n")


# =====================================================================
# Progress Tracking
# =====================================================================

def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            p = json.load(f)
        log.info(f"Resuming gap fill from ID {p['last_scanned_id'] + 1}")
        return p
    log.info(f"Starting fresh gap fill from ID {ID_SCAN_START}")
    return {
        "last_scanned_id": ID_SCAN_START - 1,
        "total_recovered":  0,
        "month_counts":     {},
        "started_at":       datetime.now(timezone.utc).isoformat(),
        "last_updated":     datetime.now(timezone.utc).isoformat(),
    }


def save_progress(p: dict) -> None:
    FILLER_DIR.mkdir(parents=True, exist_ok=True)
    p["last_updated"] = datetime.now(timezone.utc).isoformat()
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(p, f, indent=2)


# =====================================================================
# Monthly CSV Management
# =====================================================================

def instance_dir_for(article_id: int) -> Path:
    """Route the article to the correct instance folder by ID range."""
    if article_id <= 40_300_000:
        return INSTANCE_DIRS[1]
    return INSTANCE_DIRS[2]


def monthly_csv_path(article_id: int, date_str: str) -> Path:
    folder = instance_dir_for(article_id)
    try:
        d = date.fromisoformat(date_str[:10])
        return folder / f"brecorder_{d.year}_{d.month:02d}.csv"
    except (ValueError, TypeError):
        return folder / "brecorder_unknown.csv"


def append_to_monthly_csv(record: dict) -> None:
    path = monthly_csv_path(record["article_id"], record.get("date", ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    pd.DataFrame([record]).to_csv(
        path, mode="a", header=not file_exists,
        index=False, encoding="utf-8-sig"
    )


# =====================================================================
# Session Factory
# =====================================================================

def make_session() -> cffi_requests.Session:
    s = cffi_requests.Session(impersonate="chrome120")
    s.headers.update(HEADERS)
    log.info("Session ready: curl_cffi / chrome120 TLS impersonation")
    return s


# =====================================================================
# HTML Parsers
# =====================================================================

def _first_text(soup, selectors: list) -> str:
    for sel in selectors:
        el = soup.select_one(sel)
        if el and el.get_text(strip=True):
            return el.get_text(separator=" ", strip=True)
    return ""


def _extract_headline(soup) -> str:
    return _first_text(soup, [
        "h1.title", "h1.article-title", "h1.entry-title",
        "h1.post-title", "article h1", "h1",
    ])


def _extract_body(soup) -> str:
    container = (
        soup.select_one(".story-body")
        or soup.select_one(".article-body")
        or soup.select_one(".entry-content")
        or soup.select_one(".post-content")
        or soup.select_one(".td-post-content")
        or soup.select_one('[itemprop="articleBody"]')
        or soup.find("article")
    )
    if not container:
        return ""
    for noise in container(["script", "style", "figure", "aside",
                             "nav", "iframe", "noscript"]):
        noise.decompose()
    paras = [p.get_text(separator=" ", strip=True)
             for p in container.find_all("p")
             if len(p.get_text(strip=True)) > 30]
    return " ".join(paras)


def _extract_date(soup) -> str:
    meta = soup.find("meta", property="article:published_time")
    if meta and meta.get("content"):
        return meta["content"][:10]
    raw = _first_text(soup, [
        "time[datetime]", ".publish-date", ".story-date",
        ".post-date", ".entry-date", ".date",
    ])
    m = re.search(r"\d{4}-\d{2}-\d{2}", raw)
    return m.group() if m else raw[:30]


def _extract_author(soup) -> str:
    return _first_text(soup, [
        ".author-name", ".byline", ".reporter",
        '[rel="author"]', ".entry-author", '[itemprop="author"]',
    ])


def parse_page(html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    return {
        "headline":  _extract_headline(soup),
        "body_text": _extract_body(soup),
        "date":      _extract_date(soup),
        "author":    _extract_author(soup),
    }


# =====================================================================
# Single Article Fetch
# =====================================================================

def fetch_article(session, article_id: int) -> dict:
    url    = ART_URL.format(article_id=article_id)
    result = {
        "article_id":   article_id,
        "url":          url,
        "headline":     "",
        "body_text":    "",
        "date":         "",
        "author":       "",
        "http_status":  0,
        "retries_used": 0,
        "scraped_at":   datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    }

    for attempt in range(1, MAX_RETRIES + 1):
        if attempt > 1:
            result["retries_used"] += 1
        try:
            r = session.get(url, timeout=75, allow_redirects=True)
            result["http_status"] = r.status_code

            if r.status_code == 404:
                return result
            if r.status_code == 403:
                wait = RETRY_BACKOFF_403[attempt - 1]
                log.warning(f"  [{article_id}] 403 (attempt {attempt}/{MAX_RETRIES})"
                            f" — retrying in {wait}s")
                if attempt < MAX_RETRIES:
                    time.sleep(wait)
                    continue
                return result
            if r.status_code != 200:
                log.warning(f"  [{article_id}] HTTP {r.status_code}")
                return result
            result.update(parse_page(r.text))
            return result
        except Exception as exc:
            log.warning(f"  [{article_id}] Error (attempt {attempt}): {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_OTHER[attempt - 1])

    return result


# =====================================================================
# Main Gap Filler Runner
# =====================================================================

def run_filler(downloaded_ids: set) -> None:
    """
    Iterate the full ID range. Skip IDs already downloaded (no HTTP
    request). Fetch every other ID and save any articles found.
    Can be interrupted and resumed — progress is tracked in filler_progress.json.
    """
    FILLER_DIR.mkdir(parents=True, exist_ok=True)
    progress    = load_progress()
    resume_from = progress["last_scanned_id"] + 1

    if resume_from > ID_SCAN_END:
        log.info("Gap fill already complete — all IDs have been checked.")
        return

    session = make_session()

    ids_fetched   = 0   # IDs where an HTTP request was made
    recovered     = 0   # new articles found and saved
    misses        = 0   # 404s
    blocked       = 0   # 403s (after all retries)

    # Rolling 15-min throughput monitor
    RATE_WINDOW = 15 * 60
    saved_times = deque()

    remaining = ID_SCAN_END - resume_from + 1
    log.info(f"Gap fill: {remaining:,} IDs to scan from {resume_from:,}")
    log.info(f"  Hit delay          : {DELAY_HIT}s (fixed)")

    article_id = resume_from

    try:
        while article_id <= ID_SCAN_END:

            # ── Fetch every ID — no skipping ──
            ids_fetched += 1
            result = fetch_article(session, article_id)
            status = result["http_status"]

            if status == 200:
                recovered += 1
                progress["total_recovered"] += 1
                month_key = result["date"][:7] if result["date"] else "unknown"
                progress["month_counts"][month_key] = (
                    progress["month_counts"].get(month_key, 0) + 1
                )
                append_to_monthly_csv(result)
                saved_times.append(time.time())

                overall_pct = (article_id - ID_SCAN_START + 1) / \
                              (ID_SCAN_END - ID_SCAN_START + 1) * 100
                log.info(f"  {article_id}  RECOVERED  {result['date']}  "
                         f"\"{result['headline'][:55]}\"  "
                         f"[overall {overall_pct:.1f}%]")
                time.sleep(DELAY_HIT)

            elif status == 404:
                misses += 1
                log.debug(f"  {article_id}  404")
                time.sleep(random.uniform(DELAY_MISS_MIN, DELAY_MISS_MAX))

            elif status == 403:
                blocked += 1
                log.warning(f"  {article_id}  BLOCKED (403)")
                time.sleep(random.uniform(DELAY_MISS_MIN, DELAY_MISS_MAX))

            else:
                log.warning(f"  {article_id}  HTTP {status}")
                time.sleep(random.uniform(DELAY_MISS_MIN, DELAY_MISS_MAX))

            progress["last_scanned_id"] = article_id

            # Save progress + throughput log every 100 fetched IDs
            if ids_fetched % 100 == 0:
                save_progress(progress)
                cutoff = time.time() - RATE_WINDOW
                while saved_times and saved_times[0] < cutoff:
                    saved_times.popleft()
                rate = len(saved_times) / 15.0
                overall_pct = (article_id - ID_SCAN_START + 1) / \
                              (ID_SCAN_END - ID_SCAN_START + 1) * 100
                log.info(f"  --- Progress saved  "
                         f"[overall {overall_pct:.1f}% | fetched {ids_fetched:,} | "
                         f"recovered {recovered} | {rate:.1f} art/min (15-min avg)] ---")

            if ids_fetched % 500 == 0:
                session.cookies.clear()
                gc.collect()

            article_id += 1

    except KeyboardInterrupt:
        progress["last_scanned_id"] = article_id
        save_progress(progress)
        cutoff = time.time() - RATE_WINDOW
        while saved_times and saved_times[0] < cutoff:
            saved_times.popleft()
        rate = len(saved_times) / 15.0
        print(f"\n\n  Ctrl+C detected — saving progress and stopping cleanly...")
        print(f"\n{'='*55}")
        print(f"  Interrupted at ID       : {article_id:,}")
        print(f"  IDs fetched             : {ids_fetched:,}")
        print(f"  Articles recovered      : {recovered:,}")
        print(f"  404 (no article)        : {misses:,}")
        print(f"  Blocked (403)           : {blocked:,}")
        print(f"  Throughput (15-min)     : {rate:.1f} articles/min")
        print(f"\n  To resume, run:")
        print(f"\n      python news_filler.py")
        print(f"\n{'='*55}\n")
        return

    # ── Final save and summary ──
    save_progress(progress)
    cutoff = time.time() - RATE_WINDOW
    while saved_times and saved_times[0] < cutoff:
        saved_times.popleft()
    rate = len(saved_times) / 15.0

    print(f"\n{'='*55}")
    print(f"  Gap fill complete!")
    print(f"  IDs fetched             : {ids_fetched:,}")
    print(f"  Articles recovered      : {recovered:,}")
    print(f"  404 (no article)        : {misses:,}")
    print(f"  Blocked (403)           : {blocked:,}")
    print(f"  Throughput (15-min avg) : {rate:.1f} articles/min")
    print(f"  Written to instance_1   : {INSTANCE_DIRS[1].resolve()}")
    print(f"  Written to instance_2   : {INSTANCE_DIRS[2].resolve()}")
    print(f"\n{'='*55}\n")


# =====================================================================
# Entry Point
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BRecorder gap filler — finds and re-fetches missing articles"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show gap analysis report without fetching"
    )
    args = parser.parse_args()

    print("=" * 55)
    print(f"  BRecorder News Gap Filler")
    print(f"  ID range   : {ID_SCAN_START:,} → {ID_SCAN_END:,}")
    print(f"  Output     : {FILLER_DIR.resolve()}")
    print("=" * 55)

    log.info("Loading downloaded article IDs from both instances...")
    downloaded_ids = load_downloaded_ids()
    log.info(f"Total downloaded: {len(downloaded_ids):,} articles")

    missing_ids = analyse_gaps(downloaded_ids)
    print_gap_report(downloaded_ids, missing_ids)

    if args.status:
        sys.exit(0)

    if not missing_ids:
        print("  No gaps found — extraction is already complete!")
        sys.exit(0)

    print(f"  Starting gap fill for {len(missing_ids):,} unchecked IDs...\n")
    run_filler(downloaded_ids)
