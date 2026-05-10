"""
Project: Information Transmission and Sectoral Asymmetry in the KSE-100
Module : Phase 0 — News Extractor (BRecorder Scraper)
Author : Dr. Yasir Saeed
Affiliation: Kohat University of Science & Technology (KUST)

Description:
  Scrapes all BRecorder articles from ID 40000000 to the current date.
  Saves one CSV file per calendar month and tracks progress in a JSON
  file so any interruption can be resumed exactly where it stopped —
  no article is ever fetched twice.

Architecture:
  - progress.json  : tracks last scanned ID + per-month article counts
  - brecorder_YYYY_MM.csv : one file per month, appended incrementally
  - 404 responses use a short delay (0.3–0.8 s) to keep scanning fast
  - 200 responses use a polite delay (2–5 s) to avoid rate limiting
  - Each session processes SESSION_SIZE IDs then stops cleanly

Setup (run once):
    pip install curl_cffi beautifulsoup4 lxml requests pandas

Usage:
    python news_extractor.py              # resumes automatically
    python news_extractor.py --debug      # test single article only
    python news_extractor.py --status     # show progress without scraping
"""

import re
import time
import json
import random
import logging
import argparse
import sys
from datetime import datetime, date
from pathlib import Path

import requests
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

# --- ID range --------------------------------------------------------
# 40000000 is the confirmed earliest accessible article.
# ID_SCAN_END is set high — the run stops naturally at today's articles.
ID_SCAN_START = 40_000_000
ID_SCAN_END   = 40_600_000   # well beyond current (~40420600 as of May 2026)

# --- Session size ----------------------------------------------------
# Number of IDs to attempt per run. Adjust to fit available time.
# At ~1–2 s average per ID (mix of 404s and hits), 5000 IDs ≈ 2–3 hours.
# Run daily or overnight until complete.
SESSION_SIZE = 5_000

# --- Delays (seconds) ------------------------------------------------
DELAY_HIT_MIN  = 2.0   # after a successful article fetch (200)
DELAY_HIT_MAX  = 5.0
DELAY_MISS_MIN = 0.3   # after a 404 — keep the scan moving quickly
DELAY_MISS_MAX = 0.8
DELAY_ERR_MIN  = 8.0   # after a 403 / network error — back off
DELAY_ERR_MAX  = 15.0

# --- Retry policy ----------------------------------------------------
MAX_RETRIES   = 3
RETRY_BACKOFF = 10   # seconds × attempt number

# --- Output ----------------------------------------------------------
OUTPUT_DIR    = Path(__file__).parent / "brecorder_data"
PROGRESS_FILE = OUTPUT_DIR / "progress.json"

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
# Progress Tracking
# =====================================================================

def load_progress() -> dict:
    """
    Load progress from disk. Returns default state if no progress file
    exists yet (i.e. first ever run).
    """
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            p = json.load(f)
        log.info(f"Resuming from ID {p['last_scanned_id'] + 1}  "
                 f"({p['total_articles']} articles collected so far)")
        return p
    log.info(f"No progress file found — starting fresh from ID {ID_SCAN_START}")
    return {
        "last_scanned_id":  ID_SCAN_START - 1,
        "total_articles":   0,
        "month_counts":     {},   # {"2023-01": 45, "2023-02": 67, ...}
        "started_at":       datetime.utcnow().isoformat(),
        "last_updated":     datetime.utcnow().isoformat(),
    }


def save_progress(p: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    p["last_updated"] = datetime.utcnow().isoformat()
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(p, f, indent=2)


def print_status(p: dict) -> None:
    """Print a human-readable summary of current progress."""
    done    = p["last_scanned_id"] - ID_SCAN_START + 1
    total   = ID_SCAN_END - ID_SCAN_START + 1
    pct     = done / total * 100 if total > 0 else 0
    remain  = ID_SCAN_END - p["last_scanned_id"]

    print(f"\n{'='*55}")
    print(f"  Progress Report")
    print(f"  Started          : {p.get('started_at','—')[:19]}")
    print(f"  Last updated     : {p.get('last_updated','—')[:19]}")
    print(f"  IDs scanned      : {done:,} / {total:,}  ({pct:.1f}%)")
    print(f"  IDs remaining    : {remain:,}")
    print(f"  Articles saved   : {p['total_articles']:,}")
    print(f"\n  Monthly breakdown:")
    for month, count in sorted(p["month_counts"].items()):
        print(f"    {month}  →  {count:>5} articles")
    print(f"{'='*55}\n")


# =====================================================================
# Monthly CSV Management
# =====================================================================

def monthly_csv_path(date_str: str) -> Path:
    """
    Return the Path for the monthly CSV that should hold this article.
    Format: brecorder_YYYY_MM.csv
    Falls back to brecorder_unknown.csv if date cannot be parsed.
    """
    try:
        d = date.fromisoformat(date_str[:10])
        return OUTPUT_DIR / f"brecorder_{d.year}_{d.month:02d}.csv"
    except (ValueError, TypeError):
        return OUTPUT_DIR / "brecorder_unknown.csv"


def append_to_monthly_csv(record: dict) -> None:
    """
    Append one article record to the correct monthly CSV.
    Creates the file with a header row if it does not exist yet;
    appends without header if the file already exists.
    """
    path = monthly_csv_path(record.get("date", ""))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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

def _first_text(soup: BeautifulSoup, selectors: list) -> str:
    for sel in selectors:
        el = soup.select_one(sel)
        if el and el.get_text(strip=True):
            return el.get_text(separator=" ", strip=True)
    return ""


def _extract_headline(soup: BeautifulSoup) -> str:
    return _first_text(soup, [
        "h1.title", "h1.article-title", "h1.entry-title",
        "h1.post-title", "article h1", "h1",
    ])


def _extract_body(soup: BeautifulSoup) -> str:
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


def _extract_date(soup: BeautifulSoup) -> str:
    meta = soup.find("meta", property="article:published_time")
    if meta and meta.get("content"):
        return meta["content"][:10]
    raw = _first_text(soup, [
        "time[datetime]", ".publish-date", ".story-date",
        ".post-date", ".entry-date", ".date",
    ])
    m = re.search(r"\d{4}-\d{2}-\d{2}", raw)
    return m.group() if m else raw[:30]


def _extract_author(soup: BeautifulSoup) -> str:
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
        "article_id":  article_id,
        "url":         url,
        "headline":    "",
        "body_text":   "",
        "date":        "",
        "author":      "",
        "http_status": 0,
        "scraped_at":  datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, timeout=25, allow_redirects=True)
            result["http_status"] = r.status_code

            if r.status_code == 404:
                return result

            if r.status_code == 403:
                log.warning(f"  [{article_id}] 403 (attempt {attempt}/{MAX_RETRIES})")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF * attempt)
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
                time.sleep(RETRY_BACKOFF * attempt)

    return result


# =====================================================================
# Main Session Runner
# =====================================================================

def run_session(session_size: int = SESSION_SIZE) -> None:
    """
    Scan up to session_size IDs starting from where the last session
    stopped. Saves each article immediately to its monthly CSV and
    updates progress.json every 100 IDs.

    Run this script repeatedly (daily, overnight) until complete.
    Each run picks up exactly where the last one ended.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    progress = load_progress()

    resume_from = progress["last_scanned_id"] + 1
    scan_to     = min(resume_from + session_size - 1, ID_SCAN_END)

    if resume_from > ID_SCAN_END:
        log.info("All IDs scanned — extraction complete.")
        print_status(progress)
        return

    session = make_session()

    ids_attempted  = 0
    hits           = 0   # 200 responses
    misses         = 0   # 404 responses
    blocked        = 0   # 403 responses
    session_articles = 0

    log.info(f"Session: scanning IDs {resume_from} → {scan_to}  "
             f"({scan_to - resume_from + 1} IDs)")

    for article_id in range(resume_from, scan_to + 1):
        ids_attempted += 1
        result = fetch_article(session, article_id)
        status = result["http_status"]

        if status == 200:
            hits += 1
            month_key = result["date"][:7] if result["date"] else "unknown"
            progress["month_counts"][month_key] = (
                progress["month_counts"].get(month_key, 0) + 1
            )
            progress["total_articles"] += 1
            session_articles += 1

            append_to_monthly_csv(result)

            log.info(f"  {article_id}  SAVED  {result['date']}  "
                     f"\"{result['headline'][:65]}\"")

            time.sleep(random.uniform(DELAY_HIT_MIN, DELAY_HIT_MAX))

        elif status == 404:
            misses += 1
            log.debug(f"  {article_id}  404")
            time.sleep(random.uniform(DELAY_MISS_MIN, DELAY_MISS_MAX))

        elif status == 403:
            blocked += 1
            log.warning(f"  {article_id}  BLOCKED (403) — "
                        f"backing off {DELAY_ERR_MAX:.0f}s")
            time.sleep(random.uniform(DELAY_ERR_MIN, DELAY_ERR_MAX))

        else:
            log.warning(f"  {article_id}  HTTP {status}")
            time.sleep(random.uniform(DELAY_MISS_MIN, DELAY_MISS_MAX))

        # Save progress every 100 IDs
        progress["last_scanned_id"] = article_id
        if ids_attempted % 100 == 0:
            save_progress(progress)
            log.info(f"  --- Progress saved  "
                     f"[session: {session_articles} articles | "
                     f"total: {progress['total_articles']}] ---")

    # Final save
    save_progress(progress)

    remaining = ID_SCAN_END - scan_to
    print(f"\n{'='*55}")
    print(f"  Session complete")
    print(f"  IDs scanned this session : {ids_attempted:,}")
    print(f"  Articles saved           : {session_articles:,}")
    print(f"  404 (no article)         : {misses:,}")
    print(f"  Blocked (403)            : {blocked:,}")
    print(f"  Total articles so far    : {progress['total_articles']:,}")
    print(f"  IDs remaining            : {remaining:,}")
    pct = (progress["last_scanned_id"] - ID_SCAN_START + 1) / \
          (ID_SCAN_END - ID_SCAN_START + 1) * 100
    print(f"  Overall progress         : {pct:.1f}%")
    print(f"  Output folder            : {OUTPUT_DIR.resolve()}")
    print(f"\n  Run the script again to continue.")
    print(f"{'='*55}\n")


# =====================================================================
# Selector Debugger
# =====================================================================

def debug_selectors(article_id: int = 40420446) -> None:
    session = make_session()
    url     = ART_URL.format(article_id=article_id)
    log.info(f"Probing: {url}")
    r = session.get(url, timeout=25, allow_redirects=True)
    print(f"\nHTTP {r.status_code}   Final URL: {r.url}\n")
    if r.status_code != 200:
        print(f"Cannot parse — HTTP {r.status_code}")
        return
    soup = BeautifulSoup(r.text, "lxml")
    probes = {
        "Headline  h1.title":               soup.select_one("h1.title"),
        "Headline  h1.article-title":       soup.select_one("h1.article-title"),
        "Headline  article h1":             soup.select_one("article h1"),
        "Headline  h1 (generic)":           soup.find("h1"),
        "Body      .story-body":            soup.select_one(".story-body"),
        "Body      .article-body":          soup.select_one(".article-body"),
        "Body      .entry-content":         soup.select_one(".entry-content"),
        "Body      [itemprop=articleBody]": soup.select_one('[itemprop="articleBody"]'),
        "Body      article":                soup.find("article"),
        "Date      meta published_time":    soup.find("meta", property="article:published_time"),
        "Date      time[datetime]":         soup.find("time"),
        "Author    .author-name":           soup.select_one(".author-name"),
        "Author    .byline":                soup.select_one(".byline"),
    }
    print("─── Selector Probe ─────────────────────────────────────────────")
    for label, el in probes.items():
        if el is None:
            print(f"  MISS   {label}")
        else:
            snippet = (el.get_text(strip=True) if hasattr(el, "get_text")
                       else el.get("content", ""))[:80]
            print(f"  FOUND  {label:<46} → {snippet!r}")
    parsed = parse_page(r.text)
    print(f"\n─── Extracted Values ───────────────────────────────────────────")
    print(f"  Headline : {parsed['headline'][:80]}")
    print(f"  Date     : {parsed['date']}")
    print(f"  Author   : {parsed['author']}")
    print(f"  Body     : {parsed['body_text'][:200]}"
          f"{'...' if len(parsed['body_text']) > 200 else ''}")


# =====================================================================
# Entry Point
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BRecorder news extractor — resumes automatically from last position"
    )
    parser.add_argument(
        "--debug",  action="store_true",
        help="Run selector debug on article 40420446 only (no scraping)"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Print progress summary without scraping"
    )
    parser.add_argument(
        "--session-size", type=int, default=SESSION_SIZE,
        help=f"IDs to scan this run (default: {SESSION_SIZE})"
    )
    args = parser.parse_args()

    if args.debug:
        debug_selectors(40420446)

    elif args.status:
        p = load_progress()
        print_status(p)

    else:
        print("=" * 55)
        print("  BRecorder News Extractor")
        print(f"  Full range : {ID_SCAN_START:,} → {ID_SCAN_END:,}")
        print(f"  Session    : {args.session_size:,} IDs this run")
        print(f"  Output     : {OUTPUT_DIR.resolve()}")
        print("=" * 55 + "\n")
        run_session(session_size=args.session_size)
