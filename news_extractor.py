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
import gc
from datetime import datetime, date, timezone
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
# Total range split into two halves for parallel instances:
#   Instance 1 : 40,000,000 → 40,300,000  (lower half)
#   Instance 2 : 40,300,001 → 40,600,000  (upper half)
# Run each in a separate PowerShell window simultaneously.
INSTANCE_RANGES = {
    1: (40_000_000, 40_300_000),
    2: (40_300_001, 40_600_000),
}
# Defaults (overridden at runtime by --instance flag)
ID_SCAN_START     = 40_000_000
ID_SCAN_END       = 40_600_000
_current_instance = 1   # updated at runtime; used in restart message

# --- Session size ----------------------------------------------------
# Number of IDs to attempt per run. Adjust to fit available time.
# At ~1–2 s average per ID (mix of 404s and hits), 5000 IDs ≈ 2–3 hours.
# Run daily or overnight until complete.
SESSION_SIZE = 10_000

# --- Delays (seconds) ------------------------------------------------
DELAY_HIT  = 0.15  # fixed delay after every successful article fetch (200)
DELAY_MISS_MIN = 0.1   # after a 404 — fast, Cloudflare resolves these at edge
DELAY_MISS_MAX = 0.3

# --- Smart gap skipping ----------------------------------------------
GAP_TRIGGER = 50    # consecutive 404s before skipping
GAP_SKIP    = 200   # IDs to jump forward when gap detected

# --- Retry policy ----------------------------------------------------
MAX_RETRIES         = 3
RETRY_BACKOFF_403   = [4, 6, 9]    # seconds per attempt: quick first, escalate if persists
RETRY_BACKOFF_OTHER = [4, 6, 9]    # same progression for non-403 errors

# --- Output ----------------------------------------------------------
# Base output directory — each instance writes to its own subfolder
# to avoid any file conflicts when running in parallel.
BASE_OUTPUT_DIR = Path(r"D:\Stockmarket-Sentiment-Analysis\Extracted Data")
OUTPUT_DIR      = BASE_OUTPUT_DIR   # overridden at runtime by --instance flag
PROGRESS_FILE   = OUTPUT_DIR / "progress.json"

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
        "retries_used": 0,   # number of retry attempts made (0 = clean first-try)
        "scraped_at":  datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    }

    for attempt in range(1, MAX_RETRIES + 1):
        if attempt > 1:
            result["retries_used"] += 1   # count each retry attempt

        try:
            r = session.get(url, timeout=75, allow_redirects=True)
            result["http_status"] = r.status_code

            if r.status_code == 404:
                return result

            if r.status_code == 403:
                # Cloudflare 403 = "I don't trust this request" — retrying
                # the same URL just looks like a bot. Skip immediately.
                log.warning(f"  [{article_id}] 403 — skipping (no retry)")
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
# Main Session Runner
# =====================================================================

def run_session(session_size: int = SESSION_SIZE) -> None:
    """
    Scan up to session_size IDs starting from where the last session
    stopped. Saves each article immediately to its monthly CSV and
    updates progress.json every 100 IDs.

    Speed optimisations active:
      Strategy 1 — Smart gap skipping : after GAP_TRIGGER consecutive 404s,
                   jump article_id forward by GAP_SKIP to skip sparse regions.
      Strategy 2 — Adaptive delay     : hit delay floats between
                   ADAPTIVE_DELAY_MIN and ADAPTIVE_DELAY_MAX, stepping down
                   after each clean 200 and up after each 403.
      Strategy 3 — Fast 404 delay     : Cloudflare resolves 404s at the edge;
                   only 0.1–0.3 s sleep needed (no origin hit).

    Run this script repeatedly (daily, overnight) until complete.
    Each run picks up exactly where the last one ended.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    progress = load_progress()

    resume_from = progress["last_scanned_id"] + 1

    if resume_from > ID_SCAN_END:
        log.info("All IDs scanned — extraction complete.")
        print_status(progress)
        return

    session = make_session()

    ids_attempted    = 0
    hits             = 0    # 200 responses
    misses           = 0    # 404 responses
    blocked          = 0    # 403 responses
    session_articles = 0
    gaps_skipped     = 0    # times gap-skip triggered

    # Strategy 1: gap skipping state
    consecutive_misses = 0

    # 403 deferred retry queue — IDs blocked this checkpoint window.
    # Retried once every 100 IDs (just before progress save) after a
    # natural cooldown. Cleared after each retry pass.
    pending_403 = []

    log.info(f"Session: scanning up to {session_size:,} IDs from {resume_from:,}")
    log.info(f"  Hit delay            : {DELAY_HIT}s (fixed)")
    log.info(f"  Gap skip trigger     : {GAP_TRIGGER} consecutive 404s "
             f"→ jump +{GAP_SKIP} IDs")

    # ----- while loop (replaces for loop to support gap skipping) -----
    article_id = resume_from

    try:
        while article_id <= ID_SCAN_END and ids_attempted < session_size:
            ids_attempted += 1
            result = fetch_article(session, article_id)
            status = result["http_status"]

            if status == 200:
                # ── Strategy 1: reset miss counter on a real article ──
                consecutive_misses = 0

                hits += 1
                month_key = result["date"][:7] if result["date"] else "unknown"
                progress["month_counts"][month_key] = (
                    progress["month_counts"].get(month_key, 0) + 1
                )
                progress["total_articles"] += 1
                session_articles += 1

                append_to_monthly_csv(result)

                session_pct = ids_attempted / session_size * 100
                overall_pct = (article_id - ID_SCAN_START + 1) / \
                              (ID_SCAN_END - ID_SCAN_START + 1) * 100
                log.info(f"  {article_id}  SAVED  {result['date']}  "
                         f"\"{result['headline'][:55]}\"  "
                         f"[session {session_pct:.1f}% | overall {overall_pct:.1f}%]")

                time.sleep(DELAY_HIT)

            elif status == 404:
                # ── Strategy 1: count misses; skip ahead when gap detected ──
                consecutive_misses += 1
                misses += 1
                log.debug(f"  {article_id}  404  [streak={consecutive_misses}]")

                if consecutive_misses >= GAP_TRIGGER:
                    log.info(f"  {consecutive_misses} consecutive 404s detected — "
                             f"skipping {GAP_SKIP} IDs ahead")
                    article_id += GAP_SKIP
                    consecutive_misses = 0
                    gaps_skipped += 1

                # ── Strategy 3: fast 404 delay (Cloudflare edge, no origin) ──
                time.sleep(random.uniform(DELAY_MISS_MIN, DELAY_MISS_MAX))

            elif status == 403:
                # Cloudflare edge block — queue for retry at next checkpoint
                # (~100 IDs later) rather than retrying immediately.
                consecutive_misses = 0
                blocked += 1
                pending_403.append(article_id)
                log.warning(f"  {article_id}  BLOCKED (403) — queued for retry "
                            f"[{len(pending_403)} pending]")
                time.sleep(random.uniform(DELAY_MISS_MIN, DELAY_MISS_MAX))

            else:
                log.warning(f"  {article_id}  HTTP {status}")
                time.sleep(random.uniform(DELAY_MISS_MIN, DELAY_MISS_MAX))

            # Every 100 IDs: retry queued 403s then save progress
            progress["last_scanned_id"] = article_id
            if ids_attempted % 100 == 0:

                if pending_403:
                    log.info(f"  --- Retrying {len(pending_403)} queued 403 article(s) ---")
                    recovered = 0
                    for rid in pending_403:
                        r403 = fetch_article(session, rid)
                        if r403["http_status"] == 200:
                            month_key = r403["date"][:7] if r403["date"] else "unknown"
                            progress["month_counts"][month_key] = (
                                progress["month_counts"].get(month_key, 0) + 1
                            )
                            progress["total_articles"] += 1
                            session_articles += 1
                            append_to_monthly_csv(r403)
                            recovered += 1
                            log.info(f"    {rid}  RECOVERED  {r403['date']}  "
                                     f"\"{r403['headline'][:55]}\"")
                        else:
                            log.warning(f"    {rid}  still {r403['http_status']} — dropped")
                        time.sleep(random.uniform(DELAY_MISS_MIN, DELAY_MISS_MAX))
                    log.info(f"  --- Retry pass done: {recovered}/{len(pending_403)} recovered ---")
                    pending_403.clear()

                save_progress(progress)
                session_pct = ids_attempted / session_size * 100
                overall_pct = (article_id - ID_SCAN_START + 1) / \
                              (ID_SCAN_END - ID_SCAN_START + 1) * 100
                log.info(f"  --- Progress saved  "
                         f"[session {session_pct:.1f}% | overall {overall_pct:.1f}% | "
                         f"{session_articles} articles this session | "
                         f"total: {progress['total_articles']}] ---")

            # Every 500 IDs: clear session cookies and run garbage collection
            # to prevent curl_cffi accumulating session data and growing C: drive
            if ids_attempted % 500 == 0:
                session.cookies.clear()
                gc.collect()
                log.info(f"  --- Memory cleanup done (cookies cleared, GC run) ---")

            article_id += 1

    except KeyboardInterrupt:
        print(f"\n\n  Ctrl+C detected — saving progress and stopping cleanly...")
        progress["last_scanned_id"] = article_id
        if pending_403:
            print(f"  Retrying {len(pending_403)} queued 403(s) before exit...")
            for rid in pending_403:
                r403 = fetch_article(session, rid)
                if r403["http_status"] == 200:
                    month_key = r403["date"][:7] if r403["date"] else "unknown"
                    progress["month_counts"][month_key] = (
                        progress["month_counts"].get(month_key, 0) + 1
                    )
                    progress["total_articles"] += 1
                    session_articles += 1
                    append_to_monthly_csv(r403)
                    print(f"    {rid}  RECOVERED")
                time.sleep(random.uniform(DELAY_MISS_MIN, DELAY_MISS_MAX))
        save_progress(progress)
        session_pct = ids_attempted / session_size * 100
        overall_pct = (article_id - ID_SCAN_START + 1) / \
                      (ID_SCAN_END - ID_SCAN_START + 1) * 100
        print(f"\n{'='*55}")
        print(f"  Interrupted at ID     : {article_id:,}")
        print(f"  Session progress      : {session_pct:.1f}%  ({ids_attempted:,} / {session_size:,} IDs)")
        print(f"  Overall progress      : {overall_pct:.1f}%")
        print(f"  Articles this session : {session_articles:,}")
        print(f"  Total articles saved  : {progress['total_articles']:,}")
        print(f"\n  Progress saved. To resume, run:")
        print(f"\n      python news_extractor.py --instance {_current_instance}")
        print(f"\n{'='*55}\n")
        return

    # Final save
    save_progress(progress)

    remaining = ID_SCAN_END - progress["last_scanned_id"]
    session_pct = ids_attempted / session_size * 100
    overall_pct = (progress["last_scanned_id"] - ID_SCAN_START + 1) / \
                  (ID_SCAN_END - ID_SCAN_START + 1) * 100
    print(f"\n{'='*55}")
    print(f"  Session complete")
    print(f"  Session progress         : {session_pct:.1f}%  ({ids_attempted:,} IDs)")
    print(f"  Overall progress         : {overall_pct:.1f}%")
    print(f"  Articles saved           : {session_articles:,}")
    print(f"  404 (no article)         : {misses:,}")
    print(f"  Blocked (403)            : {blocked:,}")
    print(f"  Gap skips triggered      : {gaps_skipped:,}")
    print(f"  Total articles so far    : {progress['total_articles']:,}")
    print(f"  IDs remaining            : {remaining:,}")
    print(f"  Output folder            : {OUTPUT_DIR.resolve()}")
    print(f"\n  To continue, run:")
    print(f"\n      python news_extractor.py --instance {_current_instance}")
    print(f"\n{'='*55}\n")


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
        "--instance", type=int, choices=[1, 2], default=1,
        help="Instance number for parallel extraction (1=lower half, 2=upper half)"
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

    # --- Apply instance settings -------------------------------------
    ID_SCAN_START, ID_SCAN_END = INSTANCE_RANGES[args.instance]
    OUTPUT_DIR    = BASE_OUTPUT_DIR / f"instance_{args.instance}"
    PROGRESS_FILE = OUTPUT_DIR / "progress.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _current_instance = args.instance

    if args.debug:
        debug_selectors(40420446)

    elif args.status:
        p = load_progress()
        print_status(p)

    else:
        print("=" * 55)
        print(f"  BRecorder News Extractor  [Instance {args.instance}]")
        print(f"  ID range   : {ID_SCAN_START:,} → {ID_SCAN_END:,}")
        print(f"  Output     : {OUTPUT_DIR.resolve()}")
        print("=" * 55)

        # Ask user for session size interactively
        try:
            raw = input(f"\n  How many IDs to scan this session? "
                        f"[default: {SESSION_SIZE:,}]: ").strip()
            session_size = int(raw) if raw else args.session_size
            if session_size <= 0:
                raise ValueError
        except ValueError:
            print(f"  Invalid input — using default: {SESSION_SIZE:,}")
            session_size = args.session_size

        print(f"  Session    : {session_size:,} IDs (~"
              f"{session_size/2500:.1f}–{session_size/1500:.1f} hours)\n")
        run_session(session_size=session_size)
