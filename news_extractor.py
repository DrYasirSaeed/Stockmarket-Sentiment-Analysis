"""
Project: Information Transmission and Sectoral Asymmetry in the KSE-100
Module : Phase 0 — News Extractor (BRecorder Scraper)
Author : Dr. Yasir Saeed
Affiliation: Kohat University of Science & Technology (KUST)

Description:
  ID-scan extraction pipeline for BRecorder financial news.

  URL pattern confirmed: https://www.brecorder.com/news/<article_id>
  Article IDs are sequential integers. The extractor scans a range of IDs,
  fetches each page, extracts headline + body, and keeps only articles whose
  publication date falls within the target window.

  Output: brecorder_articles.csv — ready for topic_router.py classifier.

Setup — run once in PowerShell:
    pip install curl_cffi beautifulsoup4 lxml requests pandas
"""

# =====================================================================
# Step 0: Imports
# =====================================================================
import re
import time
import random
import logging
import sys
from datetime import datetime, date
from pathlib import Path

import requests
import pandas as pd
from bs4 import BeautifulSoup

try:
    from curl_cffi import requests as cffi_requests
    _CFFI_OK = True
except ImportError:
    _CFFI_OK = False
    print("ERROR: curl_cffi not installed.")
    print("Fix : pip install curl_cffi")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# =====================================================================
# Step 1: Configuration — edit these before each run
# =====================================================================

# --- Date filter (inclusive) -----------------------------------------
# Only articles published within this window are kept in the output CSV.
DATE_START = date(2026, 5, 2)
DATE_END   = date(2026, 5, 8)   # today

# --- ID scan range ---------------------------------------------------
# BRecorder article IDs are sequential 8-digit integers.
# Article 40420446 is confirmed live. Scan a window around it.
# Increase the range to cover more days; ~50–80 IDs per day is typical.
ID_SCAN_START = 40420200   # scan from here
ID_SCAN_END   = 40420600   # scan to here  (~400 IDs covers ~5–7 days)

# --- Rate limiting ---------------------------------------------------
DELAY_MIN = 2.0   # seconds between requests (randomised)
DELAY_MAX = 5.0

# --- Retry policy ----------------------------------------------------
MAX_RETRIES   = 3
RETRY_BACKOFF = 10   # seconds × attempt number

# --- Output ----------------------------------------------------------
OUTPUT_PATH      = Path(__file__).parent / "brecorder_articles.csv"
CHECKPOINT_EVERY = 25   # save progress every N successfully parsed articles

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
# Step 2: Session Factory
# =====================================================================

def make_session() -> cffi_requests.Session:
    """
    curl_cffi session impersonating Chrome 120.
    Replicates Chrome's exact TLS handshake so Cloudflare Bot Management
    cannot fingerprint it as automated traffic.
    Works from a home/university IP; datacenter IPs (Colab) stay blocked.
    """
    s = cffi_requests.Session(impersonate="chrome120")
    s.headers.update(HEADERS)
    log.info("Session ready: curl_cffi / chrome120 TLS impersonation")
    return s


# =====================================================================
# Step 3: HTML Parsers
# =====================================================================

def _first_text(soup: BeautifulSoup, selectors: list) -> str:
    """Try CSS selectors in order; return first non-empty text found."""
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
    # <meta property="article:published_time"> is the most reliable
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
# Step 4: Single-Article Fetch
# =====================================================================

def fetch_article(session, article_id: int) -> dict:
    """
    Fetch one article page. Returns a result dict with all extracted
    fields plus http_status. Returns http_status=0 on network error.
    """
    url    = ART_URL.format(article_id=article_id)
    result = {"article_id": article_id, "url": url,
              "headline": "", "body_text": "", "date": "",
              "author": "", "http_status": 0,
              "scraped_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, timeout=25, allow_redirects=True)
            result["http_status"] = r.status_code

            if r.status_code == 404:
                return result   # ID doesn't exist — skip silently

            if r.status_code == 403:
                log.warning(f"  [{article_id}] 403 (attempt {attempt}/{MAX_RETRIES})"
                            " — Cloudflare blocking this IP")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF * attempt)
                    continue
                return result

            if r.status_code != 200:
                log.warning(f"  [{article_id}] HTTP {r.status_code}")
                return result

            parsed = parse_page(r.text)
            result.update(parsed)
            return result

        except Exception as exc:
            log.warning(f"  [{article_id}] Error (attempt {attempt}): {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)

    return result


# =====================================================================
# Step 5: Full ID-Scan Pipeline
# =====================================================================

def _save_csv(records: list, path: Path) -> pd.DataFrame:
    df = pd.DataFrame(records)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


def _in_date_range(date_str: str, start: date, end: date) -> bool:
    """Return True if date_str (YYYY-MM-DD) falls within [start, end]."""
    if not date_str:
        return True   # keep articles with unparsed dates for manual review
    try:
        return start <= date.fromisoformat(date_str) <= end
    except ValueError:
        return True


def run(
    id_start:  int  = ID_SCAN_START,
    id_end:    int  = ID_SCAN_END,
    date_start: date = DATE_START,
    date_end:   date = DATE_END,
    output:    Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Scan article IDs from id_start to id_end inclusive.
    Keep only articles whose publication date is within [date_start, date_end].
    Saves results to output CSV with periodic checkpoints.
    """
    total    = id_end - id_start + 1
    session  = make_session()
    kept     = []    # articles within date range
    seen     = 0     # total IDs attempted
    skipped  = 0     # 404 / out-of-date-range
    blocked  = 0     # 403

    log.info(f"ID scan: {id_start} → {id_end}  ({total} IDs)")
    log.info(f"Date filter: {date_start} → {date_end}")
    log.info(f"Output: {output.resolve()}\n")

    for article_id in range(id_start, id_end + 1):
        seen += 1
        result = fetch_article(session, article_id)

        status = result["http_status"]

        if status == 404:
            log.debug(f"  {article_id}  404 — skip")
            skipped += 1

        elif status == 403:
            blocked += 1
            log.warning(f"  {article_id}  BLOCKED (403)  "
                        f"[kept={len(kept)} skipped={skipped} blocked={blocked}]")

        elif status == 200:
            if not _in_date_range(result["date"], date_start, date_end):
                skipped += 1
                log.info(f"  {article_id}  out of range ({result['date']}) — skip")
            else:
                kept.append(result)
                log.info(f"  {article_id}  OK  {result['date']}  "
                         f"\"{result['headline'][:60]}\"")

                if len(kept) % CHECKPOINT_EVERY == 0:
                    _save_csv(kept, output)
                    log.info(f"  >>> Checkpoint: {len(kept)} articles saved")
        else:
            skipped += 1
            log.warning(f"  {article_id}  HTTP {status} — skip")

        if article_id < id_end:
            time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

    df = _save_csv(kept, output) if kept else pd.DataFrame()

    print(f"\n{'='*55}")
    print(f"  Scan complete")
    print(f"  IDs scanned      : {seen}")
    print(f"  Articles kept    : {len(kept)}")
    print(f"  Skipped/no match : {skipped}")
    print(f"  Blocked (403)    : {blocked}")
    print(f"  Output           : {output.resolve()}")
    print(f"{'='*55}\n")
    return df


# =====================================================================
# Step 6: Selector Debugger — run first to verify HTML structure
# =====================================================================

def debug_selectors(article_id: int = 40420446) -> None:
    """
    Fetch one article and probe every CSS selector.
    Run this BEFORE a full scan to confirm the site is reachable and
    that headlines/body/date are being extracted correctly.
    """
    session = make_session()
    url     = ART_URL.format(article_id=article_id)
    log.info(f"Probing: {url}")

    r = session.get(url, timeout=25, allow_redirects=True)
    print(f"\nHTTP {r.status_code}   Final URL: {r.url}\n")

    if r.status_code != 200:
        print(f"Cannot parse — HTTP {r.status_code}.")
        if r.status_code == 403:
            print("Still blocked. Ensure you are running this from a home/university")
            print("network — NOT from Colab or any cloud server.")
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
        "Date      .publish-date":          soup.select_one(".publish-date"),
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

    # Also show live extracted values
    parsed = parse_page(r.text)
    print(f"\n─── Extracted Values ───────────────────────────────────────────")
    print(f"  Headline : {parsed['headline'][:80]}")
    print(f"  Date     : {parsed['date']}")
    print(f"  Author   : {parsed['author']}")
    print(f"  Body     : {parsed['body_text'][:200]}{'...' if len(parsed['body_text']) > 200 else ''}")

    print("\n─── Tags matching body|article|story|content ───────────────────")
    for tag in soup.find_all(class_=re.compile(r"body|article|story|content", re.I)):
        classes = " ".join(tag.get("class", []))
        snippet = tag.get_text(strip=True)[:60]
        print(f"  <{tag.name} class=\"{classes}\">  {snippet!r}")


# =====================================================================
# Step 7: Entry Point
# =====================================================================

if __name__ == "__main__":
    print("=" * 55)
    print("  BRecorder News Extractor")
    print(f"  ID range   : {ID_SCAN_START} → {ID_SCAN_END}")
    print(f"  Date filter: {DATE_START}  →  {DATE_END}")
    print(f"  Output     : {OUTPUT_PATH.resolve()}")
    print("=" * 55)

    # Step 1 — verify site is reachable and selectors are correct
    print("\n[Step 1 of 2]  Selector debug on article 40420446 ...\n")
    debug_selectors(40420446)

    # Step 2 — full ID scan with date filter
    print("\n[Step 2 of 2]  Running ID scan ...\n")
    df = run()

    if not df.empty:
        print("First 5 articles:")
        print(df[["article_id", "date", "headline", "http_status"]]
              .head()
              .to_string(index=False))
