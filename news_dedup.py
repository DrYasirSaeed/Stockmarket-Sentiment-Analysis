"""
Project: Information Transmission and Sectoral Asymmetry in the KSE-100
Module : Phase 0 — CSV Deduplicator
Author : Dr. Yasir Saeed
Affiliation: Kohat University of Science & Technology (KUST)

Description:
  Scans every brecorder_*.csv in instance_1 and instance_2 folders,
  removes duplicate rows (same article_id), and writes the clean file
  back in place. A backup of each file is saved before overwriting.

Usage:
    python news_dedup.py            # deduplicate all CSVs
    python news_dedup.py --status   # report duplicates without changing files
"""

import sys
import csv
import shutil
import argparse
import logging
from pathlib import Path

import pandas as pd

# Some article body fields exceed Python's default CSV field size limit
csv.field_size_limit(sys.maxsize)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_OUTPUT_DIR = Path(r"D:\Stockmarket-Sentiment-Analysis\Extracted Data")
INSTANCE_DIRS   = {
    1: BASE_OUTPUT_DIR / "instance_1",
    2: BASE_OUTPUT_DIR / "instance_2",
}
BACKUP_DIR = BASE_OUTPUT_DIR / "dedup_backup"


def dedup_file(csv_path: Path, dry_run: bool) -> tuple[int, int]:
    """
    Load csv_path in chunks (handles bloated files), drop duplicate
    article_ids, write back. Returns (original_count, duplicates_removed).
    """
    try:
        chunks = pd.read_csv(
            csv_path, dtype={"article_id": int}, chunksize=50_000,
            engine="python", on_bad_lines="skip"
        )
        df = pd.concat(chunks, ignore_index=True)
    except Exception as e:
        log.error(f"  Could not read {csv_path.name}: {e}")
        return 0, 0

    original   = len(df)
    df_clean   = df.drop_duplicates(subset=["article_id"], keep="first")
    duplicates = original - len(df_clean)

    if duplicates == 0:
        log.info(f"  {csv_path.name:<35}  {original:>6,} rows   no duplicates")
        return original, 0

    log.info(f"  {csv_path.name:<35}  {original:>6,} rows   "
             f"{duplicates:>5,} duplicates removed  →  {len(df_clean):,} rows")

    if not dry_run:
        # Back up original before overwriting
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        backup_path = BACKUP_DIR / f"{csv_path.parent.name}_{csv_path.name}"
        shutil.copy2(csv_path, backup_path)

        df_clean.to_csv(csv_path, index=False, encoding="utf-8-sig")

    return original, duplicates


def main(dry_run: bool) -> None:
    total_rows       = 0
    total_duplicates = 0
    files_affected   = 0

    print("=" * 55)
    print(f"  BRecorder CSV Deduplicator")
    if dry_run:
        print(f"  Mode: STATUS ONLY (no files changed)")
    else:
        print(f"  Mode: CLEAN  (backups saved to dedup_backup/)")
    print("=" * 55)

    for inst, folder in INSTANCE_DIRS.items():
        csv_files = sorted(folder.glob("brecorder_*.csv"))
        if not csv_files:
            log.warning(f"Instance {inst}: no CSV files found in {folder}")
            continue

        log.info(f"\nInstance {inst}  ({folder})")
        for csv_path in csv_files:
            rows, dupes = dedup_file(csv_path, dry_run)
            total_rows       += rows
            total_duplicates += dupes
            if dupes > 0:
                files_affected += 1

    print(f"\n{'='*55}")
    print(f"  Summary")
    print(f"  Total rows scanned   : {total_rows:,}")
    print(f"  Duplicate rows found : {total_duplicates:,}")
    print(f"  Files affected       : {files_affected}")
    if not dry_run and total_duplicates > 0:
        print(f"  Backups saved to     : {BACKUP_DIR.resolve()}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove duplicate article_id rows from BRecorder CSVs"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Report duplicates without modifying any files"
    )
    args = parser.parse_args()
    main(dry_run=args.status)
