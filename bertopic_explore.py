"""
BERTopic exploration script for Business Recorder financial news corpus.

Flags:
  --refit   Force re-encode embeddings and refit model (deletes all caches).
  --status  Print which output files exist, then exit.

Speed notes
-----------
Two expensive steps are cached to disk so they only run once:

  embeddings_cache.npy  — sentence-transformer output (~minutes on CPU)
  umap_cache.npy        — UMAP-reduced embeddings  (~hours on CPU, 227K docs)

On subsequent runs the script loads both caches and jumps straight to
HDBSCAN clustering and output generation.
Pass --refit to delete all caches and start from scratch.

Memory note
-----------
calculate_probabilities=True allocates a (n_docs × n_topics) float64 matrix
that can exceed 7 GB on large corpora. It is set to False here.
Per-article distributions are still available via approximate_distribution()
which is used for the 10-article terminal sample.
"""

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
import os

import sys as _sys

# ---------------------------------------------------------------------------
# Environment detection — same script runs locally and on Colab
# ---------------------------------------------------------------------------
_ON_COLAB = "google.colab" in _sys.modules or "/content/" in __file__

if _ON_COLAB:
    _BASE = "/content/drive/MyDrive/Github-Cursor/Stockmarket-Sentiment-Analysis"
else:
    _BASE = r"D:\Stockmarket-Sentiment-Analysis"

CONFIG = {
    # ── Input ────────────────────────────────────────────────────────────────
    "data_dir":         _BASE + ("/Extracted Data" if _ON_COLAB else r"\Extracted Data"),

    # ── Saved artefacts ──────────────────────────────────────────────────────
    "model_path":       _BASE + ("/bertopic_model.pkl" if _ON_COLAB else r"\bertopic_model.pkl"),
    # Sentence-transformer output — recomputed only on --refit or corpus change
    "embedding_cache":  _BASE + ("/embeddings_cache.npy" if _ON_COLAB else r"\embeddings_cache.npy"),
    # UMAP-reduced embeddings — recomputed only on --refit or corpus change
    "umap_cache":       _BASE + ("/umap_cache.npy" if _ON_COLAB else r"\umap_cache.npy"),

    # ── Output directory ─────────────────────────────────────────────────────
    "out_dir":          _BASE + ("/BERTopic Outputs" if _ON_COLAB else r"\BERTopic Outputs"),

    # ── Output filenames (all placed inside out_dir) ──────────────────────
    "topics_xlsx":      "bertopic_topics.xlsx",
    "assignments_csv":  "bertopic_assignments.csv",
    "topic_info_txt":   "topic_info.txt",
    "time_xlsx":        "topic_over_time.xlsx",

    # Visualization base-names  (.html and .png are appended automatically)
    "viz_topics":       "viz_intertopic_distance",
    "viz_hierarchy":    "viz_hierarchy",
    "viz_barchart":     "viz_barchart",
    "viz_heatmap":      "viz_heatmap",
    "viz_documents":    "viz_documents",
    "viz_time":         "viz_topics_over_time",

    # ── Embedding ─────────────────────────────────────────────────────────────
    "embedding_model":  "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "embed_batch_size": 128 if _ON_COLAB else 64,   # GPU batch on Colab, CPU batch locally
    "embed_device":     "cuda" if _ON_COLAB else None,  # auto-detected locally

    # ── BERTopic / UMAP / HDBSCAN ────────────────────────────────────────────
    "umap_n_neighbors": 15,
    "umap_n_components": 5,
    "umap_min_dist":    0.0,
    "umap_metric":      "cosine",

    "hdbscan_min_cluster":  30,
    "hdbscan_min_samples":  10,
    "hdbscan_method":       "eom",

    "ngram_range":          (1, 1),    # unigrams only — bigrams OOM on 413K docs at 12GB RAM
    "min_df":               10,        # raised from 5 — cuts rare terms, reduces matrix size
    "max_features":         50000,     # hard cap on vocabulary to prevent sparse matrix OOM
    "nr_topics":            "auto",

    # ── Misc ─────────────────────────────────────────────────────────────────
    "sanity_keywords":      ["sbp", "imf", "fbr", "nepra"],
    "distribution_sample":  10,    # articles printed for distribution preview
}

OUTPUT_FILES = [
    CONFIG["topics_xlsx"],
    CONFIG["assignments_csv"],
    CONFIG["topic_info_txt"],
    CONFIG["time_xlsx"],
    CONFIG["viz_topics"]    + ".html",  CONFIG["viz_topics"]    + ".png",
    CONFIG["viz_hierarchy"] + ".html",  CONFIG["viz_hierarchy"] + ".png",
    CONFIG["viz_barchart"]  + ".html",  CONFIG["viz_barchart"]  + ".png",
    CONFIG["viz_heatmap"]   + ".html",  CONFIG["viz_heatmap"]   + ".png",
    CONFIG["viz_documents"] + ".html",  CONFIG["viz_documents"] + ".png",
    CONFIG["viz_time"]      + ".html",  CONFIG["viz_time"]      + ".png",
]

# ---------------------------------------------------------------------------
# IMPORTS — kept minimal at module level so --status works without heavy deps
# ---------------------------------------------------------------------------
import argparse
import sys


def parse_args():
    p = argparse.ArgumentParser(description="BERTopic explorer for BR corpus")
    p.add_argument("--refit",  action="store_true",
                   help="Delete all caches + saved model and refit from scratch")
    p.add_argument("--status", action="store_true",
                   help="Show output file status and exit without running anything")
    return p.parse_args()


def status_check():
    out_dir = CONFIG["out_dir"]
    print("\n=== Output file status ===")
    all_present = True
    for fname in OUTPUT_FILES:
        fpath  = os.path.join(out_dir, fname)
        exists = os.path.isfile(fpath)
        tag    = "[OK]     " if exists else "[MISSING]"
        if not exists:
            all_present = False
        print(f"  {tag} {fname}")

    for label, path in [
        ("bertopic_model.pkl",   CONFIG["model_path"]),
        ("embeddings_cache.npy", CONFIG["embedding_cache"]),
        ("umap_cache.npy",       CONFIG["umap_cache"]),
    ]:
        exists = os.path.isfile(path)
        print(f"\n  {'[OK]     ' if exists else '[MISSING]'} {label}  ({path})")

    print("\nAll outputs present." if all_present else "\nSome outputs are missing.")
    print()


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def _detect_device():
    """Return 'cuda' if a CUDA GPU is available, else 'cpu'."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class CachedUMAP:
    """
    Drop-in UMAP replacement that returns a pre-computed embedding matrix.

    Used so BERTopic's fit_transform() skips the UMAP fitting step entirely
    when a cached umap_cache.npy already exists.  After fitting, BERTopic's
    internal umap_model is replaced with the real fitted UMAP before the model
    is saved to disk, so future transform() calls work correctly.
    """
    def __init__(self, cached):
        self.cached     = cached
        self.embedding_ = cached   # attribute BERTopic may inspect

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.cached

    def fit_transform(self, X, y=None):
        return self.cached


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    if args.status:
        status_check()
        sys.exit(0)

    # ── Heavy imports ────────────────────────────────────────────────────────
    import glob
    import pickle

    import numpy as np
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP
    from hdbscan import HDBSCAN
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired
    import plotly.io as pio

    # kaleido ≥1.0 is a subprocess binary — do NOT import it as a module.
    # plotly auto-detects it when write_image() is called, as long as it is
    # installed and Chrome is available (required by kaleido ≥1.0 on Windows).

    os.makedirs(CONFIG["out_dir"], exist_ok=True)

    def out(fname):
        return os.path.join(CONFIG["out_dir"], fname)

    # ────────────────────────────────────────────────────────────────────────
    # STEP 1 — LOAD DATA
    # ────────────────────────────────────────────────────────────────────────
    print("\n[1/7] Loading CSV files …")
    csv_files = glob.glob(os.path.join(CONFIG["data_dir"], "*.csv"))
    if not csv_files:
        sys.exit(f"No CSV files found in {CONFIG['data_dir']}")

    frames = []
    for f in csv_files:
        try:
            frames.append(pd.read_csv(f, low_memory=False, on_bad_lines="skip"))
        except Exception as e:
            print(f"  WARNING: could not read {f}: {e}")

    df = pd.concat(frames, ignore_index=True)
    print(f"  Raw rows: {len(df):,}")

    if "http_status" in df.columns:
        df = df[df["http_status"] == 200].copy()
        print(f"  After http_status==200 filter: {len(df):,}")

    required = ["article_id", "headline", "body_text"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        sys.exit(f"CSV is missing required columns: {missing_cols}")

    df["headline"]  = df["headline"].fillna("").astype(str)
    df["body_text"] = df["body_text"].fillna("").astype(str)
    df["text"] = (
        df["headline"] + " " +
        df["headline"] + " " +
        df["headline"] + " " +
        df["body_text"]
    )
    df = df[df["text"].str.strip().str.len() > 0].reset_index(drop=True)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    docs       = df["text"].tolist()
    timestamps = df["date"].tolist() if "date" in df.columns else None
    total_docs = len(docs)
    print(f"  Final usable documents: {total_docs:,}")

    # ────────────────────────────────────────────────────────────────────────
    # STEP 2 — SENTENCE-TRANSFORMER EMBEDDINGS  (cached)
    # ────────────────────────────────────────────────────────────────────────
    embed_cache  = CONFIG["embedding_cache"]
    umap_cache   = CONFIG["umap_cache"]
    model_path   = CONFIG["model_path"]

    # Delete all caches if --refit requested
    if args.refit:
        for path in [embed_cache, umap_cache, model_path]:
            if os.path.isfile(path):
                os.remove(path)
                print(f"  Deleted: {path}")

    cache_exists = os.path.isfile(embed_cache)
    model_exists = os.path.isfile(model_path)

    if cache_exists:
        print(f"\n[2/7] Loading cached embeddings from {embed_cache} …")
        embeddings = np.load(embed_cache)
        if len(embeddings) != total_docs:
            print(f"  WARNING: cache has {len(embeddings):,} rows but corpus has "
                  f"{total_docs:,} — discarding embedding + UMAP caches and re-encoding.")
            for path in [embed_cache, umap_cache]:
                if os.path.isfile(path):
                    os.remove(path)
            cache_exists = False

    if not cache_exists:
        device = CONFIG.get("embed_device") or _detect_device()
        print(f"\n[2/7] Encoding {total_docs:,} documents "
              f"(model={CONFIG['embedding_model']}, device={device}) …")
        print("  This is the slowest step; subsequent runs skip it via the cache.")
        st_model   = SentenceTransformer(CONFIG["embedding_model"], device=device)
        embeddings = st_model.encode(
            docs,
            batch_size        = CONFIG["embed_batch_size"],
            show_progress_bar = True,
            convert_to_numpy  = True,
        )
        np.save(embed_cache, embeddings)
        print(f"  Embeddings saved to {embed_cache}  (shape: {embeddings.shape})")

    # ────────────────────────────────────────────────────────────────────────
    # STEP 2b — UMAP REDUCTION  (cached separately — takes ~2 hrs on CPU)
    # ────────────────────────────────────────────────────────────────────────
    umap_exists = os.path.isfile(umap_cache)

    # Build the real UMAP model (needed whether we fit or just load)
    umap_model = UMAP(
        n_neighbors  = CONFIG["umap_n_neighbors"],
        n_components = CONFIG["umap_n_components"],
        min_dist     = CONFIG["umap_min_dist"],
        metric       = CONFIG["umap_metric"],
        random_state = 42,
    )

    if umap_exists:
        print(f"\n[2b/7] Loading cached UMAP embeddings from {umap_cache} …")
        umap_embeddings = np.load(umap_cache)
        if len(umap_embeddings) != total_docs:
            print(f"  WARNING: UMAP cache size mismatch — re-running UMAP.")
            os.remove(umap_cache)
            umap_exists = False

    if not umap_exists:
        print(f"\n[2b/7] Running UMAP on {total_docs:,} documents …")
        print("  This takes ~2 hrs on CPU for 227K docs. Cached after first run.")
        umap_embeddings = umap_model.fit_transform(embeddings)
        np.save(umap_cache, umap_embeddings)
        print(f"  UMAP embeddings saved to {umap_cache}  (shape: {umap_embeddings.shape})")
    else:
        # Fit the real UMAP model on the cached output so it can transform new
        # points correctly when loaded from the saved pkl.
        # This is fast — we're fitting on already-reduced 5D data.
        print("  Re-fitting UMAP model on cached embeddings for transform() …")
        umap_model.fit(embeddings, y=None)

    # ────────────────────────────────────────────────────────────────────────
    # STEP 3 — FIT / LOAD BERTOPIC MODEL
    # ────────────────────────────────────────────────────────────────────────
    model_exists = os.path.isfile(model_path)

    if model_exists:
        print(f"\n[3/7] Loading saved model from {model_path} …")
        with open(model_path, "rb") as fh:
            topic_model = pickle.load(fh)
        print("  Transforming corpus (using cached UMAP embeddings) …")
        # Swap in CachedUMAP so transform() skips the slow UMAP step
        topic_model.umap_model = CachedUMAP(umap_embeddings)
        topics, probs = topic_model.transform(docs, embeddings=embeddings)
        # Restore the real umap_model in memory (pkl on disk is unchanged)
        topic_model.umap_model = umap_model

    else:
        print("\n[3/7] Fitting BERTopic …")
        print("  UMAP already done — only HDBSCAN + representation remain.")

        hdbscan_model = HDBSCAN(
            min_cluster_size         = CONFIG["hdbscan_min_cluster"],
            min_samples              = CONFIG["hdbscan_min_samples"],
            cluster_selection_method = CONFIG["hdbscan_method"],
            prediction_data          = True,
        )

        vectorizer_model = CountVectorizer(
            ngram_range  = CONFIG["ngram_range"],
            min_df       = CONFIG["min_df"],
            max_features = CONFIG.get("max_features"),   # None = no cap (safe default)
            stop_words   = "english",
        )

        topic_model = BERTopic(
            embedding_model         = CONFIG["embedding_model"],
            umap_model              = CachedUMAP(umap_embeddings),  # skips UMAP step
            hdbscan_model           = hdbscan_model,
            vectorizer_model        = vectorizer_model,
            representation_model    = KeyBERTInspired(),
            nr_topics               = CONFIG["nr_topics"],
            calculate_probabilities = False,   # True needs ~7 GB RAM — not feasible
            verbose                 = True,
        )

        # Pre-computed embeddings passed → BERTopic skips its internal encoder.
        # CachedUMAP passed as umap_model → BERTopic skips UMAP fitting.
        # Only HDBSCAN + topic representation run fresh.
        topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)

        # Replace CachedUMAP with the real fitted UMAP before saving so that
        # future transform() calls on new documents work correctly.
        topic_model.umap_model = umap_model

        print(f"\n  Saving model to {model_path} …")
        with open(model_path, "wb") as fh:
            pickle.dump(topic_model, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print("  Model saved.")

    # ────────────────────────────────────────────────────────────────────────
    # STEP 4 — TOPIC INFO TXT
    # ────────────────────────────────────────────────────────────────────────
    print("\n[4/7] Writing topic_info.txt …")
    topic_info_df = topic_model.get_topic_info()
    all_topics    = topic_model.get_topics()

    lines = [
        "=" * 80,
        "get_topic_info()",
        "=" * 80,
        topic_info_df.to_string(index=False),
        "",
        "=" * 80,
        "get_topics()  — topic_id : [(word, score), …]",
        "=" * 80,
    ]
    for tid, words in sorted(all_topics.items()):
        lines.append(f"\nTopic {tid:>4d}: {words}")

    info_text = "\n".join(lines)
    print(info_text)

    txt_path = out(CONFIG["topic_info_txt"])
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(info_text)
    print(f"  Saved → {txt_path}")

    # ────────────────────────────────────────────────────────────────────────
    # STEP 5 — TOPIC SPREADSHEET + ASSIGNMENTS CSV
    # ────────────────────────────────────────────────────────────────────────
    print("\n[5/7] Writing bertopic_topics.xlsx and bertopic_assignments.csv …")

    # --- topics xlsx ---------------------------------------------------------
    rows = []
    for _, row in topic_info_df.iterrows():
        tid   = row["Topic"]
        count = row["Count"]
        words = topic_model.get_topic(tid)
        rows.append({
            "topic_id":         tid,
            "top_10_keywords":  ", ".join(w for w, _ in words[:10]) if words else "",
            "document_count":   count,
            "pct_of_corpus":    round(count / total_docs * 100, 4),
        })
    pd.DataFrame(rows).to_excel(out(CONFIG["topics_xlsx"]), index=False)
    print(f"  Saved → {out(CONFIG['topics_xlsx'])}")

    # --- assignments csv -----------------------------------------------------
    # calculate_probabilities=False → probs is None; column written as blank
    if probs is not None and np.ndim(probs) == 2:
        topic_probs = probs[np.arange(len(topics)), [max(t, 0) for t in topics]]
    elif probs is not None:
        topic_probs = probs
    else:
        topic_probs = [None] * len(topics)

    assign_df = pd.DataFrame({
        "article_id":        df["article_id"].values,
        "date":              df["date"].values if "date" in df.columns else [None] * len(df),
        "headline":          df["headline"].values,
        "topic_id":          topics,
        "topic_probability": topic_probs,
    })
    assign_df.to_csv(out(CONFIG["assignments_csv"]), index=False)
    print(f"  Saved → {out(CONFIG['assignments_csv'])}")

    # ────────────────────────────────────────────────────────────────────────
    # STEP 6 — VISUALIZATIONS
    # ────────────────────────────────────────────────────────────────────────
    print("\n[6/7] Generating visualizations …")

    def save_fig(fig, base_name):
        html_path = out(base_name + ".html")
        png_path  = out(base_name + ".png")
        fig.write_html(html_path)
        try:
            pio.write_image(fig, png_path, width=1400, height=900, scale=2)
        except Exception as e:
            print(f"  WARNING: PNG export failed ({e})")
            print("  Tip: ensure kaleido≥1.0 is installed and Chrome is available.")
        print(f"  Saved → {html_path}")
        print(f"         → {png_path}")

    # 6a. Intertopic distance map
    print("  visualize_topics() …")
    try:
        save_fig(topic_model.visualize_topics(), CONFIG["viz_topics"])
    except Exception as e:
        print(f"  WARNING: {e}")

    # 6b. Hierarchy
    print("  visualize_hierarchy() …")
    try:
        save_fig(topic_model.visualize_hierarchy(), CONFIG["viz_hierarchy"])
    except Exception as e:
        print(f"  WARNING: {e}")

    # 6c. Barchart
    print("  visualize_barchart() …")
    try:
        n_vis = min(30, len([t for t in all_topics if t != -1]))
        save_fig(
            topic_model.visualize_barchart(top_n_topics=n_vis, n_words=10),
            CONFIG["viz_barchart"],
        )
    except Exception as e:
        print(f"  WARNING: {e}")

    # 6d. Heatmap
    print("  visualize_heatmap() …")
    try:
        save_fig(topic_model.visualize_heatmap(), CONFIG["viz_heatmap"])
    except Exception as e:
        print(f"  WARNING: {e}")

    # 6e. Document scatter — 2D UMAP on the cached high-dim embeddings
    print("  visualize_documents() …")
    try:
        umap_2d = UMAP(
            n_neighbors  = CONFIG["umap_n_neighbors"],
            n_components = 2,
            min_dist     = CONFIG["umap_min_dist"],
            metric       = CONFIG["umap_metric"],
            random_state = 42,
        )
        reduced_2d = umap_2d.fit_transform(embeddings)
        save_fig(
            topic_model.visualize_documents(
                docs,
                embeddings          = reduced_2d,
                hide_document_hover = False,
            ),
            CONFIG["viz_documents"],
        )
    except Exception as e:
        print(f"  WARNING: {e}")

    # 6f. Per-article topic distribution — terminal preview for 10 articles
    print("\n  Topic probability distribution — sample articles:")
    sample_idx = np.linspace(0, len(docs) - 1, CONFIG["distribution_sample"], dtype=int)
    print("-" * 80)
    for idx in sample_idx:
        headline_preview = df["headline"].iloc[idx][:80]
        try:
            article_probs, _ = topic_model.approximate_distribution(
                [docs[idx]], window=8, batch_size=32
            )
            top5 = sorted(enumerate(article_probs[0]), key=lambda x: -x[1])[:5]
            top_str = "  |  ".join(
                f"T{t:>3d}: {p:.3f}" for t, p in top5 if p > 0.001
            )
        except Exception as e:
            top_str = f"(distribution unavailable: {e})"
        print(f"\n  [{idx:>5d}] {headline_preview}")
        print(f"         {top_str}")
    print("-" * 80)

    # ────────────────────────────────────────────────────────────────────────
    # STEP 7 — TOPICS OVER TIME
    # ────────────────────────────────────────────────────────────────────────
    print("\n[7/7] Topics over time …")
    if timestamps and any(pd.notnull(t) for t in timestamps):
        try:
            topics_over_time = topic_model.topics_over_time(
                docs,
                timestamps,
                global_tuning    = True,
                evolution_tuning = True,
                nr_bins          = 20,
            )
            tot_path = out(CONFIG["time_xlsx"])
            topics_over_time.to_excel(tot_path, index=False)
            print(f"  Saved → {tot_path}")

            save_fig(
                topic_model.visualize_topics_over_time(
                    topics_over_time, top_n_topics=20
                ),
                CONFIG["viz_time"],
            )
        except Exception as e:
            print(f"  WARNING: topics_over_time failed: {e}")
    else:
        print("  No valid date column — skipping topics-over-time.")

    # ────────────────────────────────────────────────────────────────────────
    # TERMINAL SUMMARY
    # ────────────────────────────────────────────────────────────────────────
    n_outliers  = sum(1 for t in topics if t == -1)
    n_topics    = len([t for t in set(topics) if t != -1])
    top5_rows   = topic_info_df[topic_info_df["Topic"] != -1].nlargest(5, "Count")

    sanity_results = {}
    for kw in CONFIG["sanity_keywords"]:
        matched = [
            tid for tid, words in all_topics.items()
            if tid != -1 and any(kw in w.lower() for w, _ in words)
        ]
        sanity_results[kw] = matched

    summary = (
        f"\n{'=' * 80}\n"
        f"BERTOPIC EXPLORATION SUMMARY\n"
        f"{'=' * 80}\n"
        f"  Total articles processed : {total_docs:,}\n"
        f"  Topics discovered        : {n_topics}\n"
        f"  Outlier documents (-1)   : {n_outliers:,}  "
        f"({n_outliers / total_docs * 100:.1f}% of corpus)\n"
        f"\nTOP 5 LARGEST TOPICS\n{'─' * 80}"
    )
    for _, row in top5_rows.iterrows():
        tid   = row["Topic"]
        cnt   = row["Count"]
        words = topic_model.get_topic(tid)
        kws   = ", ".join(w for w, _ in words[:8]) if words else "—"
        summary += f"\n  Topic {tid:>3d} ({cnt:>5,} docs): {kws}"

    summary += f"\n\n{'─' * 80}\nSANITY CHECK — macro keyword topics\n{'─' * 80}"
    for kw, tids in sanity_results.items():
        if tids:
            summary += f"\n  '{kw.upper()}' appears in topics: {tids}"
        else:
            summary += f"\n  '{kw.upper()}' — not prominent in any topic top-words"

    summary += f"\n{'=' * 80}\n"
    print(summary)

    with open(txt_path, "a", encoding="utf-8") as fh:
        fh.write("\n\n" + summary)

    print("All outputs written to:", CONFIG["out_dir"])


if __name__ == "__main__":
    main()
