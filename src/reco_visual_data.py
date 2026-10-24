"""JSON-first recommender studio: metric reports and UMAP projection helpers.

This module does not open, create or change SQLite benchmark databases.
It is data-free/testable without installing UMAP or downloading MovieLens.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


MAX_REPORT_BYTES = 8_000_000


def available_json_reports(root="artifacts", maximum=200):
    """Find local, human-readable ranking reports without crawling checkpoints."""
    root = Path(root)
    if not root.is_dir():
        return []
    paths = set()
    patterns = (
        "metrics.json",
        "*metrics*.json",
        "reports/*.json",
        "benchmarks/*.json",
        "rerank/*/report*.json",
        "rerank/*/summary.json",
        "training_logs/*.json",
    )
    for pattern in patterns:
        paths.update(path for path in root.glob(pattern)
                     if path.is_file() and path.stat().st_size <= MAX_REPORT_BYTES)
    return sorted(paths, key=lambda path: str(path))[:maximum]


def read_ranking_json(path):
    """Only accept explicit baseline or measured-report schemas; never invent scores."""
    path = Path(path)
    if path.stat().st_size > MAX_REPORT_BYTES:
        raise ValueError("Report JSON is unexpectedly large")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Expected JSON report object")
    if (isinstance(payload.get("test"), dict)
            and isinstance(payload.get("validation"), dict)):
        return "baseline", payload
    if isinstance(payload.get("metrics"), dict):
        return "evaluation", payload
    if isinstance(payload.get("variants"), dict):
        return "rerank_summary", payload
    raise ValueError("JSON does not contain a recognized measured ranking report")


def metric_frame(kind, payload, split="test"):
    """Convert existing report formats into a readable numeric comparison table."""
    if kind == "baseline":
        values = payload.get(split)
        if not isinstance(values, dict):
            raise ValueError("Missing " + split + " metrics")
        frame = pd.DataFrame.from_dict(values, orient="index")
        frame.index.name = "Model"
        return frame.reset_index()
    if kind == "evaluation":
        model = payload.get("model", {})
        name = (model.get("name", model.get("id", "Model"))
                if isinstance(model, dict) else str(model))
        return pd.DataFrame([{"Model": name, **payload["metrics"]}])
    if kind == "rerank_summary":
        rows = []
        for model_name, report in payload["variants"].items():
            if not isinstance(report, dict):
                continue
            candidate = report.get(split)
            if isinstance(candidate, dict):
                rows.append({"Model": "LambdaRank " + str(model_name), **candidate})
            if split == "test" and isinstance(report.get("baseline_test"), dict):
                rows.append({
                    "Model": str(model_name) + " (frozen base)",
                    **report["baseline_test"],
                })
        return pd.DataFrame(rows)
    raise ValueError("Unsupported JSON report kind: " + str(kind))


def _movie_rows_for_ids(ids, candidates):
    """Lookup titles/genres by exact MovieLens movie IDs."""
    ids = np.asarray(ids, dtype="int64")
    for filename in candidates:
        file = Path(filename)
        if not file.is_file():
            continue
        movies = pd.read_csv(file, usecols=["movieId", "title", "genres"])
        lookup = movies.drop_duplicates("movieId").set_index("movieId")
        rows = lookup.reindex(ids)
        return pd.DataFrame({
            "movieId": ids,
            "title": rows["title"].fillna("").to_numpy(),
            "genres": rows["genres"].fillna("").to_numpy(),
        })
    return pd.DataFrame({
        "movieId": ids,
        "title": [f"Movie ID {mid}" for mid in ids],
        "genres": ["Unknown"] * len(ids),
    })


def sample_umap_source(source, *, engine=None, cache_dir=None, count=2000,
                       seed=42, movie_files=None):
    """Return a strictly aligned movie/embedding subset; no implicit retraining.

    Sources:
      - 'SVD item factors' / 'Semantic vectors (saved recommender)'
      - 'MovieLens 32M title sentence embeddings' (memory mapped)
    """
    if count < 10 or count > 10000:
        raise ValueError("UMAP sample must contain 10–10000 movies")
    if source in ("SVD item factors", "Semantic vectors (saved recommender)"):
        if engine is None:
            raise ValueError("A trusted trained recommender.joblib is required")
        raw = (engine.item_latent if source == "SVD item factors"
               else engine.semantic_item)
        if raw is None:
            raise ValueError("This recommender has no saved semantic vectors")
        ids = np.asarray(engine.movie_ids, dtype="int64")
        movies = engine.movies[["movieId", "title", "genres"]].copy()
        lookup = movies.drop_duplicates("movieId").set_index("movieId")
    elif source == "MovieLens 32M title sentence embeddings":
        cache_dir = Path(cache_dir or "artifacts/semantic/ml32m_minilm_full_v1")
        ids_file = cache_dir / "movie_ids.npy"
        vector_file = cache_dir / "title_embeddings.npy"
        if not ids_file.is_file() or not vector_file.is_file():
            raise FileNotFoundError("Run src.encode_ml32m_text encode to create movie vectors")
        ids = np.load(ids_file, mmap_mode="r", allow_pickle=False)
        raw = np.load(vector_file, mmap_mode="r", allow_pickle=False)
        movies = None
        lookup = None
    else:
        raise ValueError("Unknown UMAP embedding source")

    if len(ids) != len(raw) or getattr(raw, "ndim", None) != 2:
        raise ValueError("Movie IDs are not aligned to the embedding matrix")
    if not np.issubdtype(raw.dtype, np.number):
        raise ValueError("Expected numeric movie embeddings")
    if len(ids) < 10:
        raise ValueError("At least ten movies required for UMAP")
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(len(ids), size=min(count, len(ids)), replace=False))
    matrix = np.asarray(raw[rows], dtype="float32")
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite embedding vector found")
    selected_ids = np.asarray(ids[rows], dtype="int64")
    if lookup is not None:
        selected = lookup.reindex(selected_ids)
        info = pd.DataFrame({
            "movieId": selected_ids,
            "title": selected["title"].fillna("").to_numpy(),
            "genres": selected["genres"].fillna("").to_numpy(),
        })
    else:
        info = _movie_rows_for_ids(selected_ids, movie_files or (
            "data/raw/ml-32m/movies.csv", "data/temporal/ml-32m/movies.csv",
        ))
    return matrix, info


def umap_projection(matrix, info, *, n_neighbors=20, min_dist=0.1, seed=42):
    """Compute an actual UMAP projection, returning film metadata and coordinates."""
    try:
        import umap
    except ImportError as exc:
        raise RuntimeError(
            "UMAP is not installed. Run: python -m pip install -r requirements-umap.txt"
        ) from exc
    if len(matrix) != len(info) or len(matrix) < 10:
        raise ValueError("Invalid UMAP data alignment")
    if n_neighbors < 2 or not (0 <= min_dist <= 1):
        raise ValueError("Invalid UMAP hyperparameters")
    reducer = umap.UMAP(
        n_components=2, n_neighbors=min(int(n_neighbors), len(matrix) - 1),
        min_dist=float(min_dist), metric="cosine", random_state=int(seed),
        n_jobs=1,
    )
    positions = reducer.fit_transform(matrix)
    output = info.reset_index(drop=True).copy()
    output["UMAP 1"] = positions[:, 0]
    output["UMAP 2"] = positions[:, 1]
    output["Primary genre"] = output["genres"].fillna("").astype(str).str.split("|").str[0]
    output["Primary genre"] = output["Primary genre"].replace({"": "Unknown", "(no genres listed)": "Unknown"})
    return output
