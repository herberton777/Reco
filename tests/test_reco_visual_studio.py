"""Data-free checks for the JSON-first recommender studio and aligned UMAP data."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.reco_visual_data import (
    available_json_reports, metric_frame, read_ranking_json,
    sample_umap_source, umap_projection,
)


APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_missing_models_and_database_still_show_recommender_ui(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception
    assert len(app.tabs) == 5
    assert [t.label for t in app.tabs] == [
        "Recommendations", "Choose favorites", "UMAP movie map",
        "Results (JSON)", "Methodology",
    ]
    assert not (tmp_path / "artifacts" / "benchmarks.sqlite").exists()


def test_latest_small_baseline_metrics_are_read_from_json(tmp_path):
    root = tmp_path / "artifacts"
    root.mkdir()
    report = {
        "dataset": "MovieLens latest-small",
        "k": 10,
        "validation": {"svd": {"ndcg_at_k": .11}, "hybrid": {"ndcg_at_k": .18}},
        "test": {"svd": {"ndcg_at_k": .10}, "hybrid": {"ndcg_at_k": .17}},
    }
    path = root / "metrics.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert available_json_reports(root) == [path]
    kind, loaded = read_ranking_json(path)
    assert kind == "baseline"
    frame = metric_frame(kind, loaded, "test")
    assert set(frame["Model"]) == {"svd", "hybrid"}
    assert frame.set_index("Model").loc["hybrid", "ndcg_at_k"] == .17


def test_actual_eval_report_and_reranker_summary_json(tmp_path):
    report = {
        "model": {"id": "hofm4_jointbcd", "name": "HOFM4 + JointBCD"},
        "dataset": {"id": "ml-32m"},
        "protocol": {"id": "full"},
        "evaluation_split": "test",
        "source_type": "measured",
        "metrics": {"ndcg@10": .13, "recall@10": .2},
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    kind, obj = read_ranking_json(path)
    frame = metric_frame(kind, obj)
    assert frame.loc[0, "Model"] == "HOFM4 + JointBCD"
    assert frame.loc[0, "ndcg@10"] == .13
    summary = {
        "variants": {
            "hofm3": {
                "test": {"ndcg@10": .2},
                "baseline_test": {"ndcg@10": .1},
                "validation": {"ndcg@10": .18},
            }
        }
    }
    path.write_text(json.dumps(summary), encoding="utf-8")
    kind, obj = read_ranking_json(path)
    frame = metric_frame(kind, obj, "test")
    assert len(frame) == 2
    assert set(frame["ndcg@10"]) == {.1, .2}


def test_reject_unknown_report_instead_of_inventing_scores(tmp_path):
    path = tmp_path / "incomplete.json"
    path.write_text('{"status":"completed","ndcg":0.0}', encoding="utf-8")
    with pytest.raises(ValueError, match="recognized"):
        read_ranking_json(path)


def test_full_32m_embedding_ids_and_titles_are_aligned(tmp_path):
    directory = tmp_path / "artifacts" / "semantic" / "ml32m_minilm_full_v1"
    directory.mkdir(parents=True)
    np.save(directory / "movie_ids.npy", np.array([500, 800, 1000, 2000,
                                                    3000, 4000, 5000, 6000,
                                                    7000, 8000, 9000]))
    vectors = np.arange(11 * 4, dtype="float32").reshape(11, 4)
    np.save(directory / "title_embeddings.npy", vectors)
    movie_file = tmp_path / "movies.csv"
    pd.DataFrame({
        "movieId": [800, 500, 6000],
        "title": ["Movie B", "Movie A", "Movie F"],
        "genres": ["Horror|Action", "Comedy", "Drama"],
    }).to_csv(movie_file, index=False)
    matrix, info = sample_umap_source(
        "MovieLens 32M title sentence embeddings",
        cache_dir=directory, count=11, movie_files=[movie_file],
    )
    assert np.array_equal(matrix, vectors)
    assert list(info.movieId) == [
        500, 800, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000
    ]
    assert info.loc[0, "title"] == "Movie A"
    assert info.loc[1, "title"] == "Movie B"
    assert info.loc[7, "title"] == "Movie F"
    assert info.loc[2, "title"] == ""


def test_saved_recommender_factors_are_usable_for_umap():
    movies = pd.DataFrame({
        "movieId": range(1, 14),
        "title": [f"Film {k}" for k in range(1, 14)],
        "genres": ["Comedy"] * 13,
    })
    engine = SimpleNamespace(
        movie_ids=np.arange(1, 14),
        movies=movies,
        item_latent=np.eye(13, 6, dtype="float32"),
        semantic_item=None,
    )
    matrix, info = sample_umap_source(
        "SVD item factors", engine=engine, count=10, seed=42,
    )
    assert matrix.shape == (10, 6)
    assert len(info) == 10
    assert all(info.movieId.isin(engine.movie_ids))
    with pytest.raises(ValueError, match="no saved semantic"):
        sample_umap_source(
            "Semantic vectors (saved recommender)", engine=engine,
        )


def test_umap_is_optional_and_projections_preserve_row_alignment(monkeypatch):
    import sys
    class FakeUmap:
        def __init__(self, **kwargs):
            assert kwargs["metric"] == "cosine"
        def fit_transform(self, matrix):
            return np.asarray(matrix[:, :2], dtype="float32")
    monkeypatch.setitem(sys.modules, "umap", SimpleNamespace(UMAP=FakeUmap))
    info = pd.DataFrame({
        "movieId": range(10), "title": ["X"] * 10,
        "genres": ["Comedy|Drama"] * 10,
    })
    result = umap_projection(np.arange(40).reshape(10, 4), info)
    assert result["Primary genre"].unique().tolist() == ["Comedy"]
    assert np.array_equal(result["UMAP 1"].to_numpy(), np.arange(0, 40, 4))
    assert np.array_equal(result["UMAP 2"].to_numpy(), np.arange(1, 40, 4))
