"""Data-free safety and integration tests for multi-model LambdaRank."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.benchmark_registry import connect
from src.lambdarank_rerank import (
    ensure_split_integrity, experiment, join_scores, load_candidates, ranking_metrics,
)


def _frame(query_prefix, n_queries=8, n_candidates=6):
    records = []
    for q in range(n_queries):
        for i in range(n_candidates):
            positive = int(i == (q % n_candidates))
            records.append({
                "query_id": f"{query_prefix}-{q}",
                "item_id": f"{i}",
                "label": positive,
                "score_hofm3": 3 * positive + 0.1 * (i % 3) + q * 0.01,
                "score_bpr": positive + 0.2 * i,
                "feature_historical_popularity": i / n_candidates,
            })
    return pd.DataFrame(records)


def _manifest():
    return {
        "dataset": {"id": "ml-32m", "name": "MovieLens 32M", "version": "32m"},
        "protocol": {
            "id": "synthetic-evaluation-protocol-v1", "dataset_id": "ml-32m",
            "split_method": "synthetic distinct temporal queries",
            "candidate_policy": "6 synthetic candidate items per query; not full catalogue",
            "label_rule": "binary observed positive",
            "cohort_signature": "synthetic-holdout-sha", "cutoff": "synthetic-cutoff",
        },
        "features_from_past_only": True,
        "frozen_base_models": True,
        "test_untouched_for_training": True,
    }


def test_query_metrics_exact_with_multiple_positives():
    frame = pd.DataFrame({
        "query_id": ["a", "a", "a", "b", "b"],
        "item_id": ["1", "2", "3", "4", "5"],
        "label": [1, 1, 0, 1, 0],
        "score_hofm3": [0.3, 0.2, 0.1, 0.2, 0.1],
    })
    frame = load_file_via_csv(frame)
    # perfect candidate ranking for each query
    scores = ranking_metrics(frame, frame["score_hofm3"], k=2)
    assert scores["ndcg@2"] == pytest.approx(1.0)
    assert scores["recall@2"] == pytest.approx(1.0)
    assert scores["queries_evaluated"] == 2


def load_file_via_csv(frame, tmp_path=None):
    # the loader sorts identically to ranking_metrics assumptions
    return frame.assign(query_id=frame.query_id.astype(str),
                        item_id=frame.item_id.astype(str)).sort_values(
                        ["query_id", "item_id"], kind="stable").reset_index(drop=True)


def test_source_join_requires_identical_candidates(tmp_path):
    full = _frame("train", n_queries=2)
    labeled = full[["query_id", "item_id", "label",
                    "feature_historical_popularity"]]
    anchor = tmp_path / "base.csv"
    labeled.to_csv(anchor, index=False)
    source1 = tmp_path / "hofm3.csv"
    source2 = tmp_path / "bpr.csv"
    full[["query_id", "item_id", "score_hofm3"]].rename(
        columns={"score_hofm3": "score"}).to_csv(source1, index=False)
    full[["query_id", "item_id", "score_bpr"]].rename(
        columns={"score_bpr": "score"}).to_csv(source2, index=False)
    destination = tmp_path / "joined.csv"
    join_scores(anchor, [f"hofm3={source1}", f"bpr={source2}"], destination)
    result = load_candidates(destination)
    assert len(result) == len(full)
    assert "score_hofm3" in result and "score_bpr" in result
    missing = pd.read_csv(source2).iloc[:-1]
    missing.to_csv(source2, index=False)
    with pytest.raises(ValueError, match="same candidates"):
        join_scores(anchor, [f"hofm3={source1}", f"bpr={source2}"], destination)


def test_overlap_rejected():
    train, valid, test = (_frame("t"), _frame("v"), _frame("t"))
    with pytest.raises(ValueError, match="overlap"):
        ensure_split_integrity(train, valid, test)


def test_leakage_columns_rejected(tmp_path):
    frame = _frame("train")
    frame["future_user_rating"] = 5
    file = tmp_path / "bad.csv"
    frame.to_csv(file, index=False)
    with pytest.raises(ValueError, match="Unsupported columns"):
        load_candidates(file)


def test_no_missing_score_or_degenerate_queries(tmp_path):
    frame = _frame("train")
    frame.loc[0, "score_bpr"] = np.nan
    file = tmp_path / "bad.csv"
    frame.to_csv(file, index=False)
    with pytest.raises(ValueError, match="Non-finite"):
        load_candidates(file)
    frame.loc[0, "score_bpr"] = 0.2
    frame.loc[frame.query_id == "train-0", "label"] = 0
    frame.to_csv(file, index=False)
    with pytest.raises(ValueError, match="positive"):
        load_candidates(file)


def test_multiple_models_ensemble_and_registry(tmp_path):
    frames = {
        "train": _frame("train", n_queries=12, n_candidates=8),
        "valid": _frame("valid", n_queries=5, n_candidates=8),
        "test": _frame("test", n_queries=5, n_candidates=8),
    }
    for split, data in frames.items():
        data.to_csv(tmp_path / f"{split}.csv", index=False)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest()), encoding="utf-8")
    from argparse import Namespace
    dbfile = tmp_path / "benchmarks.sqlite"
    params = Namespace(
        train=str(tmp_path / "train.csv"), valid=str(tmp_path / "valid.csv"),
        test=str(tmp_path / "test.csv"), manifest=str(manifest),
        models="hofm3,bpr", ensemble=True, k=3, trees=25, early_stop=5,
        lr=.08, leaves=7, min_leaf=3, l1=0.0, l2=2.0, seed=42, threads=1,
        output=str(tmp_path / "out"), db=str(dbfile),
    )
    summary = experiment(params)
    assert set(summary["variants"]) == {"hofm3", "bpr", "ensemble"}
    for variant in ("hofm3", "bpr"):
        actual = summary["variants"][variant]
        assert "ndcg@3" in actual["test"]
        assert "ndcg@3" in actual["baseline_test"]
    assert (tmp_path / "out" / "lambdarank_ensemble.txt").is_file()
    assert (tmp_path / "out" / "report_lambdarank_hofm3.json").is_file()
    assert (tmp_path / "out" / "predictions_hofm3.csv").is_file()
    with connect(dbfile) as db:
        assert db.execute("SELECT count(*) FROM runs").fetchone()[0] == 5
        assert db.execute("SELECT count(*) FROM metrics WHERE metric='ndcg' AND k=3"
                          ).fetchone()[0] == 5
    # Reusing a run directory must fail rather than silently overwrite results.
    with pytest.raises(FileExistsError):
        experiment(params)


def test_manifest_provenance_required(tmp_path):
    frames = {split: _frame(split) for split in ("train", "valid", "test")}
    for split, frame in frames.items():
        frame.to_csv(tmp_path / (split + ".csv"), index=False)
    m = _manifest()
    m["test_untouched_for_training"] = False
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(m), encoding="utf-8")
    from argparse import Namespace
    p = Namespace(
        train=str(tmp_path / "train.csv"), valid=str(tmp_path / "valid.csv"),
        test=str(tmp_path / "test.csv"), manifest=str(manifest),
        models="hofm3", ensemble=False, k=10, trees=5, early_stop=2,
        lr=.1, leaves=3, min_leaf=2, l1=0., l2=1., seed=42, threads=1,
        output=str(tmp_path / "output"), db=None,
    )
    with pytest.raises(ValueError, match="test_untouched"):
        experiment(p)

def test_evaluation_counts_missing_retrieved_positive_as_zero(tmp_path):
    """A retrieval miss must NOT be removed from evaluation query cohort."""
    frame = pd.DataFrame({
        "query_id": ["hit", "hit", "miss", "miss"],
        "item_id": ["a", "b", "a", "b"],
        "label": [1, 0, 0, 0],
        "total_relevant": [1, 1, 1, 1],
        "score_hofm3": [2.0, 1.0, 2.0, 1.0],
    })
    file = tmp_path / "retrieval.csv"
    frame.to_csv(file, index=False)
    with pytest.raises(ValueError, match="Training queries"):
        load_candidates(file)
    loaded = load_candidates(file, allow_missing_positive=True)
    measured = ranking_metrics(loaded, loaded["score_hofm3"].to_numpy(), k=1)
    assert measured["queries_evaluated"] == 2
    assert measured["ndcg@1"] == pytest.approx(0.5)
    assert measured["recall@1"] == pytest.approx(0.5)
    assert measured["hit_rate@1"] == pytest.approx(0.5)
    bad = frame.drop(columns=["total_relevant"])
    bad.to_csv(file, index=False)
    with pytest.raises(ValueError, match="total_relevant"):
        load_candidates(file, allow_missing_positive=True)


def test_total_relevant_consistent_with_labels(tmp_path):
    frame = _frame("valid", n_queries=3)
    frame["total_relevant"] = 1
    frame.loc[0, "total_relevant"] = 2
    file = tmp_path / "inconsistent.csv"
    frame.to_csv(file, index=False)
    with pytest.raises(ValueError, match="constant"):
        load_candidates(file, allow_missing_positive=True)
