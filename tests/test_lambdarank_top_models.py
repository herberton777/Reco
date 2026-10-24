"""Offline checks for top-model MovieLens 32M LambdaRank experiments."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import lambdarank_rerank as lr
from src.lambdarank_top_models import (
    _normalized_fusion, _per_query_ndcg, audit, bootstrap_delta,
    run, select_models,
)


def _inputs(tmp_path, query_counts=(12, 5, 5), size=6):
    paths = []
    for split, nq in zip(("train", "valid", "test"), query_counts):
        rows = []
        for query in range(nq):
            for movie in range(size):
                label = int(movie == (query % size))
                # Four distinct strengths; all positive labels are present.
                rows.append({
                    "query_id": f"{split}_{query:04d}",
                    "item_id": str(movie),
                    "label": label,
                    "total_relevant": 1,
                    "score_hofm3_semi25_jointbcd": (
                        3. if label else -float(movie + query % 3) / 10.
                    ),
                    "score_hofm4_jointbcd": (
                        3.2 if label else -float((movie + query) % 3) / 10.
                    ),
                    "score_bpr": (
                        2. if label else -float((movie + query + 1) % 4) / 10.
                    ),
                    "score_semantic": float(movie) / 10,
                    "feature_historical_popularity": float(movie) / size,
                })
        path = tmp_path / f"{split}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        paths.append(path)
    manifest = {
        "dataset": {"id": "ml-32m", "name": "MovieLens 32M", "version": "32m"},
        "protocol": {
            "id": "SYNTHETIC_FIXTURE_NOT_REAL_MOVIELENS",
            "dataset_id": "ml-32m",
            "split_method": "synthetic query splits for testing",
            "candidate_policy": f"{size} artificial candidates, all positives",
            "label_rule": "synthetic labels",
            "cohort_signature": "synthetic-fixture-not-real",
            "cutoff": "synthetic",
        },
        "features_from_past_only": True,
        "frozen_base_models": True,
        "test_untouched_for_training": True,
        "query_id_is_user": False,
    }
    file = tmp_path / "manifest.json"
    file.write_text(json.dumps(manifest), encoding="utf-8")
    return [*paths, file]


def test_select_top_models_by_validation_only(tmp_path):
    train, valid_file, test, manifest = _inputs(tmp_path)
    chosen, scores = select_models(
        lr.load_candidates(valid_file), top=3, k=3,
    )
    assert len(chosen) == 3
    assert chosen[0] == "hofm4_jointbcd"  # same score as HOFM3, priority tie
    assert set(chosen) <= set(scores)
    assert 0 <= scores["semantic"] <= 1
    names, _ = select_models(
        lr.load_candidates(valid_file),
        names="bpr,hofm4_jointbcd", top=2, k=3,
    )
    assert names == ["bpr", "hofm4_jointbcd"]
    with pytest.raises(ValueError, match="No frozen candidate scores"):
        select_models(lr.load_candidates(valid_file),
                      names="hofm3,unknown", k=3)


def test_audit_is_read_only_and_rejects_missing_exports(tmp_path):
    paths = _inputs(tmp_path)
    info, train, valid, test = audit(*paths, top=4, k=3)
    assert info["query_counts"] == {"train": 12, "valid": 5, "test": 5}
    assert info["candidate_rows"]["train"] == 72
    assert "input_sha256" in info
    assert len(info["selected_models"]) == 4
    assert not (tmp_path / "benchmarks.sqlite").exists()
    with pytest.raises(FileNotFoundError, match="Missing input exports"):
        audit(tmp_path / "not_here.csv", *paths[1:], k=3)


def test_audit_rejects_query_overlap_and_score_schema_mismatch(tmp_path):
    files = _inputs(tmp_path)
    data = pd.read_csv(files[1])
    data.loc[data["query_id"] == "valid_0000", "query_id"] = "train_0000"
    data.to_csv(files[1], index=False)
    with pytest.raises(ValueError, match="overlap"):
        audit(*files, k=3)
    files = _inputs(tmp_path)
    data = pd.read_csv(files[2])
    data = data.drop(columns=["score_bpr"])
    data.to_csv(files[2], index=False)
    with pytest.raises(ValueError, match="schema"):
        audit(*files, k=3)


def test_fusions_are_per_query_and_not_ground_truth_leaks(tmp_path):
    files = _inputs(tmp_path)
    val = lr.load_candidates(files[1])
    cols = ["score_bpr", "score_semantic"]
    z = _normalized_fusion(val, cols, method="zmean")
    rrf = _normalized_fusion(val, cols, method="rrf")
    assert len(z) == len(val) == len(rrf)
    assert np.isfinite(z).all() and np.isfinite(rrf).all()
    assert np.allclose(z.reshape(5, 6).mean(axis=1), 0.0)
    # Labels are NEVER an input. This transformation must not change scores.
    altered = val.copy()
    altered["label"] = 1 - altered["label"]
    assert np.allclose(z, _normalized_fusion(altered, cols, method="zmean"))


def test_bootstrap_identical_predictions_have_zero_delta(tmp_path):
    files = _inputs(tmp_path)
    val = lr.load_candidates(files[1])
    pred = val["score_hofm3_semi25_jointbcd"].to_numpy()
    ndcg = _per_query_ndcg(val, pred, k=3)
    assert len(ndcg) == 5 and np.all(ndcg >= 0)
    result = bootstrap_delta(val, pred, pred, 3, repeats=120, seed=42)
    assert result["delta_ndcg"] == 0.0
    assert result["paired_query_bootstrap_ci_95"] == [0.0, 0.0]


def test_top_models_end_to_end_train_writes_genuine_json(tmp_path):
    files = _inputs(tmp_path, query_counts=(15, 6, 6), size=7)
    from argparse import Namespace
    args = Namespace(
        train=str(files[0]), valid=str(files[1]), test=str(files[2]),
        manifest=str(files[3]),
        models="hofm3_semi25_jointbcd,hofm4_jointbcd,bpr",
        top=3, k=3,
        output=str(tmp_path / "experiment"),
        trees=20, early_stop=5, lr=.08, leaves=7, min_leaf=2,
        l1=.0, l2=1., seed=42, threads=2, bootstrap=120, dry_run=True,
    )
    assert run(args) is None
    assert not (tmp_path / "experiment").exists()
    args.dry_run = False
    output = run(args)
    assert output["status"] == "COMPLETED_WITH_MEASURED_SCORES"
    assert "score_hofm3_semi25_jointbcd" in output["feature_importances"]
    assert len(output["variants"]) == 9  # 3 frozen + 2 fusions + 3 single LTR + ensemble
    assert (tmp_path / "experiment" / "summary.json").is_file()
    assert (tmp_path / "experiment" / "top_models_analysis.json").is_file()
    assert (tmp_path / "experiment" / "report_lambdarank_ensemble.json").is_file()
    assert (tmp_path / "experiment" / "predictions_ensemble.csv").is_file()
    assert not (tmp_path / "benchmarks.sqlite").exists()
    assert output["selected_champion_on_validation"] in {
        row["variant"] for row in output["variants"]
    }
    with pytest.raises(FileExistsError, match="Previous experiment exists"):
        run(args)


def test_top_model_analysis_is_visible_in_json_first_studio(tmp_path):
    from src.reco_visual_data import read_ranking_json, metric_frame
    from src.reco_studio_data import discover_reports, comparable_records
    folder = tmp_path / "artifacts" / "rerank" / "top_models_v1"
    folder.mkdir(parents=True)
    report = {
        "status": "COMPLETED_WITH_MEASURED_SCORES",
        "audit": {
            "dataset": {"id": "ml-32m"},
            "protocol": {"id": "synthetic-fixed-candidates"},
        },
        "variants": [
            {"variant": "frozen_hofm3", "validation": {"ndcg@10": .15},
             "test": {"ndcg@10": .13}},
            {"variant": "lambdarank_ensemble", "validation": {"ndcg@10": .17},
             "test": {"ndcg@10": .14}},
            {"variant": "zmean", "validation": {"ndcg@10": .16},
             "test": {"ndcg@10": .135}},
        ],
    }
    filename = folder / "top_models_analysis.json"
    filename.write_text(json.dumps(report), encoding="utf-8")
    kind, loaded = read_ranking_json(filename)
    assert kind == "top_models"
    visible = metric_frame(kind, loaded, "test")
    assert list(visible.Model) == [
        "frozen_hofm3", "lambdarank_ensemble", "zmean",
    ]
    found = discover_reports(tmp_path / "artifacts")
    rows = comparable_records(found)
    assert len(rows[(rows["Metric"] == "ndcg@10") & (rows["Split"] == "test")]) == 3
    assert rows["Scope"].nunique() == 2  # one per validation/test split
