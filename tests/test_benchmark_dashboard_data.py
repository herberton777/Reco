"""Data-free tests for dashboard queries, SQLite safety and protocol isolation."""
from __future__ import annotations

import json
import sqlite3

import pytest

from src.benchmark_dashboard_data import (
    DashboardDataError,
    available_metrics,
    chart_rows,
    leaderboard_rows,
    load_snapshot,
    metric_label,
    protocol_runs,
    training_history,
)
from src.benchmark_registry import connect, import_record


def _record(model_id, value, protocol="full-catalogue", source_type="measured",
            split="test", sample_users=70, recall=0.4):
    return {
        "model": {
            "id": model_id, "name": model_id,
            "family": "test family", "stage": "ranking", "maturity": "test",
        },
        "dataset": {"id": "ml-32m", "name": "MovieLens 32M", "version": "32m"},
        "protocol": {
            "id": protocol, "dataset_id": "ml-32m",
            "split_method": "global chronological test",
            "candidate_policy": "full catalog" if protocol == "full-catalogue" else "99 negative sample",
            "label_rule": "rating >= 4 observed positive",
            "cohort_signature": "test-users-A", "cutoff": "synthetic date",
        },
        "evaluation_split": split,
        "source_type": source_type,
        "source_uri": "test://fake-source",
        "sample_users": sample_users,
        "config": {"seed": 42, "lambda": 0.1},
        "metrics": {"ndcg@10": value, "recall@10": recall, "recall@200": .8},
    }


def test_empty_db_reads_catalog_but_not_invented_scores(tmp_path):
    path = tmp_path / "registry.sqlite"
    with connect(path):
        pass
    before = path.stat().st_mtime_ns
    data = load_snapshot(path)
    assert data is not None
    assert len(data["models"]) >= 10
    assert data["metrics"] == []
    assert data["runs"] == []
    assert data["training_attempts"] == []
    assert path.stat().st_mtime_ns == before


def test_missing_db_does_not_create_file(tmp_path):
    path = tmp_path / "missing.sqlite"
    assert load_snapshot(path) is None
    assert not path.exists()


def test_invalid_schema_error(tmp_path):
    path = tmp_path / "unrelated.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE unrelated (id INTEGER)")
    with pytest.raises(DashboardDataError, match="missing"):
        load_snapshot(path)


def test_protocol_split_and_literature_are_never_mixed(tmp_path):
    path = tmp_path / "bench.sqlite"
    with connect(path) as db:
        import_record(db, _record("hofm3", 0.25))
        import_record(db, _record("svd", 0.18))
        import_record(db, _record("hofm3", 0.95, protocol="sampled"))
        import_record(db, _record("hofm3", 0.65, source_type="literature"))
        import_record(db, _record("hofm3", 0.75, split="validation"))
    data = load_snapshot(path)
    test_runs = protocol_runs(data, "ml-32m", "full-catalogue", "test", "measured")
    assert len(test_runs) == 2
    assert available_metrics(data, test_runs) == [
        ("ndcg", 10), ("recall", 10), ("recall", 200)
    ]
    scores = leaderboard_rows(data, test_runs, "ndcg", 10)
    assert [r["value"] for r in scores] == [0.25, 0.18]
    assert [r["model_id"] for r in scores] == ["hofm3", "svd"]
    chart = chart_rows(data, test_runs, [("ndcg", 10), ("recall", 200)])
    assert len(chart) == 2
    assert all("NDCG@10" in row and "RECALL@200" in row for row in chart)
    assert all("Run ID" in row for row in chart)


def test_best_per_model_selects_one_actual_run_without_averaging(tmp_path):
    path = tmp_path / "bench.sqlite"
    with connect(path) as db:
        import_record(db, _record("hofm3", .31, recall=.40))
        import_record(db, _record("hofm3", .38, recall=.45))
        import_record(db, _record("svd", .21))
    data = load_snapshot(path)
    runs = protocol_runs(data, "ml-32m", "full-catalogue", "test", "measured")
    assert len(leaderboard_rows(data, runs, "ndcg", 10, best_per_model=False)) == 3
    best = leaderboard_rows(data, runs, "ndcg", 10)
    assert [r["value"] for r in best] == [.38, .21]
    assert len(best) == 2


def test_lower_is_better_metrics(tmp_path):
    path = tmp_path / "bench.sqlite"
    with connect(path) as db:
        a = _record("hofm3", .25)
        a["metrics"] = {"rmse": .83}
        b = _record("svd", .21)
        b["metrics"] = {"rmse": .76}
        import_record(db, a)
        import_record(db, b)
    data = load_snapshot(path)
    runs = protocol_runs(data, "ml-32m", "full-catalogue", "test", "measured")
    scores = leaderboard_rows(data, runs, "rmse", 0)
    assert [r["model_id"] for r in scores] == ["svd", "hofm3"]
    assert metric_label("rmse", 0) == "RMSE"


def test_training_attempts_and_links_visible(tmp_path):
    path = tmp_path / "bench.sqlite"
    with connect(path) as db:
        run_id = import_record(db, _record("hofm3_semi25_jointbcd", .27))
        db.execute(
            """INSERT INTO training_attempts
            (id, model_id, dataset_id, run_label, script_path, command_json,
             working_directory, started_at_utc, status, configuration_json)
             VALUES (?,?,?,?,?,?,?,?,?,?)""",
            ("attempt-test-1", "hofm3_semi25_jointbcd", "ml-32m",
             "test-run", "fake.py", "[]", "/tmp", "2026-10-23T00:00:00+00:00",
             "imported", json.dumps({"l2": .5})),
        )
        db.execute("INSERT INTO training_benchmarks VALUES (?,?)",
                   ("attempt-test-1", run_id))
        db.commit()
    data = load_snapshot(path)
    attempts = training_history(data, dataset_id="ml-32m")
    assert len(attempts) == 1
    assert attempts[0]["run_label"] == "test-run"
    assert attempts[0]["status"] == "imported"
    assert attempts[0]["evaluation_count"] == 1
    assert training_history(data, dataset_id="ml-1m") == []


def test_older_db_without_training_tables_still_loads(tmp_path):
    path = tmp_path / "bench.sqlite"
    with connect(path) as db:
        db.execute("DROP TABLE training_benchmarks")
        db.execute("DROP TABLE training_attempts")
        db.commit()
    data = load_snapshot(path)
    assert data["training_attempts"] == []
    assert data["training_benchmarks"] == []
