"""Data-free integration tests for tracked HOFM experiment attempts."""
import json
from pathlib import Path
import sys

import pytest

from src.benchmark_registry import connect, leaderboard
from src.hofm_experiments import attach_evaluation, list_attempts, register_existing, run_trainer


def _fake_trainer(path, exit_code=0):
    path.write_text("import sys\nsys.exit(%d)\n" % exit_code, encoding="utf-8")
    return path


def _evaluation():
    return {
        "model": {
            "id": "hofm3_semi25_jointbcd", "name": "HOFM3 Semi25 Joint BCD",
            "family": "higher-order factorization", "stage": "ranking",
            "maturity": "research proposal",
        },
        "dataset": {"id": "ml-32m", "name": "MovieLens 32M", "version": "32m"},
        "protocol": {
            "id": "synthetic-32m-full-v1", "dataset_id": "ml-32m",
            "split_method": "global chronological",
            "candidate_policy": "full catalogue excluding training history",
            "label_rule": "ratings >=4 are observed positives",
            "cohort_signature": "synthetic-cohort-A", "cutoff": "synthetic-cutoff",
        },
        "evaluation_split": "test",
        "source_type": "measured", "source_uri": "replaced-by-attach",
        "sample_users": 11, "config": {"k": 10, "negative_share": 0.25},
        "metrics": {"ndcg@10": 0.29, "recall@10": 0.36},
    }


def test_successful_training_has_no_benchmark_scores(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    script = _fake_trainer(tmp_path / "fake.py")
    with connect(tmp_path / "bench.sqlite") as db:
        assert run_trainer(db, script, ["--dataset", "ml-32m",
                                        "--run-label", "hofm_reg_fixture"]) == 0
        row = list_attempts(db)[0]
        assert row["status"] == "completed"
        assert row["evaluation_count"] == 0
        assert db.execute("SELECT count(*) FROM runs").fetchone()[0] == 0
        details = db.execute(
            "SELECT command_json, configuration_json FROM training_attempts WHERE id=?",
            (row["id"],)).fetchone()
        assert "--run-label" in json.loads(details["command_json"])
        assert json.loads(details["configuration_json"])["script_sha256"]


def test_failures_are_tracked_without_metrics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    script = _fake_trainer(tmp_path / "failed.py", 7)
    with connect(tmp_path / "bench.sqlite") as db:
        assert run_trainer(db, script, ["--run-label=failed_fixture"]) == 7
        assert list_attempts(db)[0]["status"] == "failed"
        assert db.execute("SELECT count(*) FROM metrics").fetchone()[0] == 0


def test_attach_measured_provenance_and_idempotency(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    script = _fake_trainer(tmp_path / "fake.py")
    report = tmp_path / "actual_evaluation.json"
    report.write_text(json.dumps(_evaluation()), encoding="utf-8")
    with connect(tmp_path / "bench.sqlite") as db:
        assert run_trainer(db, script, ["--run-label", "fixture"]) == 0
        benchmark_id = attach_evaluation(db, report, label="fixture")
        assert attach_evaluation(db, report, label="fixture") == benchmark_id
        assert list_attempts(db)[0]["evaluation_count"] == 1
        rows = leaderboard(db, "ml-32m", "synthetic-32m-full-v1")
        assert len(rows) == 1 and rows[0]["value"] == 0.29
        metric = db.execute("SELECT source_sha256 FROM runs WHERE id=?",
                            (benchmark_id,)).fetchone()
        assert metric["source_sha256"]
        assert db.execute("SELECT count(*) FROM metrics").fetchone()[0] == 2


def test_cannot_link_literature_or_wrong_dataset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    script = _fake_trainer(tmp_path / "fake.py")
    report = tmp_path / "report.json"
    with connect(tmp_path / "bench.sqlite") as db:
        run_trainer(db, script, ["--run-label", "fixture"])
        literature = _evaluation()
        literature["source_type"] = "literature"
        report.write_text(json.dumps(literature), encoding="utf-8")
        with pytest.raises(ValueError, match="measured"):
            attach_evaluation(db, report, label="fixture")
        bad = _evaluation()
        bad["dataset"] = {"id": "ml-1m", "name": "MovieLens 1M", "version": "1m"}
        bad["protocol"]["dataset_id"] = "ml-1m"
        report.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(ValueError, match="dataset"):
            attach_evaluation(db, report, label="fixture")
        assert db.execute("SELECT count(*) FROM runs").fetchone()[0] == 0


def test_refuse_missing_or_mismatched_run_labels(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    script = _fake_trainer(tmp_path / "fake.py")
    with connect(tmp_path / "bench.sqlite") as db:
        with pytest.raises(ValueError, match="run-label"):
            run_trainer(db, script, ["--epochs", "2"])
        with pytest.raises(ValueError, match="Dataset mismatch"):
            run_trainer(db, script, ["--run-label", "fixture", "--dataset", "ml-1m"])
        assert list_attempts(db) == []


def test_existing_regularization_config_is_recorded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    script = _fake_trainer(tmp_path / "fake.py")
    checkpoint = (tmp_path / "models/ml-32m/unified_hofm3_bcd/"
                  "fm_all_hofm3semi25_bpr_fixture")
    checkpoint.mkdir(parents=True)
    (checkpoint / "regularization_config.json").write_text(
        json.dumps({"bcd_dropout": 0.15}), encoding="utf-8")
    with connect(tmp_path / "bench.sqlite") as db:
        run_trainer(db, script, ["--run-label", "fixture"])
        payload = db.execute("SELECT configuration_json FROM training_attempts").fetchone()[0]
        assert json.loads(payload)["regularization"]["bcd_dropout"] == 0.15

def test_import_historical_checkpoint_without_guessing_training_history(tmp_path):
    folder = tmp_path / "models/ml-32m/unified_hofm3_bcd/parent_old"
    folder.mkdir(parents=True)
    (folder / "regularization_config.json").write_text(
        json.dumps({"factor_l2": 0.0025}), encoding="utf-8")
    report = tmp_path / "report.json"
    report.write_text(json.dumps(_evaluation()), encoding="utf-8")
    with connect(tmp_path / "bench.sqlite") as db:
        first = register_existing(db, folder, "old")
        assert first == register_existing(db, folder, "old")
        attempt = db.execute("SELECT * FROM training_attempts WHERE id=?", (first,)).fetchone()
        assert attempt["status"] == "imported"
        assert attempt["git_commit"] is None
        assert attempt["exit_code"] is None
        assert list_attempts(db)[0]["evaluation_count"] == 0
        assert attach_evaluation(db, report, attempt_id=first)
        assert list_attempts(db)[0]["evaluation_count"] == 1


def test_empty_historical_folder_refused(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with connect(tmp_path / "bench.sqlite") as db:
        with pytest.raises(ValueError, match="weights.pt"):
            register_existing(db, empty, "old")
