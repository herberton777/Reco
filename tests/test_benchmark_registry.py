"""Data-free regression tests for the benchmark registry."""
import json
import sqlite3

import pytest

from src.benchmark_registry import (
    BASELINE_PROTOCOL,
    connect,
    export_records,
    import_baseline,
    import_record,
    leaderboard,
)


def _sample_record():
    return {
        "model": {
            "id": "hofm3_semi25_jointbcd", "name": "HOFM3 Semi25 Joint BCD",
            "family": "higher-order factorization", "stage": "ranking",
            "maturity": "research proposal",
        },
        "dataset": {"id": "ml-32m", "name": "MovieLens 32M", "version": "32m"},
        "protocol": {
            "id": "ml32m-test-protocol", "dataset_id": "ml-32m",
            "split_method": "global chronological holdout",
            "candidate_policy": "full catalogue excluding train interactions",
            "label_rule": "ratings >=4 are observed positives",
            "cohort_signature": "synthetic-test-cohort-v1",
            "cutoff": "synthetic-test-cutoff",
        },
        "evaluation_split": "test",
        "source_type": "measured",
        "source_uri": "synthetic-test-fixture.json",
        "sample_users": 20,
        "config": {"regularization": 0.1},
        "metrics": {"ndcg@10": 0.32, "recall@10": 0.4, "recall@200": 0.7},
    }


def test_import_idempotent_and_metrics(tmp_path):
    with connect(tmp_path / "bench.sqlite") as db:
        payload = _sample_record()
        run_id = import_record(db, payload)
        assert import_record(db, payload) == run_id
        assert db.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM metrics").fetchone()[0] == 3
        rows = leaderboard(db, "ml-32m", "ml32m-test-protocol", "test", "ndcg", 10)
        assert len(rows) == 1
        assert rows[0]["value"] == 0.32
        assert rows[0]["sample_users"] == 20
        assert export_records(db)[0]["config"] == {"regularization": 0.1}


def test_protocols_keep_sampling_and_full_catalogue_separate(tmp_path):
    with connect(tmp_path / "bench.sqlite") as db:
        p = _sample_record()
        import_record(db, p)
        sampled = _sample_record()
        sampled["protocol"] = {**sampled["protocol"], "id": "sampled-negatives-test",
                               "candidate_policy": "1 positive + 99 sampled unknowns"}
        sampled["metrics"] = {"ndcg@10": 0.9}
        import_record(db, sampled)
        assert len(leaderboard(db, "ml-32m", "ml32m-test-protocol")) == 1
        assert len(leaderboard(db, "ml-32m", "sampled-negatives-test")) == 1
        assert leaderboard(db, "ml-32m", "ml32m-test-protocol")[0]["value"] == 0.32
        with pytest.raises(ValueError, match="Protocol ID reused"):
            conflict = _sample_record()
            conflict["protocol"]["candidate_policy"] = "different protocol, same ID"
            import_record(db, conflict)


def test_literature_source_is_not_mixed_into_measured_results(tmp_path):
    with connect(tmp_path / "bench.sqlite") as db:
        p = _sample_record()
        import_record(db, p)
        literature = _sample_record()
        literature["source_type"] = "literature"
        literature["source_uri"] = "https://example.org/research-paper"
        literature["metrics"] = {"ndcg@10": 0.5}
        import_record(db, literature)
        assert len(leaderboard(db, "ml-32m", "ml32m-test-protocol")) == 1
        assert len(leaderboard(db, "ml-32m", "ml32m-test-protocol",
                               source_type="literature")) == 1


@pytest.mark.parametrize("bad_metric", [
    {"ndcg@10": 1.1},
    {"recall@0": 0.2},
    {"ndcg@10": float("nan")},
    {"ndcg@10": "0.3"},
])
def test_invalid_metric_rejected(tmp_path, bad_metric):
    with connect(tmp_path / "bench.sqlite") as db:
        p = _sample_record()
        p["metrics"] = bad_metric
        with pytest.raises(ValueError):
            import_record(db, p)
        assert db.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0


def test_import_existing_baseline_metrics_json(tmp_path):
    fixture = {
        "dataset": "MovieLens latest-small (GroupLens)", "k": 10,
        "timestamp_utc": "2026-10-23T00:00:00+00:00",
        "positive_rating_threshold": 4,
        "split": "per-user last 3 positive events: rank-train / valid / test",
        "counts": {"users_test": 100}, "catalog_size": 2500,
        "validation": {"popularity": {"users_evaluated": 100, "ndcg_at_k": 0.1,
                                      "recall_at_k": 0.2, "catalog_coverage_at_k": 0.3}},
        "test": {"popularity": {"users_evaluated": 100, "ndcg_at_k": 0.12,
                               "recall_at_k": 0.22, "catalog_coverage_at_k": 0.33}},
    }
    f = tmp_path / "metrics.json"
    f.write_text(json.dumps(fixture), encoding="utf-8")
    with connect(tmp_path / "bench.sqlite") as db:
        assert len(import_baseline(db, f)) == 2
        assert len(import_baseline(db, f)) == 2
        assert db.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 2
        assert leaderboard(db, "ml-latest-small", BASELINE_PROTOCOL["id"])[0]["value"] == 0.12
        assert leaderboard(db, "ml-latest-small", BASELINE_PROTOCOL["id"], "validation")[0]["value"] == 0.1


def test_db_enforces_foreign_keys(tmp_path):
    with connect(tmp_path / "bench.sqlite") as db:
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("INSERT INTO metrics (run_id,metric,k,value) VALUES ('missing','ndcg',10,.5)")
