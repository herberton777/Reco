"""Streamlit smoke tests with no MovieLens data or model checkpoints."""
from __future__ import annotations

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.benchmark_registry import connect, import_record


DASHBOARD = str(Path(__file__).resolve().parents[1] / "benchmark_dashboard.py")


def test_missing_database_explains_setup_without_creating_file(tmp_path, monkeypatch):
    db_file = tmp_path / "absent.sqlite"
    monkeypatch.setenv("RECO_BENCHMARK_DB", str(db_file))
    app = AppTest.from_file(DASHBOARD, default_timeout=30).run()
    assert not app.exception
    assert len(app.info) >= 1  # Setup guidance is rendered without creating the file
    assert not db_file.exists()


def test_seeded_database_shows_catalog_but_no_fake_scores(tmp_path, monkeypatch):
    db_file = tmp_path / "empty.sqlite"
    with connect(db_file):
        pass
    monkeypatch.setenv("RECO_BENCHMARK_DB", str(db_file))
    app = AppTest.from_file(DASHBOARD, default_timeout=30).run()
    assert not app.exception
    assert len(app.metric) == 4
    assert len(app.info) >= 1  # Unpopulated registry displays an explicit empty state


def test_measured_report_populates_chart_and_history(tmp_path, monkeypatch):
    db_file = tmp_path / "results.sqlite"
    with connect(db_file) as db:
        import_record(db, {
            "model": {
                "id": "hofm3_semi25_jointbcd", "name": "HOFM3 Semi25 Joint BCD",
                "family": "higher-order factorization", "stage": "ranking",
                "maturity": "research prototype",
            },
            "dataset": {"id": "ml-32m", "name": "MovieLens 32M", "version": "32m"},
            "protocol": {
                "id": "synthetic-eval-protocol", "dataset_id": "ml-32m",
                "split_method": "synthetic holdout", "candidate_policy": "full catalogue",
                "label_rule": "observed positive", "cohort_signature": "synthetic",
                "cutoff": "synthetic",
            },
            "evaluation_split": "test", "source_type": "measured",
            "source_uri": "synthetic://test-only", "sample_users": 50,
            "config": {"regularization": 0.1},
            "metrics": {"ndcg@10": 0.24, "recall@10": 0.35},
        })
    monkeypatch.setenv("RECO_BENCHMARK_DB", str(db_file))
    app = AppTest.from_file(DASHBOARD, default_timeout=30).run()
    assert not app.exception
    assert len(app.metric) == 4
    assert any(item.label == "Dataset" for item in app.selectbox)
    assert any(item.label == "Ranking metric and cutoff" for item in app.selectbox)
