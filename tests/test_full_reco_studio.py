"""Offline tests for the broad JSON-only research studio."""
import json

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.reco_studio_data import (
    MODEL_CATALOG, checkpoint_inventory, comparable_records, dataset_inventory,
    discover_logs, discover_reports, feature_importances, parse_training_log,
    training_configuration,
)
from pathlib import Path


APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _eval(name, ndcg, *, candidate_policy="full catalogue", split="test"):
    return {
        "dataset": {"id": "ml-32m", "name": "MovieLens 32M"},
        "protocol": {
            "id": "experiment-32m", "dataset_id": "ml-32m",
            "split_method": "chronological", "candidate_policy": candidate_policy,
            "label_rule": "rating >= 4", "cohort_signature": "test-users-hash",
            "cutoff": "2020-01-01",
        },
        "model": {"id": name, "name": name},
        "evaluation_split": split, "source_type": "measured",
        "metrics": {"ndcg@10": ndcg, "recall@500": .2,
                    "hit_rate@10": .14, "not_a_metric": 123},
    }


def test_inventory_has_required_model_families():
    all_models = {entry[0] for entry in MODEL_CATALOG}
    assert {"popularity", "svd", "bpr", "fm_all_bpr",
            "fm_all_l2try_bpr", "hofm3_semi25_jointbcd",
            "hofm4_jointbcd", "deepfm", "temporal_fm",
            "two_tower", "lambdarank", "tiger"} <= all_models


def test_json_discovery_accepts_reports_not_training_configs(tmp_path):
    root = tmp_path / "artifacts"
    root.mkdir()
    real = root / "reports" / "eval.json"
    real.parent.mkdir()
    real.write_text(json.dumps(_eval("hofm4_jointbcd", .13)), encoding="utf-8")
    meta = root / "experiment_config.json"
    meta.write_text(json.dumps({"regularization": 0.1}), encoding="utf-8")
    found = discover_reports(root)
    assert len(found) == 1
    assert found[0][0] == real
    assert found[0][1] == "evaluation"


def test_no_cross_protocol_or_cross_split_comparison(tmp_path):
    docs = []
    for i, (name, policy, split) in enumerate([
        ("hofm3", "full catalogue", "test"),
        ("hofm4", "full catalogue", "test"),
        ("bpr", "100 sampled negatives", "test"),
        ("fm", "full catalogue", "validation"),
    ]):
        path = tmp_path / f"eval_{i}.json"
        report = _eval(name, .10 + i / 100, candidate_policy=policy, split=split)
        docs.append((path, "evaluation", report))
    data = comparable_records(docs)
    scores = data[data["Metric"] == "ndcg@10"]
    assert len(scores) == 4
    scopes = {name: scores[scores["Model"] == name]["Scope"].iloc[0]
              for name in ("hofm3", "hofm4", "bpr", "fm")}
    assert scopes["hofm3"] == scopes["hofm4"]
    assert scopes["hofm3"] != scopes["bpr"]
    assert scopes["hofm3"] != scopes["fm"]
    assert "not_a_metric" not in data["Metric"].tolist()


def test_missing_provenance_isolated_to_own_json(tmp_path):
    a = _eval("hofm3", .13)
    a["protocol"].pop("cohort_signature")
    b = _eval("hofm4", .15)
    b["protocol"].pop("cohort_signature")
    rows = comparable_records([
        (tmp_path / "a.json", "evaluation", a),
        (tmp_path / "b.json", "evaluation", b),
    ])
    assert rows[rows["Metric"] == "ndcg@10"]["Scope"].nunique() == 2


def test_same_baseline_report_can_compare_models(tmp_path):
    report = {
        "dataset": "MovieLens latest-small", "k": 10,
        "validation": {"svd": {"ndcg_at_k": .15}},
        "test": {
            "svd": {"ndcg_at_k": .10},
            "hybrid": {"ndcg_at_k": .18},
        },
    }
    rows = comparable_records([(tmp_path / "metrics.json", "baseline", report)])
    measured = rows[(rows["Metric"] == "ndcg_at_k") &
                    (rows["Split"] == "test")]
    assert set(measured["Model"]) == {"svd", "hybrid"}
    assert measured["Scope"].nunique() == 1


def test_log_parse_rejects_unlabeled_metrics_and_preserves_epoch():
    source = """
    Setup ngdc@10=0.99 (not an evaluated epoch)
    Epoch 1/80 train loss=0.71 val_ndcg@10=0.1201
    epoch 2/80 train loss=0.65 validation_ndcg@10=0.125
    epoch 3/80 training done
    """
    rows = parse_training_log(source)
    assert len(rows[rows["Metric"].str.contains("ndcg")]) == 2
    assert sorted(rows["Epoch"].unique().tolist()) == [1, 2]
    assert rows["Value"].max() == pytest.approx(.71)
    assert parse_training_log("loss=0.1 no epoch reference").empty


def test_checkpoint_config_is_inspected_not_loaded(tmp_path):
    root = tmp_path / "work"
    folder = root / "models" / "ml-32m" / "hofm4"
    folder.mkdir(parents=True)
    (folder / "weights.pt").write_bytes(b"not a real checkpoint")
    (folder / "regularization_config.json").write_text(
        '{"bcd_dropout": 0.3}', encoding="utf-8",
    )
    rows = checkpoint_inventory(root)
    assert {r["Artifact"] for r in rows} == {
        "weights.pt", "regularization_config.json"
    }
    filename, config = training_configuration(folder)
    assert filename == "regularization_config.json"
    assert config["bcd_dropout"] == .3


def test_feature_importance_only_when_saved():
    assert feature_importances({"metrics": {"ndcg@10": .13}}).empty
    sample = {
        "config": {"features": ["score_hofm3", "score_bpr"],
                   "feature_importances": [42, 17]},
    }
    frame = feature_importances(sample)
    assert set(frame["Feature"]) == {"score_hofm3", "score_bpr"}


def test_detect_actual_32m_dataset(tmp_path):
    root = tmp_path
    folder = root / "data/raw/ml-32m"
    folder.mkdir(parents=True)
    (folder / "movies.csv").write_text(
        "movieId,title,genres\n1,Toy Story,Comedy\n", encoding="utf-8")
    (folder / "tags.csv").write_text(
        "userId,movieId,tag,timestamp\n1,1,animated,10\n", encoding="utf-8")
    item = next(entry for entry in dataset_inventory(root)
                if entry["Dataset"] == "MovieLens 32M")
    assert item["Found"] and item["Tags available"]


@pytest.mark.parametrize("page", [
    "Overview", "Model comparisons", "Training analytics",
    "Retrieval & LambdaRank", "Movie search", "UMAP explorer",
    "Checkpoint explorer", "JSON reports", "Methodology",
])
def test_studio_pages_render_without_sqlite_or_training(tmp_path, monkeypatch, page):
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception
    assert app.sidebar.radio[0].value == "Overview"
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception
    assert not (tmp_path / "artifacts/benchmarks.sqlite").exists()


def test_pca_projection_keeps_movie_ids_and_genre_labels():
    import numpy as np
    from src.reco_visual_data import pca_projection
    rng = np.random.default_rng(42)
    info = pd.DataFrame({
        "movieId": range(10), "title": [f"Movie {i}" for i in range(10)],
        "genres": ["Drama|Comedy"] * 10,
    })
    result = pca_projection(rng.normal(size=(10, 6)).astype("float32"), info)
    assert list(result["movieId"]) == list(range(10))
    assert (result["Primary genre"] == "Drama").all()
    assert list(result.columns)[-3:] == ["PCA 1", "PCA 2", "Primary genre"]


def test_saved_movielens_1m_vectors_supported(tmp_path):
    import numpy as np
    from src.reco_visual_data import sample_umap_source
    folder = tmp_path / "ml1m"
    folder.mkdir()
    np.save(folder / "movie_ids.npy", np.arange(1, 12))
    np.save(folder / "title_embeddings.npy", np.eye(11, 6))
    movie_file = tmp_path / "movies.csv"
    pd.DataFrame({
        "movieId": range(1, 12),
        "title": [f"Film {i}" for i in range(1, 12)],
        "genres": ["Action"] * 11,
    }).to_csv(movie_file, index=False)
    x, movies = sample_umap_source(
        "MovieLens 1M title sentence embeddings",
        cache_dir=folder, count=11, movie_files=[movie_file],
    )
    assert x.shape == (11, 6)
    assert movies.iloc[0]["title"] == "Film 1"
    assert movies.iloc[-1]["movieId"] == 11
