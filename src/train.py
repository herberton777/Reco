"""Training entry point: python -m src.train [--neural] [--no-ranker]."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from .data import ensure_dataset, chronological_split
from .recommender import HybridRecommender
from .evaluation import evaluate_models, union_seen


def run(data_dir="data", output_dir="artifacts", neural=False, epochs=4,
        no_ranker=False, k=10, rank=48, registry_db=None):
    ratings, movies = ensure_dataset(data_dir)
    splits = chronological_split(ratings)
    print(f"Positive events train={len(splits.train):,}, rank={len(splits.rank_train):,}, "
          f"valid={len(splits.valid):,}, test={len(splits.test):,}")
    model = HybridRecommender().fit(splits.train, movies, rank=rank)
    if neural:
        model.fit_neural(splits.train, epochs=epochs)
    if not no_ranker:
        model.fit_ranker(splits.rank_train)
    valid = evaluate_models(model, splits.valid, union_seen(splits.rank_train), k=k)
    test = evaluate_models(model, splits.test,
                           union_seen(splits.rank_train, splits.valid), k=k)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "recommender.joblib")
    metrics = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "MovieLens latest-small (GroupLens)",
        "positive_rating_threshold": 4.0,
        "split": "per-user last 3 positive events: rank-train / valid / test",
        "caveat": "unobserved items are not observed negatives; offline metrics are biased",
        "counts": {"all_ratings": len(ratings), "training_positives": len(splits.train),
                   "ranker_positives": len(splits.rank_train), "users_test": len(splits.test)},
        "catalog_size": len(movies), "k": k,
        "validation": valid, "test": test,
    }
    metrics_path = out / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if registry_db is not None:
        from .benchmark_registry import connect, import_baseline
        with connect(registry_db) as db:
            imported_ids = import_baseline(db, metrics_path)
        print(f"Registered {len(imported_ids)} validation/test model runs in {registry_db}")
    print("\nTest results (full catalogue, one held-out positive/user):")
    for name, data in test.items():
        print(f"{name:12s} Recall@{k}={data['recall_at_k']}  "
              f"NDCG@{k}={data['ndcg_at_k']}  Coverage={data['catalog_coverage_at_k']}")
    print(f"\nSaved to {out.resolve()}")
    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--neural", action="store_true", help="Train a PyTorch two-tower model")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--no-ranker", action="store_true", help="Skip LightGBM LambdaMART")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--rank", type=int, default=48, help="SVD factor dimension")
    parser.add_argument("--registry-db", default=None,
                        help="Optional SQLite benchmark database to auto-register results")
    args = parser.parse_args()
    run(**vars(args))
