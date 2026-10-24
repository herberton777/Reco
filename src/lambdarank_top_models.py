"""Top-model MovieLens 32M LambdaRank experiment (JSON-first).

Use *real, frozen* scores from the existing advanced models; this program
never pretends to load research-only HOFM checkpoints or create scores.
Protocol and per-query alignment checks run before any output is created.

Examples:
  python -m src.lambdarank_top_models audit
  python -m src.lambdarank_top_models run --output artifacts/rerank/top4_v1

For all details see docs/TOP_MODELS_LAMBDARANK.md.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import lambdarank_rerank as lr


PREFERRED_MODELS = (
    "hofm4_jointbcd",
    "hofm3_semi25_jointbcd",
    "hofm3",
    "bpr",
    "temporal_fm",
    "deepfm",
    "semantic",
    "svd",
    "popularity",
)


def select_models(valid, *, names="", top=4, k=10):
    """Select by frozen VALIDATION NDCG, never by held-out test metrics."""
    columns = {c.removeprefix("score_"): c
               for c in valid if c.startswith("score_")}
    if not columns:
        raise ValueError("No score_<model> columns in validation table")
    if top < 1:
        raise ValueError("--top must be positive")
    quality = {
        name: lr.ranking_metrics(valid, valid[col].to_numpy(), k=k)[f"ndcg@{k}"]
        for name, col in columns.items()
    }
    if names:
        chosen = [s.strip() for s in names.split(",") if s.strip()]
        if len(chosen) != len(set(chosen)) or not chosen:
            raise ValueError("Explicit model names must be nonempty and unique")
        unknown = [name for name in chosen if name not in columns]
        if unknown:
            raise ValueError(
                "No frozen candidate scores for: " + ", ".join(unknown)
                + ". Available: " + ", ".join(sorted(columns))
            )
    else:
        # Prioritize actual validation NDCG. Known architecture names are
        # used ONLY as deterministic tie-breakers.
        priority = {name: rank for rank, name in enumerate(PREFERRED_MODELS)}
        chosen = sorted(columns, key=lambda name: (
            -quality[name], priority.get(name, len(priority)), name))[:top]
    return chosen, quality


def audit(train_path, valid_path, test_path, manifest_path, *,
          models="", top=4, k=10):
    """Validate score provenance claims, query independence and input schemas."""
    paths = [Path(x) for x in (train_path, valid_path, test_path, manifest_path)]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing input exports: " + ", ".join(missing)
            + ". Freeze base-model predictions on common chronological"
            " train/valid/test candidate sets first."
        )
    manifest = lr._manifest(manifest_path)
    if manifest["dataset"]["id"] != "ml-32m":
        raise ValueError("Top-model experiment requires dataset.id='ml-32m'")
    train = lr.load_candidates(train_path)
    valid = lr.load_candidates(valid_path, allow_missing_positive=True)
    test = lr.load_candidates(test_path, allow_missing_positive=True)
    features = lr.ensure_split_integrity(train, valid, test)
    chosen, quality = select_models(valid, names=models, top=top, k=k)
    if len(chosen) < 2:
        raise ValueError(
            "At least TWO frozen model score columns are required to test"
            " stacked LambdaRank against an ensemble baseline."
        )
    query_counts = {
        split: int(table["query_id"].nunique())
        for split, table in (("train", train), ("valid", valid), ("test", test))
    }
    snapshot = {
        "dataset": manifest["dataset"],
        "protocol": manifest["protocol"],
        "model_selection": "explicit" if models else "validation_ndcg_only",
        "available_models": sorted(c.removeprefix("score_")
                                   for c in features if c.startswith("score_")),
        "selected_models": chosen,
        "validation_frozen_ndcg": {
            k: quality[k] for k in sorted(quality)
        },
        "query_counts": query_counts,
        "candidate_rows": {
            "train": len(train), "valid": len(valid), "test": len(test),
        },
        "features": features,
        "input_sha256": {
            "train": lr._sha256(train_path),
            "valid": lr._sha256(valid_path),
            "test": lr._sha256(test_path),
            "manifest": lr._sha256(manifest_path),
        },
    }
    return snapshot, train, valid, test


def _per_query_ndcg(frame, scores, k):
    """Query-level NDCG for paired bootstrap CIs, including missing positives."""
    arr = np.asarray(scores, dtype="float64")
    if len(arr) != len(frame) or not np.isfinite(arr).all():
        raise ValueError("Invalid prediction vector")
    results = []
    at = 0
    for n in lr._groups(frame):
        n = int(n)
        block = frame.iloc[at:at + n]
        labels = block["label"].to_numpy(dtype="int8")
        top = np.argsort(-arr[at:at + n], kind="stable")[:k]
        ideal_n = (int(block["total_relevant"].iat[0]) if "total_relevant" in block
                   else int(labels.sum()))
        if ideal_n < 1:
            raise ValueError("Zero total relevant items in a query")
        weights = 1 / np.log2(np.arange(2, len(top) + 2))
        ideal_weights = 1 / np.log2(np.arange(2, min(ideal_n, k) + 2))
        results.append(float(labels[top] @ weights / ideal_weights.sum()))
        at += n
    return np.asarray(results, dtype="float64")


def _normalized_fusion(frame, columns, *, method):
    """Unsupervised fusion, never fitted on held-out test labels."""
    values = frame[columns].to_numpy(dtype="float64")
    result = np.empty(len(values), dtype="float64")
    offset = 0
    for n in lr._groups(frame):
        n = int(n)
        batch = values[offset:offset + n]
        if method == "zmean":
            mu = batch.mean(axis=0)
            scale = batch.std(axis=0)
            scale[scale < 1e-12] = 1.
            result[offset:offset + n] = ((batch - mu) / scale).mean(axis=1)
        elif method == "rrf":
            accumulator = np.zeros(n, dtype="float64")
            for column in range(batch.shape[1]):
                ranks = np.empty(n, dtype="float64")
                positions = np.argsort(-batch[:, column], kind="stable")
                ranks[positions] = np.arange(1, n + 1)
                accumulator += 1. / (60. + ranks)
            result[offset:offset + n] = accumulator
        else:
            raise ValueError("Unknown fixed fusion: " + method)
        offset += n
    return result


def _read_exported_predictions(test, file):
    pred = lr._keys(lr._read(file))
    if "lambdarank_score" not in pred:
        raise ValueError("Ranker did not export lambdarank_score")
    actual = pd.MultiIndex.from_frame(pred[["query_id", "item_id"]])
    expected = pd.MultiIndex.from_frame(test[["query_id", "item_id"]])
    if len(actual) != len(expected) or not actual.symmetric_difference(expected).empty:
        raise ValueError("Ranker predictions do not match the held-out candidates")
    scores = pd.Series(
        pd.to_numeric(pred["lambdarank_score"], errors="raise").to_numpy(),
        index=actual,
    ).reindex(expected).to_numpy(dtype="float64")
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite reranking scores")
    return scores


def bootstrap_delta(frame, champion_scores, reference_scores, k, *,
                    repeats=1000, seed=42):
    """Paired query-bootstrap interval, not significance under IID user assumptions."""
    if repeats < 100:
        raise ValueError("Bootstrap needs >=100 repeats")
    a = _per_query_ndcg(frame, champion_scores, k)
    b = _per_query_ndcg(frame, reference_scores, k)
    diff = a - b
    rng = np.random.default_rng(seed)
    simulations = np.empty(repeats, dtype="float64")
    for i in range(repeats):
        ids = rng.integers(0, len(diff), size=len(diff))
        simulations[i] = diff[ids].mean()
    low, high = np.quantile(simulations, [0.025, 0.975])
    return {
        "delta_ndcg": float(diff.mean()),
        "paired_query_bootstrap_ci_95": [float(low), float(high)],
        "number_of_queries": int(len(diff)),
        "bootstrap_samples": repeats,
        "assumption": "resampled queries; distinct queries may belong to the same user",
    }


def run(args):
    # Run the complete audit before starting expensive work or creating artifacts.
    snapshot, train, valid, test = audit(
        args.train, args.valid, args.test, args.manifest,
        models=args.models, top=args.top, k=args.k,
    )
    selected = snapshot["selected_models"]
    if args.dry_run:
        print(json.dumps({
            "status": "READY_NO_TRAINING_STARTED",
            **snapshot,
            "output": str(args.output),
        }, indent=2))
        return None

    import lightgbm  # noqa: F401 — fail before creating output folders
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(
            "Previous experiment exists; choose a NEW output path: " + str(output)
        )
    # Existing model-specific experiment() saves every frozen base, each
    # single-model LambdaRank and the full score-stacked LambdaRank.
    training_args = argparse.Namespace(
        train=args.train, valid=args.valid, test=args.test,
        manifest=args.manifest, models=",".join(selected), ensemble=True,
        k=args.k, trees=args.trees, early_stop=args.early_stop,
        lr=args.lr, leaves=args.leaves, min_leaf=args.min_leaf,
        l1=args.l1, l2=args.l2, seed=args.seed, threads=args.threads,
        output=str(output), db=None,
    )
    experiment_summary = lr.experiment(training_args)
    cols = ["score_" + model for model in selected]
    candidates = {}
    # Frozen score baselines: one per model. Best original selected strictly
    # on validation. Other columns remain available but are not compared
    # as if part of the selected ablation.
    for name in selected:
        candidates["frozen_" + name] = (
            valid["score_" + name].to_numpy(),
            test["score_" + name].to_numpy(),
        )
    for method in ("zmean", "rrf"):
        candidates[method] = (
            _normalized_fusion(valid, cols, method=method),
            _normalized_fusion(test, cols, method=method),
        )
    for name in [*selected, "ensemble"]:
        val_file = output / ("predictions_valid_" + name + ".csv")
        # The underlying runner exports test-only predictions; for validation
        # use the validation NDCG already recorded in summary.json.
        test_pred = _read_exported_predictions(
            test, output / ("predictions_" + name + ".csv"))
        candidates["lambdarank_" + name] = (None, test_pred)

    rows = []
    for name, (val_score, test_score) in candidates.items():
        if val_score is None:
            key = name.removeprefix("lambdarank_")
            val_metrics = experiment_summary["variants"][key]["validation"]
        else:
            val_metrics = lr.ranking_metrics(valid, val_score, args.k)
        test_metrics = lr.ranking_metrics(test, test_score, args.k)
        rows.append({
            "variant": name, "type": (
                "lambdarank" if name.startswith("lambdarank_")
                else "frozen" if name.startswith("frozen_")
                else "unsupervised_fusion"),
            "validation": val_metrics, "test": test_metrics,
        })

    reference = max(("frozen_" + name for name in selected),
                    key=lambda key: next(
                        row["validation"][f"ndcg@{args.k}"]
                        for row in rows if row["variant"] == key))
    champion = max(rows, key=lambda row: row["validation"][f"ndcg@{args.k}"])
    champ_name = champion["variant"]
    test_winner = candidates[champ_name][1]
    base_test = candidates[reference][1]
    uncertainty = bootstrap_delta(
        test, test_winner, base_test, args.k, repeats=args.bootstrap, seed=args.seed,
    )
    import lightgbm as lgb
    stacked_booster = lgb.Booster(
        model_file=str(output / "lambdarank_ensemble.txt"))
    stacked_gain = {
        str(feature): float(value)
        for feature, value in zip(
            stacked_booster.feature_name(),
            stacked_booster.feature_importance(importance_type="gain"),
        )
    }
    outcome = {
        "status": "COMPLETED_WITH_MEASURED_SCORES",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "max validation NDCG only; test untouched until selection",
        "selected_champion_on_validation": champ_name,
        "best_frozen_reference_on_validation": reference,
        "compared_against": "exact same query/item candidate sets",
        "measured_test_delta_vs_frozen": uncertainty,
        "feature_importances": stacked_gain,
        "feature_importance_type": "LightGBM gain, trained stacked LambdaRank",
        "audit": snapshot,
        "variants": rows,
        "files": {
            "rerank_summary": "summary.json",
            "ranker_reports": "report_lambdarank_*.json",
            "ranking_predictions": "predictions_*.csv",
        },
    }
    # JSON-first Studio can display each individual report and the prior
    # ranker's summary.json. This analysis is intentionally separate.
    filename = output / "top_models_analysis.json"
    if filename.exists():
        raise FileExistsError(str(filename))
    filename.write_text(json.dumps(outcome, indent=2), encoding="utf-8")
    print("Top-model experiment summary:", filename)
    print("CHAMPION selected on validation:", champ_name)
    print("NDCG test paired delta vs strongest frozen model:",
          uncertainty["delta_ndcg"])
    return outcome


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "run"))
    parser.add_argument("--train", default="artifacts/rerank/train.csv")
    parser.add_argument("--valid", default="artifacts/rerank/valid.csv")
    parser.add_argument("--test", default="artifacts/rerank/test.csv")
    parser.add_argument("--manifest",
                        default="artifacts/rerank/evaluation_manifest.json")
    parser.add_argument("--models", default="",
                        help="Exact comma-separated model IDs; default top validation models")
    parser.add_argument("--top", type=int, default=4,
                        help="Max top frozen models by validation NDCG (default 4)")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output", default="artifacts/rerank/top_models_v1")
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--early-stop", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--leaves", type=int, default=15)
    parser.add_argument("--min-leaf", type=int, default=20)
    parser.add_argument("--l1", type=float, default=0.0)
    parser.add_argument("--l2", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.k < 1 or args.threads < 1:
        parser.error("K and threads must be positive")
    if args.command == "audit":
        info, *_ = audit(
            args.train, args.valid, args.test, args.manifest,
            models=args.models, top=args.top, k=args.k,
        )
        print(json.dumps({"status": "READY", **info}, indent=2))
    else:
        run(args)


if __name__ == "__main__":
    main()
