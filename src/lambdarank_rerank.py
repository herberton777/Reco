"""Protocol-safe, reusable LightGBM LambdaRank over frozen recommender scores.

For every query/candidate row, inputs use:
  query_id, item_id, label, score_<model>..., feature_<historical_metadata>...
Train, validation and test files must contain identical score/feature columns.
The evaluator ranks ONLY the supplied candidates. It never claims full-catalogue
NDCG unless the supplied, independently documented protocol truly is full-catalogue.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd

from .benchmark_registry import connect, import_record

REQUIRED = {"query_id", "item_id", "label"}
MODEL_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
PROTOCOL_KEYS = ("id", "dataset_id", "split_method", "candidate_policy",
                 "label_rule", "cohort_signature", "cutoff")


def _sha256(path):
    with Path(path).open("rb") as f:
        digest = hashlib.sha256()
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Candidates must be CSV or Parquet: " + str(path))


def _write(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".parquet", ".pq"}:
        frame.to_parquet(path, index=False)
    elif path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    else:
        raise ValueError("Output must be .csv or .parquet")
    return path


def _keys(frame, require_label=True):
    required = {"query_id", "item_id"} | ({"label"} if require_label else set())
    missing = required - set(frame)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
    if frame.empty:
        raise ValueError("Candidate table is empty")
    if frame[["query_id", "item_id"]].isna().any().any():
        raise ValueError("Null query or item identifier")
    frame = frame.copy()
    frame["query_id"] = frame["query_id"].astype(str)
    frame["item_id"] = frame["item_id"].astype(str)
    if frame.duplicated(["query_id", "item_id"]).any():
        raise ValueError("Duplicate query_id/item_id candidate")
    return frame


def join_scores(base_path, score_sources, output):
    """Exact candidate-key join: fail on any mismatched or duplicated candidates.

    Base table has labels and optionally temporal/historical feature_* fields.
    Each score source contains query_id,item_id,score (not relevance labels).
    """
    base = _keys(_read(base_path))
    if not score_sources:
        raise ValueError("Provide at least one --score MODEL=PATH")
    allowed = REQUIRED | {"total_relevant"} | {c for c in base if c.startswith("feature_")}
    if not set(base.columns).issubset(allowed):
        raise ValueError("Base table must contain query_id,item_id,label,total_relevant,feature_* only")
    base_index = pd.MultiIndex.from_frame(base[["query_id", "item_id"]])
    for spec in score_sources:
        if "=" not in spec:
            raise ValueError("--score must be MODEL=PATH")
        model, filename = spec.split("=", 1)
        if not MODEL_PATTERN.fullmatch(model):
            raise ValueError("Invalid model identifier: " + model)
        column = "score_" + model
        if column in base:
            raise ValueError("Score was specified more than once: " + model)
        source = _keys(_read(filename), require_label=False)
        if set(source.columns) != {"query_id", "item_id", "score"}:
            raise ValueError("Model score files must contain exactly query_id,item_id,score")
        source_index = pd.MultiIndex.from_frame(source[["query_id", "item_id"]])
        if len(source) != len(base) or not source_index.symmetric_difference(base_index).empty:
            raise ValueError("Model " + model + " does not score exactly the same candidates")
        source = source.rename(columns={"score": column})
        base = base.merge(source, how="left", on=["query_id", "item_id"],
                          validate="one_to_one", sort=False)
        if base[column].isna().any():
            raise ValueError("Missing joined predictions for " + model)
    return _write(base, output)


def load_candidates(path, *, allow_missing_positive=False):
    frame = _keys(_read(path))
    if "label" not in frame:
        raise ValueError("label required")
    feature_columns = sorted(c for c in frame if c.startswith("score_") or c.startswith("feature_"))
    if not any(c.startswith("score_") for c in feature_columns):
        raise ValueError("Supply at least one frozen score_<model> column")
    extra = set(frame) - REQUIRED - {"total_relevant"} - set(feature_columns)
    if extra:
        raise ValueError("Unsupported columns (possible leakage): " + ", ".join(sorted(extra)))
    if any(not MODEL_PATTERN.fullmatch(col) for col in feature_columns):
        raise ValueError("Score/feature columns must use lowercase letters, digits, underscores")
    labels = pd.to_numeric(frame["label"], errors="coerce")
    if not labels.isin([0, 1]).all():
        raise ValueError("Binary observed-positive labels (0/1) required")
    frame["label"] = labels.astype("int8")
    if "total_relevant" in frame:
        counts = pd.to_numeric(frame["total_relevant"], errors="coerce")
        if (counts.isna().any() or not np.isfinite(counts).all()
                or (counts < 1).any() or (counts != np.floor(counts)).any()):
            raise ValueError("total_relevant must be a positive integer per query")
        frame["total_relevant"] = counts.astype("int64")
    for col in feature_columns:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
        if not np.isfinite(frame[col].to_numpy(dtype="float64")).all():
            raise ValueError("Non-finite score or feature in " + col)
    frame = frame.sort_values(["query_id", "item_id"], kind="stable").reset_index(drop=True)
    groups = frame.groupby("query_id", sort=False)["label"].agg(["size", "sum"])
    if (groups["size"] < 2).any():
        raise ValueError("Every query needs at least two candidate items")
    if not allow_missing_positive and (
            (groups["sum"] < 1).any() or (groups["sum"] >= groups["size"]).any()):
        raise ValueError("Training queries need >=1 positive and >=1 unlabeled item")
    if "total_relevant" in frame:
        relevant_counts = frame.groupby("query_id", sort=False)["total_relevant"].agg(
            ["min", "max", "first"])
        if (relevant_counts["min"] != relevant_counts["max"]).any():
            raise ValueError("total_relevant must be constant within each query")
        if (relevant_counts["first"] < groups["sum"]).any():
            raise ValueError("total_relevant cannot be smaller than positive candidates")
    elif (groups["sum"] == 0).any():
        raise ValueError(
            "Queries with no retrieved positives need total_relevant so misses are counted")
    return frame


def _groups(frame):
    counts = frame.groupby("query_id", sort=False).size().to_numpy(dtype="int32")
    if int(counts.sum()) != len(frame):
        raise ValueError("Invalid query grouping")
    return counts


def ensure_split_integrity(train, valid, test):
    columns = {c for c in train if c.startswith(("score_", "feature_"))}
    for name, frame in (("validation", valid), ("test", test)):
        actual = {c for c in frame if c.startswith(("score_", "feature_"))}
        if actual != columns:
            raise ValueError(name + " score/feature schema differs from training")
    groups = [set(df["query_id"]) for df in (train, valid, test)]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError("Query IDs overlap between train, validation and test")
    return sorted(columns)


def ranking_metrics(frame, scores, k=10):
    """Mean per-query candidate-set NDCG and Recall (not retrieval recall)."""
    if k <= 0:
        raise ValueError("k must be positive")
    scores = np.asarray(scores, dtype="float64")
    if len(scores) != len(frame) or not np.isfinite(scores).all():
        raise ValueError("Prediction length or finiteness mismatch")
    ndcg, recall, hit = [], [], []
    offset = 0
    for size in _groups(frame):
        block = frame.iloc[offset:offset + size]
        labels = block["label"].to_numpy(dtype="int8")
        local_scores = scores[offset:offset + size]
        # Stable sorting gives a documented tie-break by item ID.
        top = np.argsort(-local_scores, kind="stable")[:k]
        gains = labels[top].astype("float64")
        weights = 1 / np.log2(np.arange(2, len(top) + 2))
        dcg = float(np.dot(gains, weights))
        actual_total = (int(block["total_relevant"].iat[0])
                        if "total_relevant" in block else int(labels.sum()))
        if actual_total < 1:
            raise ValueError("Cannot score a query without known total relevant items")
        ideal_size = min(actual_total, k)
        ideal = float((1 / np.log2(np.arange(2, ideal_size + 2))).sum())
        ndcg.append(dcg / ideal)
        recall.append(float(gains.sum() / actual_total))
        hit.append(float(gains.sum() > 0))
        offset += int(size)
    return {
        f"ndcg@{k}": float(np.mean(ndcg)),
        f"recall@{k}": float(np.mean(recall)),
        f"hit_rate@{k}": float(np.mean(hit)),
        "queries_evaluated": len(ndcg),
    }


def _manifest(path):
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(m, dict) or not isinstance(m.get("dataset"), dict) or (
            not isinstance(m.get("protocol"), dict)):
        raise ValueError("Manifest must specify dataset and protocol objects")
    dataset = m["dataset"]
    for key in ("id", "name", "version"):
        if not str(dataset.get(key, "")).strip():
            raise ValueError("Missing dataset." + key)
    protocol = m["protocol"]
    for key in PROTOCOL_KEYS:
        if not str(protocol.get(key, "")).strip():
            raise ValueError("Missing protocol." + key)
    if protocol["dataset_id"] != dataset["id"]:
        raise ValueError("Protocol dataset does not match dataset")
    for statement in ("features_from_past_only", "frozen_base_models",
                      "test_untouched_for_training"):
        if m.get(statement) is not True:
            raise ValueError("Manifest must explicitly attest " + statement + "=true")
    return m


def _ranker(train, valid, columns, args):
    import lightgbm as lgb
    # Suppress built-in NDCG when a retrieved pool might omit known positives:
    # its per-query ideal ranking would otherwise exclude unretrieved positives.
    # Validate against total_relevant instead, so misses score zero.
    def validation_ndcg(y_true, y_pred, weight=None, group=None):
        result = ranking_metrics(valid, y_pred, args.k)
        return (f"end_to_end_ndcg_at_{args.k}", result[f"ndcg@{args.k}"], True)

    ranker = lgb.LGBMRanker(
        objective="lambdarank", metric="None", n_estimators=args.trees,
        learning_rate=args.lr, num_leaves=args.leaves,
        max_depth=-1, min_child_samples=args.min_leaf,
        reg_lambda=args.l2, reg_alpha=args.l1,
        random_state=args.seed, n_jobs=args.threads, verbosity=-1,
        max_bin=127,
    )
    ranker.fit(
        train[columns], train["label"].to_numpy(dtype="int8"), group=_groups(train),
        eval_set=[(valid[columns], valid["label"].to_numpy(dtype="int8"))],
        eval_group=[_groups(valid)], eval_metric=validation_ndcg, eval_at=[args.k],
        callbacks=[lgb.early_stopping(args.early_stop, first_metric_only=True,
                                      verbose=False)],
    )
    return ranker


def _record_report(output_path, dataset, protocol, model_id, model_name, metrics,
                   train_path, valid_path, test_path, features, args, best_iteration,
                   manifest_path, notes, manifest_query_is_user=False):
    """Save immutable measured report to disk; optional import occurs afterwards."""
    if not MODEL_PATTERN.fullmatch(model_id):
        raise ValueError("Invalid model id: " + model_id)
    query_count = metrics.pop("queries_evaluated")
    committed = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                               text=True, check=False)
    report = {
        "model": {
            "id": model_id, "name": model_name,
            "family": "LambdaRank reranker" if model_id.startswith("lambdarank_")
                      else "frozen candidate scorer",
            "stage": "reranking" if model_id.startswith("lambdarank_") else "ranking",
            "maturity": "experiment",
        },
        "dataset": dataset, "protocol": protocol, "evaluation_split": "test",
        "source_type": "measured", "source_uri": str(output_path.resolve()),
        # A query is not necessarily a unique user (one user may have multiple
        # chronologically distinct ranking requests). Do not mislabel query counts.
        "sample_users": query_count if manifest_query_is_user else None,
        "seed": args.seed,
        "git_commit": committed.stdout.strip() if committed.returncode == 0 else None,
        "config": {
            "features": features, "best_iteration": best_iteration,
            "lambdarank_params": {
                "trees": args.trees, "lr": args.lr, "leaves": args.leaves,
                "min_leaf": args.min_leaf, "early_stop": args.early_stop,
                "l1": args.l1, "l2": args.l2, "k": args.k,
            },
            "inputs_sha256": {
                "train": _sha256(train_path), "valid": _sha256(valid_path),
                "test": _sha256(test_path), "manifest": _sha256(manifest_path),
            },
            "query_definition": "query_id; per-query rank within supplied candidate pool",
            "test_query_count": query_count,
        },
        "notes": notes,
        "metrics": metrics,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if output_path.exists():
        raise FileExistsError("Refusing to overwrite a measured report: " + str(output_path))
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _variant_features(variant, score_columns, metadata):
    return sorted(set(metadata + score_columns) if variant == "ensemble" else
                  set(metadata + ["score_" + variant]))


def experiment(args):
    """Fit one reranker per base model plus optional multi-model ensemble."""
    manifest = _manifest(args.manifest)
    train_path, valid_path, test_path = map(Path, (args.train, args.valid, args.test))
    train = load_candidates(train_path)
    valid = load_candidates(valid_path, allow_missing_positive=True)
    test = load_candidates(test_path, allow_missing_positive=True)
    features = ensure_split_integrity(train, valid, test)
    scores = [c for c in features if c.startswith("score_")]
    metadata = [c for c in features if c.startswith("feature_")]
    names = [name.removeprefix("score_") for name in scores]
    if args.models:
        chosen = [v.strip() for v in args.models.split(",") if v.strip()]
        if len(set(chosen)) != len(chosen) or any(v not in names for v in chosen):
            raise ValueError("Requested --models must be distinct names among: " + ", ".join(names))
    else:
        chosen = names
    if not chosen:
        raise ValueError("No base model score columns available")
    variants = chosen + (["ensemble"] if args.ensemble and len(chosen) > 1 else [])
    if args.k < 1 or args.threads < 1 or args.trees < 1 or args.early_stop < 1:
        raise ValueError("K, trees, threads and early stopping must be positive")
    if args.min_leaf < 1 or args.leaves < 2:
        raise ValueError("Invalid tree size")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Output directory exists; choose a new --output: " + str(output))
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "manifest": manifest, "split_counts": {
            "train_queries": len(_groups(train)), "valid_queries": len(_groups(valid)),
            "test_queries": len(_groups(test)),
        },
        "candidate_protocol_note": "NDCG/Recall are conditional on candidates supplied. "
                                   "Not full-catalogue unless protocol explicitly verifies it.",
        "variants": {},
    }
    report_paths = []
    import lightgbm as lgb  # fail early if dependency absent
    for name in variants:
        cols = _variant_features(name, ["score_" + n for n in chosen], metadata)
        ranker = _ranker(train, valid, cols, args)
        model_file = output / ("lambdarank_" + name + ".txt")
        ranker.booster_.save_model(str(model_file))
        val_predictions = ranker.predict(valid[cols])
        test_predictions = ranker.predict(test[cols])
        after_valid = ranking_metrics(valid, val_predictions, args.k)
        after = ranking_metrics(test, test_predictions, args.k)
        # Baseline best is not pooled; a single variant uses its own base model.
        summary["variants"][name] = {
            "features": cols, "best_iteration": ranker.best_iteration_,
            "validation": after_valid, "test": after,
            "model_file": str(model_file),
        }
        columns_to_export = ["query_id", "item_id", "label"]
        if "total_relevant" in test:
            columns_to_export.append("total_relevant")
        pred = test[columns_to_export].copy()
        pred["lambdarank_score"] = test_predictions
        _write(pred, output / ("predictions_" + name + ".csv"))
        if name != "ensemble":
            score = "score_" + name
            baseline = ranking_metrics(test, test[score].to_numpy(), args.k)
            summary["variants"][name]["baseline_test"] = baseline
            baseline_id = name if name in ("bpr", "deepfm", "temporal_fm", "hofm3",
                                            "hofm3_semi25_jointbcd", "svd", "popularity") else "score_" + name
            baseline_path = output / ("report_baseline_" + name + ".json")
            _record_report(
                baseline_path, manifest["dataset"], manifest["protocol"],
                baseline_id, name + " frozen baseline", dict(baseline),
                train_path, valid_path, test_path, [score], args, None,
                args.manifest, "Baseline: frozen score ranked on exactly the same test candidate sets.",
                manifest_query_is_user=manifest.get("query_id_is_user") is True,
            )
            report_paths.append(baseline_path)
        report_path = output / ("report_lambdarank_" + name + ".json")
        _record_report(
            report_path, manifest["dataset"], manifest["protocol"],
            "lambdarank_" + name, "LambdaRank (" + name + ")", dict(after),
            train_path, valid_path, test_path, cols, args, ranker.best_iteration_,
            args.manifest, "Stage-2 reranker: fitted on train queries, early stopped on "
                           "validation queries, evaluated on held-out test queries.",
            manifest_query_is_user=manifest.get("query_id_is_user") is True,
        )
        report_paths.append(report_path)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.db:
        with connect(args.db) as db:
            for path in report_paths:
                import_record(db, json.loads(path.read_text(encoding="utf-8")))
        print("Imported", len(report_paths), "measured model/test reports to", args.db)
    print(json.dumps(summary["variants"], indent=2))
    print("Saved results in", output)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    join = sub.add_parser("join", help="Join independently scored models on identical candidate keys")
    join.add_argument("--base", required=True, help="Labeled query/item/feature candidate table")
    join.add_argument("--score", action="append", required=True, help="MODEL=score_file.csv")
    join.add_argument("--output", required=True)
    train = sub.add_parser("train", help="Train and evaluate frozen-score LambdaRank variants")
    train.add_argument("--train", required=True)
    train.add_argument("--valid", required=True)
    train.add_argument("--test", required=True)
    train.add_argument("--manifest", required=True, help="Protocol and leakage attestation JSON")
    train.add_argument("--models", default="", help="Comma-separated base model names (default all)")
    train.add_argument("--ensemble", action="store_true")
    train.add_argument("--k", type=int, default=10)
    train.add_argument("--trees", type=int, default=300)
    train.add_argument("--early-stop", type=int, default=30)
    train.add_argument("--lr", type=float, default=0.03)
    train.add_argument("--leaves", type=int, default=15)
    train.add_argument("--min-leaf", type=int, default=20)
    train.add_argument("--l1", type=float, default=0.0)
    train.add_argument("--l2", type=float, default=10.0)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--threads", type=int, default=4)
    train.add_argument("--output", required=True)
    train.add_argument("--db", default=None, help="Optional benchmark SQLite database")
    args = parser.parse_args(argv)
    if args.command == "join":
        print("Wrote", join_scores(args.base, args.score, args.output))
        return
    experiment(args)


if __name__ == "__main__":
    main()
