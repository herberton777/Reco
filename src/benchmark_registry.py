"""Provenance-aware SQLite benchmark registry (standard library only).

Usage: python -m src.benchmark_registry --db artifacts/benchmarks.sqlite init
See docs/BENCHMARK_REGISTRY.md for import formats and comparability rules.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
from pathlib import Path

DEFAULT_DB = Path("artifacts/benchmarks.sqlite")
MODEL_CATALOG = [
    ("popularity", "Popularity", "non-personalized", "retrieval", "verified baseline"),
    ("svd", "TruncatedSVD", "collaborative filtering", "retrieval", "verified baseline"),
    ("genres", "Genre TF-IDF", "content-based", "retrieval", "verified baseline"),
    ("hybrid", "Fixed hybrid", "hybrid", "ranking", "verified baseline"),
    ("two_tower", "Two-tower", "neural collaborative", "retrieval", "optional baseline"),
    ("hybrid_neural", "Neural hybrid", "hybrid", "ranking", "optional baseline"),
    ("lambdamart", "LightGBM LambdaMART", "learning-to-rank", "reranking", "verified baseline"),
    ("bpr", "BPR", "matrix factorization", "retrieval", "archived prototype"),
    ("deepfm", "DeepFM", "factorization machine", "ranking", "archived prototype"),
    ("temporal_fm", "Temporal FM", "factorization machine", "ranking", "archived prototype"),
    ("hofm3", "Third-order HOFM", "higher-order factorization", "ranking", "archived prototype"),
    ("hofm3_semi25_jointbcd", "HOFM3 Semi25 Joint BCD", "higher-order factorization", "ranking", "research proposal"),
    ("tiger", "TIGER-inspired generative retrieval", "generative retrieval", "retrieval", "archived prototype"),
]
DATASETS = [
    ("ml-latest-small", "MovieLens latest-small", "latest-small"),
    ("ml-1m", "MovieLens 1M", "1m"),
    ("ml-32m", "MovieLens 32M", "32m"),
]
BASELINE_PROTOCOL = {
    "id": "ml-small-per-user-3-fullcatalog-v1",
    "dataset_id": "ml-latest-small",
    "split_method": "per-user last 3 positives: rank-train / valid / test",
    "candidate_policy": "full catalogue; trained and earlier held-out positives excluded",
    "label_rule": "rating >= 4 is an observed positive; unobserved != negative",
    "cohort_signature": "eligible users with >=4 positive events; model-dependent scoring eligibility",
    "cutoff": "per-user event timestamps; no global cutoff",
}

DDL = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS models (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, family TEXT NOT NULL,
  stage TEXT NOT NULL, maturity TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS datasets (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, version TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS protocols (
  id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL REFERENCES datasets(id),
  split_method TEXT NOT NULL, candidate_policy TEXT NOT NULL,
  label_rule TEXT NOT NULL, cohort_signature TEXT NOT NULL, cutoff TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY, model_id TEXT NOT NULL REFERENCES models(id),
  dataset_id TEXT NOT NULL REFERENCES datasets(id),
  protocol_id TEXT NOT NULL REFERENCES protocols(id),
  evaluation_split TEXT NOT NULL,
  source_type TEXT NOT NULL CHECK(source_type IN ('measured','literature')),
  source_uri TEXT NOT NULL, source_sha256 TEXT,
  sample_users INTEGER CHECK(sample_users IS NULL OR sample_users >= 0),
  seed INTEGER, git_commit TEXT, config_json TEXT NOT NULL,
  notes TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS metrics (
  run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  metric TEXT NOT NULL, k INTEGER NOT NULL DEFAULT 0 CHECK(k >= 0),
  value REAL NOT NULL, PRIMARY KEY(run_id, metric, k)
);
CREATE INDEX IF NOT EXISTS ix_runs_comparison
  ON runs(dataset_id, protocol_id, evaluation_split, source_type);
CREATE INDEX IF NOT EXISTS ix_metrics_name ON metrics(metric, k);

-- Training attempts are separate from evaluated benchmark runs. A successful
-- process exit is not an accuracy result and must never enter the leaderboard.
CREATE TABLE IF NOT EXISTS training_attempts (
  id TEXT PRIMARY KEY,
  model_id TEXT NOT NULL REFERENCES models(id),
  dataset_id TEXT NOT NULL REFERENCES datasets(id),
  run_label TEXT NOT NULL,
  script_path TEXT NOT NULL,
  command_json TEXT NOT NULL,
  working_directory TEXT NOT NULL,
  started_at_utc TEXT NOT NULL,
  finished_at_utc TEXT,
  status TEXT NOT NULL CHECK(status IN ('running','completed','failed','interrupted','imported')),
  exit_code INTEGER,
  git_commit TEXT,
  configuration_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS ix_training_attempts_label
  ON training_attempts(dataset_id, run_label, started_at_utc);
CREATE TABLE IF NOT EXISTS training_benchmarks (
  attempt_id TEXT NOT NULL REFERENCES training_attempts(id) ON DELETE CASCADE,
  benchmark_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  PRIMARY KEY(attempt_id, benchmark_id)
);
"""

RATE_METRICS = {"ndcg", "recall", "precision", "map", "mrr", "hit_rate", "catalog_coverage"}
METRIC_PATTERN = re.compile(r"^([a-z][a-z0-9_]*)(?:@(\d+))?$")


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def connect(path=DEFAULT_DB):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(path))
    db.row_factory = sqlite3.Row
    db.executescript(DDL)
    seed_catalog(db)
    return db


def seed_catalog(db):
    db.executemany("INSERT OR IGNORE INTO models VALUES (?,?,?,?,?)", MODEL_CATALOG)
    db.executemany("INSERT OR IGNORE INTO datasets VALUES (?,?,?)", DATASETS)
    register_protocol(db, BASELINE_PROTOCOL)
    db.commit()


def register_protocol(db, protocol):
    keys = ("id", "dataset_id", "split_method", "candidate_policy", "label_rule",
            "cohort_signature", "cutoff")
    values = tuple(str(protocol[key]) for key in keys)
    existing = db.execute("SELECT * FROM protocols WHERE id=?", (values[0],)).fetchone()
    if existing:
        if tuple(existing[key] for key in keys) != values:
            raise ValueError("Protocol ID reused with different definition: " + values[0])
        return
    db.execute("INSERT INTO protocols VALUES (?,?,?,?,?,?,?)", values)


def _metric_parts(name, value):
    match = METRIC_PATTERN.fullmatch(name.lower())
    if not match:
        raise ValueError("Invalid metric name: " + name)
    metric, k_text = match.groups()
    k = int(k_text or 0)
    if k_text is not None and k < 1:
        raise ValueError("Ranking cutoff must be positive")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Metric must be a finite number: " + name)
    if metric in RATE_METRICS and not (0 <= value <= 1):
        raise ValueError("Rate metric outside [0, 1]: " + name)
    return metric, k, float(value)


def import_record(db, record):
    """Import one fully specified experiment. Repeated identical imports are idempotent."""
    model, dataset, protocol = record["model"], record["dataset"], record["protocol"]
    for field in ("id", "name", "family", "stage", "maturity"):
        if not str(model.get(field, "")).strip():
            raise ValueError("Model field required: " + field)
    for field in ("id", "name", "version"):
        if not str(dataset.get(field, "")).strip():
            raise ValueError("Dataset field required: " + field)
    if protocol["dataset_id"] != dataset["id"]:
        raise ValueError("Protocol dataset_id must equal dataset id")
    if record["source_type"] not in ("measured", "literature"):
        raise ValueError("source_type must be measured or literature")
    if not record.get("source_uri"):
        raise ValueError("source_uri required for reproducibility or citation")
    if not record.get("evaluation_split"):
        raise ValueError("evaluation_split is required")
    if not record.get("metrics") or not isinstance(record["metrics"], dict):
        raise ValueError("metrics must be a non-empty mapping")
    users = record.get("sample_users")
    if users is not None and (type(users) is not int or users < 0):
        raise ValueError("sample_users must be a nonnegative integer")
    metric_rows = [_metric_parts(name, value) for name, value in record["metrics"].items()]
    if len(set((m, k) for m, k, _ in metric_rows)) != len(metric_rows):
        raise ValueError("Duplicate metric after normalizing names")
    config = record.get("config", {})
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")
    # All user-supplied fields, including metric values, define a content-addressed run.
    run_id = "run-" + _hash(record)[:24]
    with db:
        db.execute("INSERT OR IGNORE INTO models VALUES (?,?,?,?,?)",
                   tuple(str(model[key]) for key in ("id", "name", "family", "stage", "maturity")))
        db.execute("INSERT OR IGNORE INTO datasets VALUES (?,?,?)",
                   tuple(str(dataset[key]) for key in ("id", "name", "version")))
        register_protocol(db, protocol)
        db.execute(
            """INSERT OR IGNORE INTO runs
            (id,model_id,dataset_id,protocol_id,evaluation_split,source_type,source_uri,
             source_sha256,sample_users,seed,git_commit,config_json,notes)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (run_id, model["id"], dataset["id"], protocol["id"], record["evaluation_split"],
             record["source_type"], record["source_uri"], record.get("source_sha256"),
             users, record.get("seed"), record.get("git_commit"), _canonical(config),
             record.get("notes", "")),
        )
        db.executemany("INSERT OR IGNORE INTO metrics VALUES (?,?,?,?)",
                       [(run_id, metric, k, value) for metric, k, value in metric_rows])
    return run_id


def import_baseline(db, path):
    """Read the existing src.train artifacts/metrics.json format without fabricating scores."""
    path = Path(path)
    raw = path.read_bytes()
    result = json.loads(raw)
    if not str(result.get("dataset", "")).startswith("MovieLens latest-small"):
        raise ValueError("Expected MovieLens latest-small baseline metrics.json")
    k = int(result["k"])
    if k <= 0:
        raise ValueError("k must be positive")
    model_map = {row[0]: row for row in MODEL_CATALOG}
    runs = []
    for split_name, field in (("validation", "validation"), ("test", "test")):
        for name, values in result.get(field, {}).items():
            if name not in model_map:
                raise ValueError("Unregistered baseline model: " + name)
            metrics = {}
            for key, normal in (("ndcg_at_k", "ndcg"), ("recall_at_k", "recall"),
                                ("catalog_coverage_at_k", "catalog_coverage")):
                if values.get(key) is not None:
                    metrics[f"{normal}@{k}"] = values[key]
            if not metrics:
                continue
            m = model_map[name]
            rec = {
                "model": dict(zip(("id", "name", "family", "stage", "maturity"), m)),
                "dataset": {"id": "ml-latest-small", "name": "MovieLens latest-small",
                            "version": "latest-small"},
                "protocol": BASELINE_PROTOCOL, "evaluation_split": split_name,
                "source_type": "measured", "source_uri": str(path),
                "source_sha256": hashlib.sha256(raw).hexdigest(),
                "sample_users": values.get("users_evaluated"),
                "config": {"k": k, "positive_rating_threshold": result.get("positive_rating_threshold"),
                           "split": result.get("split"), "counts": result.get("counts"),
                           "catalog_size": result.get("catalog_size"),
                           "timestamp_utc": result.get("timestamp_utc")},
                "notes": result.get("caveat", ""),
                "metrics": metrics,
            }
            runs.append(import_record(db, rec))
    return runs


def leaderboard(db, dataset, protocol, split="test", metric="ndcg", k=10,
                source_type="measured"):
    protocol_row = db.execute("SELECT dataset_id FROM protocols WHERE id=?", (protocol,)).fetchone()
    if not protocol_row or protocol_row["dataset_id"] != dataset:
        raise ValueError("Unknown protocol/dataset combination")
    rows = db.execute(
        """SELECT r.id, m.name AS model, r.model_id, v.value, r.sample_users,
                  r.source_uri, r.git_commit, r.source_type
           FROM runs r JOIN models m ON m.id=r.model_id
           JOIN metrics v ON v.run_id=r.id
           WHERE r.dataset_id=? AND r.protocol_id=? AND r.evaluation_split=?
             AND r.source_type=? AND v.metric=? AND v.k=?
           ORDER BY v.value """ + ("ASC" if metric in ("rmse", "mae", "loss") else "DESC") + ", m.name",
        (dataset, protocol, split, source_type, metric, k),
    ).fetchall()
    return [dict(row) for row in rows]


def export_records(db):
    rows = db.execute(
        """SELECT r.*, m.name AS model_name, d.name AS dataset_name
           FROM runs r JOIN models m ON m.id=r.model_id
           JOIN datasets d ON d.id=r.dataset_id ORDER BY r.id"""
    ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["config"] = json.loads(item.pop("config_json"))
        item["metrics"] = {
            key + ("@" + str(k) if k else ""): val
            for key, k, val in db.execute(
                "SELECT metric,k,value FROM metrics WHERE run_id=? ORDER BY metric,k",
                (item["id"],)).fetchall()
        }
        result.append(item)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="Create schema and known model/dataset catalog")
    imp = sub.add_parser("import-baseline", help="Import src.train metrics.json")
    imp.add_argument("path", type=Path)
    run = sub.add_parser("import-run", help="Import one fully specified JSON run")
    run.add_argument("path", type=Path)
    board = sub.add_parser("leaderboard", help="Compare only identical evaluation protocols")
    board.add_argument("--dataset", required=True)
    board.add_argument("--protocol", required=True)
    board.add_argument("--split", default="test")
    board.add_argument("--metric", default="ndcg")
    board.add_argument("--k", type=int, default=10)
    board.add_argument("--source-type", choices=("measured", "literature"), default="measured")
    sub.add_parser("models")
    sub.add_parser("protocols")
    sub.add_parser("export-json")
    args = parser.parse_args(argv)
    with connect(args.db) as db:
        if args.command == "init":
            print("Initialized", args.db)
        elif args.command == "import-baseline":
            ids = import_baseline(db, args.path)
            print(f"Imported {len(ids)} model/split records ({len(set(ids))} unique):", *ids, sep="\n")
        elif args.command == "import-run":
            print(import_record(db, json.loads(args.path.read_text(encoding="utf-8"))))
        elif args.command == "leaderboard":
            if args.k < 0:
                parser.error("--k must not be negative")
            rows = leaderboard(db, args.dataset, args.protocol, args.split,
                               args.metric, args.k, args.source_type)
            print(json.dumps(rows, indent=2, ensure_ascii=False))
        elif args.command in ("models", "protocols"):
            rows = db.execute("SELECT * FROM " + args.command + " ORDER BY id")
            print(json.dumps([dict(row) for row in rows], indent=2, ensure_ascii=False))
        elif args.command == "export-json":
            print(json.dumps(export_records(db), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
