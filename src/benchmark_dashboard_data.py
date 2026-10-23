"""Read-only SQLite queries for the Streamlit benchmark dashboard.

No Streamlit dependency: keep database access and comparability rules testable.
The dashboard never initializes, imports, trains, or changes a database.
"""
from __future__ import annotations

from pathlib import Path
import sqlite3

REQUIRED_TABLES = frozenset({"models", "datasets", "protocols", "runs", "metrics"})
LOWER_IS_BETTER = frozenset({"rmse", "mae", "loss", "mse", "logloss"})
METRIC_PRIORITY = {"ndcg": 0, "recall": 1, "precision": 2, "map": 3,
                   "mrr": 4, "hit_rate": 5, "catalog_coverage": 6}
class DashboardDataError(ValueError):
    """Database absent, malformed, or incompatible with the registry."""


def metric_label(name: str, k: int) -> str:
    """Display the exact metric cutoff; a cutoff of zero means none."""
    return name.upper() + (f"@{k}" if k else "")


def _fetch(db, query):
    return [dict(row) for row in db.execute(query).fetchall()]


def load_snapshot(path):
    """Read one SQLite database without creating or mutating anything.

    Returns None for a missing database, raising DashboardDataError on corrupt or
    incompatible files. Optional training tables support older registry versions.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        return None
    try:
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA query_only = ON")
            tables = {r[0] for r in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            missing = REQUIRED_TABLES - tables
            if missing:
                raise DashboardDataError(
                    "Not a compatible benchmark database; missing: " +
                    ", ".join(sorted(missing)))
            data = {
                "datasets": _fetch(db, "SELECT * FROM datasets ORDER BY name"),
                "models": _fetch(db, "SELECT * FROM models ORDER BY name"),
                "protocols": _fetch(db, "SELECT * FROM protocols ORDER BY id"),
                "runs": _fetch(db, """
                    SELECT r.*, m.name AS model_name, m.family AS model_family,
                           m.stage AS model_stage, d.name AS dataset_name
                    FROM runs r JOIN models m ON m.id = r.model_id
                    JOIN datasets d ON d.id = r.dataset_id
                    ORDER BY r.id
                """),
                "metrics": _fetch(db, """
                    SELECT run_id, metric, k, value FROM metrics
                    ORDER BY run_id, metric, k
                """),
                "training_attempts": [],
                "training_benchmarks": [],
            }
            if "training_attempts" in tables:
                data["training_attempts"] = _fetch(db, """
                    SELECT t.*, m.name AS model_name, d.name AS dataset_name
                    FROM training_attempts t
                    JOIN models m ON m.id=t.model_id
                    JOIN datasets d ON d.id=t.dataset_id
                    ORDER BY t.started_at_utc DESC, t.id DESC
                """)
            if "training_benchmarks" in tables:
                data["training_benchmarks"] = _fetch(
                    db, "SELECT attempt_id, benchmark_id FROM training_benchmarks")
            return data
    except (sqlite3.Error, OSError) as exc:
        raise DashboardDataError("Cannot read benchmark database: " + str(exc)) from exc


def protocol_runs(snapshot, dataset_id, protocol_id, split, source_type):
    """Only one dataset, protocol, split and source type are comparable."""
    return [
        run for run in snapshot["runs"]
        if run["dataset_id"] == dataset_id
        and run["protocol_id"] == protocol_id
        and run["evaluation_split"] == split
        and run["source_type"] == source_type
    ]


def available_metrics(snapshot, runs):
    ids = {run["id"] for run in runs}
    pairs = {
        (m["metric"], m["k"]) for m in snapshot["metrics"]
        if m["run_id"] in ids
    }
    return sorted(pairs, key=lambda mk: (
        METRIC_PRIORITY.get(mk[0], 100), mk[0], mk[1]))


def leaderboard_rows(snapshot, runs, metric, k, *, best_per_model=True):
    """Return one scored row/run; optional best-per-model aggregation.

    This function never pools metrics or users across runs. "Best" shows the
    best *single* run per model, without computing averages across cohorts.
    """
    by_run = {r["id"]: r for r in runs}
    rows = []
    for value in snapshot["metrics"]:
        if (value["run_id"] in by_run
                and value["metric"] == metric and value["k"] == k):
            run = by_run[value["run_id"]]
            rows.append({
                "run_id": run["id"],
                "model_id": run["model_id"],
                "model": run["model_name"],
                "value": value["value"],
                "users": run["sample_users"],
                "git_commit": run["git_commit"],
                "source_uri": run["source_uri"],
                "source_type": run["source_type"],
            })
    increasing = metric in LOWER_IS_BETTER
    rows.sort(key=lambda r: (r["value"] if increasing else -r["value"],
                             r["model"].lower(), r["run_id"]))
    if best_per_model:
        seen = set()
        selected = []
        for row in rows:
            if row["model_id"] not in seen:
                selected.append(row)
                seen.add(row["model_id"])
        rows = selected
    return rows


def chart_rows(snapshot, runs, metrics):
    """A table of individual runs, with selected metric columns (never averaged).

    The selection is already protocol-scoped by protocol_runs().
    """
    included = {run["id"]: run for run in runs}
    lookup = {
        (m["run_id"], m["metric"], m["k"]): m["value"]
        for m in snapshot["metrics"] if m["run_id"] in included
    }
    result = []
    for run in runs:
        row = {
            "Run": run["model_name"] + " · " + run["id"][-6:],
            "Model": run["model_name"],
            "Run ID": run["id"],
            "Users": run["sample_users"],
        }
        has_value = False
        for name, k in metrics:
            value = lookup.get((run["id"], name, k))
            row[metric_label(name, k)] = value
            has_value = has_value or value is not None
        if has_value:
            result.append(row)
    return result


def training_history(snapshot, *, dataset_id=None):
    links = {}
    for item in snapshot["training_benchmarks"]:
        links.setdefault(item["attempt_id"], set()).add(item["benchmark_id"])
    rows = []
    for attempt in snapshot["training_attempts"]:
        if dataset_id is not None and attempt["dataset_id"] != dataset_id:
            continue
        item = dict(attempt)
        item["evaluation_count"] = len(links.get(attempt["id"], ()))
        rows.append(item)
    return rows
