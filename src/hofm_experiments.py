"""Track HOFM training attempts and link verified evaluation reports.

The launcher executes the research script as a child process without modifying it.
A successful training exit never creates a benchmark score automatically.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import uuid

from .benchmark_registry import DEFAULT_DB, connect, import_record


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _option(args, flag, default=None):
    for index, token in enumerate(args):
        if token == flag:
            if index + 1 >= len(args) or args[index + 1].startswith("--"):
                raise ValueError("Missing value for " + flag)
            return args[index + 1]
        if token.startswith(flag + "="):
            return token.split("=", 1)[1]
    return default


def _load_regularization_config(directory):
    file = directory / "regularization_config.json"
    if not file.is_file():
        return None
    value = json.loads(file.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("regularization_config.json must contain a JSON object")
    return value


def _script_digest(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def run_trainer(db, script, trainer_args, model_id="hofm3_semi25_jointbcd",
                dataset_id="ml-32m"):
    """Run a child training process; record argv/config/status without parsing logs."""
    args = list(trainer_args)
    if args and args[0] == "--":
        args.pop(0)
    if not args or "--self-test" in args or "--help" in args or "-h" in args:
        raise ValueError("Supply actual trainer arguments, not --help or --self-test")
    label = _option(args, "--run-label")
    if not label:
        raise ValueError("The trainer needs --run-label; choose a unique name")
    if not label.replace("_", "").isalnum():
        raise ValueError("Run label must be alphanumeric or underscores")
    script_dataset = _option(args, "--dataset", "ml-32m")
    if script_dataset != dataset_id:
        raise ValueError("Dataset mismatch: trainer --dataset does not match registry --dataset")
    parent = _option(args, "--init-from", "fm_all_hofm3semi25_bpr")
    script = Path(script).resolve()
    if not script.is_file() or script.suffix.lower() != ".py":
        raise FileNotFoundError("Training script not found: " + str(script))
    if not db.execute("SELECT 1 FROM models WHERE id=?", (model_id,)).fetchone():
        raise ValueError("Unknown model id: " + model_id)
    if not db.execute("SELECT 1 FROM datasets WHERE id=?", (dataset_id,)).fetchone():
        raise ValueError("Unknown dataset id: " + dataset_id)

    run_folder = Path("models") / script_dataset / "unified_hofm3_bcd" / f"{parent}_{label}"
    command = [sys.executable, str(script), *args]
    attempt_id = "attempt-" + uuid.uuid4().hex
    git = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True,
                         check=False)
    git_commit = git.stdout.strip() if git.returncode == 0 else None
    config = {
        "trainer_arguments": args,
        "script_sha256": _script_digest(script),
        "checkpoint_directory": str(run_folder),
        "regularization": _load_regularization_config(run_folder),
    }

    with db:
        db.execute(
            """INSERT INTO training_attempts
               (id,model_id,dataset_id,run_label,script_path,command_json,
                working_directory,started_at_utc,status,git_commit,configuration_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (attempt_id, model_id, dataset_id, label, str(script),
             json.dumps(command), str(Path.cwd()), _utc_now(), "running",
             git_commit, json.dumps(config, sort_keys=True)),
        )
    print("Tracking training attempt:", attempt_id, flush=True)
    print("Model:", model_id, "Dataset:", dataset_id, "Label:", label, flush=True)

    status, exit_code = "failed", None
    try:
        result = subprocess.run(command, check=False)
        exit_code = result.returncode
        status = "completed" if exit_code == 0 else "failed"
    except KeyboardInterrupt:
        status = "interrupted"
        exit_code = 130
        raise
    finally:
        # Never silently report an unmeasured NDCG, even for completed jobs.
        config["regularization"] = _load_regularization_config(run_folder)
        with db:
            db.execute(
                """UPDATE training_attempts
                   SET status=?, exit_code=?, finished_at_utc=?, configuration_json=?
                   WHERE id=?""",
                (status, exit_code, _utc_now(), json.dumps(config, sort_keys=True),
                 attempt_id),
            )
        print("Training attempt:", attempt_id, "status:", status,
              "exit code:", exit_code, flush=True)
    return exit_code


def register_existing(db, directory, label, *, model_id="hofm3_semi25_jointbcd",
                      dataset_id="ml-32m"):
    """Register a preexisting checkpoint without claiming a verified training exit."""
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise FileNotFoundError("Existing model directory not found: " + str(directory))
    if not label or not label.replace("_", "").isalnum():
        raise ValueError("Explicit --label must use only letters, digits or underscores")
    if not ((directory / "weights.pt").is_file() or
            (directory / "regularization_config.json").is_file()):
        raise ValueError("Expected weights.pt or regularization_config.json in " + str(directory))
    if not db.execute("SELECT 1 FROM models WHERE id=?", (model_id,)).fetchone():
        raise ValueError("Unknown model id: " + model_id)
    if not db.execute("SELECT 1 FROM datasets WHERE id=?", (dataset_id,)).fetchone():
        raise ValueError("Unknown dataset id: " + dataset_id)
    attempt_id = "imported-" + hashlib.sha256(
        (str(directory) + "\\0" + label + "\\0" + model_id + "\\0" + dataset_id)
        .encode("utf-8")).hexdigest()[:24]
    config = {
        "checkpoint_directory": str(directory),
        "regularization": _load_regularization_config(directory),
        "provenance": "Imported existing artifacts; original command, exit status and Git SHA unknown",
    }
    with db:
        db.execute(
            """INSERT OR IGNORE INTO training_attempts
               (id,model_id,dataset_id,run_label,script_path,command_json,
                working_directory,started_at_utc,finished_at_utc,status,configuration_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (attempt_id, model_id, dataset_id, label, "(historical; unknown)",
             "[]", str(directory.parent), _utc_now(), _utc_now(), "imported",
             json.dumps(config, sort_keys=True)),
        )
    return attempt_id


def attach_evaluation(db, path, *, attempt_id=None, label=None):
    """Link a measured full-protocol report to a completed training attempt."""
    if bool(attempt_id) == bool(label):
        raise ValueError("Exactly one of attempt_id or label is required")
    if attempt_id:
        attempt = db.execute("SELECT * FROM training_attempts WHERE id=?",
                             (attempt_id,)).fetchone()
    else:
        attempt = db.execute(
            """SELECT * FROM training_attempts WHERE run_label=?
               ORDER BY started_at_utc DESC, id DESC LIMIT 1""",
            (label,)).fetchone()
    if attempt is None:
        raise ValueError("Training attempt not found")
    if attempt["status"] not in ("completed", "imported"):
        raise ValueError("Only completed or imported training attempts can have evaluations attached")
    path = Path(path).resolve()
    raw = path.read_bytes()
    record = json.loads(raw)
    if record.get("source_type") != "measured":
        raise ValueError("Training attempts only link to measured evaluations")
    if record["model"]["id"] != attempt["model_id"]:
        raise ValueError("Evaluation model does not match training attempt")
    if record["dataset"]["id"] != attempt["dataset_id"]:
        raise ValueError("Evaluation dataset does not match training attempt")
    if "config" not in record:
        raise ValueError("Evaluation must supply actual training/evaluation config")
    if record.get("git_commit") and attempt["git_commit"] and (
            record["git_commit"] != attempt["git_commit"]):
        raise ValueError("Evaluation git_commit does not match training attempt")
    record["git_commit"] = attempt["git_commit"]
    record["source_uri"] = str(path)
    record["source_sha256"] = hashlib.sha256(raw).hexdigest()
    record.setdefault("notes", "")
    record["notes"] += (" [linked training attempt: " + attempt["id"] + "]")
    benchmark_id = import_record(db, record)
    with db:
        db.execute(
            "INSERT OR IGNORE INTO training_benchmarks VALUES (?,?)",
            (attempt["id"], benchmark_id),
        )
    return benchmark_id


def list_attempts(db, limit=20):
    if limit < 1 or limit > 1000:
        raise ValueError("Limit must be from 1 to 1000")
    rows = db.execute(
        """SELECT a.id, a.run_label, a.model_id, a.dataset_id, a.status,
                  a.exit_code, a.started_at_utc, a.finished_at_utc,
                  count(b.benchmark_id) AS evaluation_count
           FROM training_attempts a LEFT JOIN training_benchmarks b
             ON b.attempt_id=a.id
           GROUP BY a.id ORDER BY a.started_at_utc DESC, a.id DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    return [dict(row) for row in rows]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run trainer and track lifecycle metadata")
    run.add_argument("--db", type=Path, default=DEFAULT_DB)
    run.add_argument("--script", type=Path, default=Path("fast_unified_regularized.py"))
    run.add_argument("--model-id", default="hofm3_semi25_jointbcd")
    run.add_argument("--dataset", default="ml-32m")
    run.add_argument("trainer_args", nargs=argparse.REMAINDER)

    existing = sub.add_parser("register-existing", help="Index a historical checkpoint without rerunning it")
    existing.add_argument("--db", type=Path, default=DEFAULT_DB)
    existing.add_argument("--folder", type=Path, required=True)
    existing.add_argument("--label", required=True)
    existing.add_argument("--model-id", default="hofm3_semi25_jointbcd")
    existing.add_argument("--dataset", default="ml-32m")

    attach = sub.add_parser("attach", help="Attach a measured evaluation JSON report")
    attach.add_argument("--db", type=Path, default=DEFAULT_DB)
    selector = attach.add_mutually_exclusive_group(required=True)
    selector.add_argument("--attempt-id")
    selector.add_argument("--label")
    attach.add_argument("--report", type=Path, required=True)

    listing = sub.add_parser("list", help="List tracked training attempts")
    listing.add_argument("--db", type=Path, default=DEFAULT_DB)
    listing.add_argument("--limit", type=int, default=20)

    args = parser.parse_args(argv)
    try:
        with connect(args.db) as db:
            if args.command == "run":
                code = run_trainer(db, args.script, args.trainer_args,
                                   model_id=args.model_id, dataset_id=args.dataset)
                return code
            if args.command == "register-existing":
                print("Registered existing checkpoint:", register_existing(
                    db, args.folder, args.label, model_id=args.model_id, dataset_id=args.dataset))
            if args.command == "attach":
                print("Linked benchmark:", attach_evaluation(
                    db, args.report, attempt_id=args.attempt_id, label=args.label))
            if args.command == "list":
                print(json.dumps(list_attempts(db, args.limit), indent=2))
    except (ValueError, FileNotFoundError, KeyError, sqlite3.DatabaseError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
