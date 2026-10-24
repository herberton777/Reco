"""JSON-first research studio inventory, comparability and training-log utilities.

No SQLite, training, network or destructive filesystem operations. Existing
checkpoints and reports are only inspected; numbers come from actual files.
"""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import re

import pandas as pd

from .reco_visual_data import read_ranking_json


MODEL_CATALOG = [
    ("popularity", "Popularity / temporal popularity", "Collaborative baseline"),
    ("svd", "TruncatedSVD / chronological SVD", "Collaborative baseline"),
    ("bpr", "BPR-MF / pairwise retrieval", "Collaborative / retrieval"),
    ("fm_all_bpr", "FM all-features BPR", "Factorization machines"),
    ("fm_all_l2try_bpr", "FM all-features L2 variant", "Factorization machines"),
    ("deepfm", "DeepFM", "Factorization machines"),
    ("hofm3_semi25_jointbcd", "HOFM3 Semi25 + JointBCD", "Higher-order / temporal"),
    ("hofm4_jointbcd", "HOFM4 + JointBCD", "Higher-order / temporal"),
    ("temporal_fm", "Temporal FM B + C + D", "Higher-order / temporal"),
    ("semantic", "Sentence Transformer semantic retrieval", "Content / semantic"),
    ("two_tower", "Neural two-tower", "Neural retrieval"),
    ("lambdarank", "LambdaRank / LambdaMART", "Learning to rank"),
    ("tiger", "TIGER-inspired generative retrieval", "Research prototype"),
    ("sequence", "Sequential / Transformer variants", "Research prototype"),
    ("slate", "Slate / multi-objective ranking", "Research prototype"),
]

DATASET_LOCATIONS = {
    "MovieLens latest-small": [
        "data/ml-latest-small", "data/raw/ml-latest-small",
        "data/ml-latest-small.csv",
    ],
    "MovieLens 1M": ["data/raw/ml-1m", "data/temporal/ml-1m", "data/ml-1m"],
    "MovieLens 32M": ["data/raw/ml-32m", "data/temporal/ml-32m", "data/ml-32m"],
}

REPORT_LIMIT_BYTES = 8_000_000
CHECKPOINT_MARKERS = {
    "weights.pt", "checkpoint.pt", "model.pt", "best.pt",
    "weights.pth", "model.pth", "checkpoint.pth",
    "recommender.joblib", "regularization_config.json",
    "config.json", "training_summary.json",
}
METRIC_PATTERN = re.compile(
    r"(ndcg|recall|precision|mrr|map|hit_rate|catalog_coverage|coverage|diversity)"
    r"\s*(?:@\s*|_at_|at_)(\d+)",
    flags=re.IGNORECASE,
)
EPOCH_PATTERN = re.compile(r"\bepoch\s*[:=#-]?\s*(\d+)(?:\s*/\s*\d+)?\b", re.I)
LOG_METRIC_PATTERN = re.compile(
    r"\b(?P<key>(?:val(?:idation)?[_\s.-]*)?"
    r"(?:ndcg|recall|hit_rate|precision|mrr|loss|bpr_loss)"
    r"(?:\s*@\s*\d+|_at_\d+)?)"
    r"\s*[:=]\s*(?P<value>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
    flags=re.IGNORECASE,
)


def dataset_inventory(root="."):
    root = Path(root)
    items = []
    for name, candidates in DATASET_LOCATIONS.items():
        locations = [
            str(root / p) for p in candidates if (root / p).exists()
        ]
        selected = next((root / p for p in candidates
                         if (root / p).is_dir() and
                             (root / p / "movies.csv").is_file()), None)
        items.append({
            "Dataset": name, "Found": bool(locations),
            "Locations": ", ".join(locations) or "Not found",
            "Movie titles file": str(selected / "movies.csv") if selected else "Unknown",
            "Tags available": bool(selected and (selected / "tags.csv").is_file()),
        })
    return items


def _report_files(root="artifacts", max_files=500):
    folder = Path(root)
    if not folder.is_dir():
        return []
    paths = []
    for path in folder.rglob("*.json"):
        if len(paths) >= max_files:
            break
        try:
            if path.is_file() and 0 < path.stat().st_size <= REPORT_LIMIT_BYTES:
                paths.append(path)
        except OSError:
            continue
    return sorted(paths, key=lambda p: str(p))


def discover_reports(root="artifacts", max_files=500):
    """Return only recognized report documents, not checkpoint config JSON."""
    reports = []
    for path in _report_files(root, max_files=max_files):
        try:
            kind, payload = read_ranking_json(path)
        except (OSError, ValueError, json.JSONDecodeError, KeyError, TypeError):
            continue
        reports.append((path, kind, payload))
    return reports


def _positive_number(value):
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _metric_values(values):
    return {
        str(key): float(value) for key, value in values.items()
        if _positive_number(value) and (
            METRIC_PATTERN.fullmatch(str(key)) or
            str(key) in {"ndcg_at_k", "recall_at_k", "mrr_at_k",
                         "precision_at_k", "hit_rate_at_k",
                         "catalog_coverage_at_k", "diversity_at_k",
                         "rmse", "mae"})
    }


def comparable_records(reports):
    """Flatten genuinely measured metrics, scoped to exact reported protocol.

    Weak/partial provenance: restrict comparisons to one JSON file rather
    than joining seemingly similar labels from different experiments.
    """
    rows = []
    for path, kind, payload in reports:
        path_string = str(path)
        if kind == "baseline":
            # A shared baseline JSON is comparable *internally*, but not to
            # HOFM/32M reports merely sharing a metric name.
            scope = path_string
            for split in ("validation", "test"):
                source = payload.get(split, {})
                if not isinstance(source, dict):
                    continue
                for name, values in source.items():
                    if not isinstance(values, dict):
                        continue
                    for metric, value in _metric_values(values).items():
                        rows.append({
                            "Dataset": str(payload.get("dataset", "Unknown")),
                            "Protocol": "latest-small per-user baseline (within one JSON)",
                            "Split": split, "Provenance": "measured",
                            "Scope": scope + "|" + split, "Model": str(name),
                            "Metric": metric, "Score": value,
                            "JSON": path_string,
                        })
        elif kind == "evaluation":
            model = payload.get("model", {})
            model = model if isinstance(model, dict) else {"name": str(model)}
            protocol = payload.get("protocol", {})
            protocol = protocol if isinstance(protocol, dict) else {}
            dataset = payload.get("dataset", {})
            dataset = dataset.get("id", "Unknown") if isinstance(dataset, dict) else dataset
            split = str(payload.get("evaluation_split", "unknown"))
            origin = str(payload.get("source_type", "unknown"))
            protocol_id = str(protocol.get("id", ""))
            keys = ("dataset_id", "split_method", "candidate_policy",
                    "label_rule", "cohort_signature", "cutoff")
            # A shared protocol ID is insufficient for joining potentially
            # different user cohorts and sampled-negative candidate pools.
            if protocol_id and all(protocol.get(k) for k in keys):
                identity = json.dumps({k: protocol[k] for k in ("id", *keys)},
                                      sort_keys=True)
                scope = str(dataset) + "|" + identity + "|" + split + "|" + origin
            else:
                scope = path_string + "|" + split  # keep isolated
            metrics = payload.get("metrics", {})
            for metric, value in _metric_values(metrics).items():
                rows.append({
                    "Dataset": str(dataset), "Protocol": protocol_id or "Undocumented",
                    "Split": split, "Provenance": origin, "Scope": scope,
                    "Model": str(model.get("name") or model.get("id") or "Unknown"),
                    "Metric": metric, "Score": value, "JSON": path_string,
                })
        elif kind == "top_models":
            # The complete model/fusion/reranker ablation is comparable
            # *within this same JSON*. Do not pool with unrelated runs.
            audit = payload.get("audit", {})
            dataset = audit.get("dataset", {})
            dataset_name = dataset.get("id", "Unknown") if isinstance(dataset, dict) else "Unknown"
            protocol = audit.get("protocol", {})
            protocol_name = protocol.get("id", "Undocumented") if isinstance(protocol, dict) else "Undocumented"
            for variant in payload["variants"]:
                if not isinstance(variant, dict):
                    continue
                for split in ("validation", "test"):
                    metrics = variant.get(split, {})
                    if not isinstance(metrics, dict):
                        continue
                    for metric, score in _metric_values(metrics).items():
                        rows.append({
                            "Dataset": str(dataset_name), "Protocol": str(protocol_name),
                            "Split": split, "Provenance": "measured",
                            "Scope": path_string + "|" + split,
                            "Model": str(variant.get("variant", "Unknown")),
                            "Metric": metric, "Score": score, "JSON": path_string,
                        })
        elif kind == "rerank_summary":
            manifest = payload.get("manifest", {})
            manifest = manifest if isinstance(manifest, dict) else {}
            dataset = manifest.get("dataset", {})
            dataset = dataset.get("id", "Unknown") if isinstance(dataset, dict) else dataset
            protocol = manifest.get("protocol", {})
            protocol_id = (protocol.get("id", "Undocumented")
                           if isinstance(protocol, dict) else "Undocumented")
            for variant, value in payload["variants"].items():
                if not isinstance(value, dict):
                    continue
                for split in ("validation", "test"):
                    metrics = value.get(split, {})
                    if not isinstance(metrics, dict):
                        continue
                    for metric, score in _metric_values(metrics).items():
                        rows.append({
                            "Dataset": str(dataset), "Protocol": str(protocol_id),
                            "Split": split, "Provenance": "measured",
                            "Scope": path_string + "|" + split,
                            "Model": "LambdaRank " + str(variant),
                            "Metric": metric, "Score": score,
                            "JSON": path_string,
                        })
                    if split == "test":
                        for metric, score in _metric_values(
                                value.get("baseline_test", {})).items():
                            rows.append({
                                "Dataset": str(dataset), "Protocol": str(protocol_id),
                                "Split": split, "Provenance": "measured",
                                "Scope": path_string + "|" + split,
                                "Model": str(variant) + " (base)",
                                "Metric": metric, "Score": score,
                                "JSON": path_string,
                            })
    return pd.DataFrame(rows, columns=[
        "Dataset", "Protocol", "Split", "Provenance", "Scope", "Model",
        "Metric", "Score", "JSON",
    ])


def discover_logs(root=".", max_files=250):
    """Include user research logs, BCD scripts and new training_logs folder."""
    root = Path(root)
    candidates = set(root.glob("*.log"))
    for sub in ("artifacts/training_logs", "artifacts/logs", "logs"):
        folder = root / sub
        if folder.is_dir():
            candidates.update(folder.glob("*.log"))
            candidates.update(folder.glob("*.txt"))
    paths = []
    for path in sorted(candidates):
        try:
            if path.is_file() and path.stat().st_size < 100_000_000:
                paths.append(path)
        except OSError:
            pass
    return paths[:max_files]


def parse_training_log(text, max_lines=200000):
    """Extract only *explicitly named* epoch and metric values from real log lines."""
    rows = []
    current_epoch = None
    for number, line in enumerate(text.splitlines()[:max_lines], start=1):
        m = EPOCH_PATTERN.search(line)
        if m:
            current_epoch = int(m.group(1))
        metrics = {}
        for match in LOG_METRIC_PATTERN.finditer(line):
            name = re.sub(r"\s+", "", match.group("key")).lower()
            try:
                value = float(match.group("value"))
            except ValueError:
                continue
            if math.isfinite(value):
                metrics[name] = value
        if not metrics:
            continue
        if current_epoch is None:
            continue  # Never assign metrics to a guessed epoch.
        for name, value in metrics.items():
            rows.append({
                "Epoch": current_epoch, "Metric": name, "Value": value,
                "Line": number,
            })
    return pd.DataFrame(rows, columns=["Epoch", "Metric", "Value", "Line"])


def checkpoint_inventory(root=".", max_rows=400):
    """Inventory actual local checkpoint markers without loading untrusted weights."""
    root = Path(root)
    bases = [root / "models", root / "artifacts"]
    rows = []
    for base in bases:
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.name not in CHECKPOINT_MARKERS or not path.is_file():
                continue
            if len(rows) >= max_rows:
                return rows
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if path.name == "config.json" and "semantic" in str(path).lower():
                continue
            rows.append({
                "Directory": str(path.parent),
                "Artifact": path.name,
                "Size (MiB)": round(size / (1024 ** 2), 3),
                "Type": "Weights / saved recommender"
                        if path.suffix in (".pt", ".pth", ".joblib")
                        else "Configuration",
            })
    return sorted(rows, key=lambda x: (x["Directory"], x["Artifact"]))


def training_configuration(folder):
    folder = Path(folder)
    for basename in ("regularization_config.json", "training_summary.json",
                     "config.json"):
        path = folder / basename
        if path.is_file() and path.stat().st_size < 2_000_000:
            try:
                return basename, json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
    return None, None


def feature_importances(payload):
    """Use only feature importance values saved by a real trained reranker."""
    objects = [payload]
    config = payload.get("config")
    if isinstance(config, dict):
        objects.append(config)
    for obj in objects:
        for key in ("feature_importances", "feature_importance",
                    "feature_importance_gain", "importance"):
            values = obj.get(key)
            if isinstance(values, dict):
                rows = [{"Feature": str(k), "Importance": float(v)}
                        for k, v in values.items() if _positive_number(v)]
                return pd.DataFrame(rows, columns=["Feature", "Importance"])
            if isinstance(values, list) and all(_positive_number(v) for v in values):
                feature_names = obj.get("features")
                if not isinstance(feature_names, list) and isinstance(config, dict):
                    feature_names = config.get("features")
                if isinstance(feature_names, list) and len(feature_names) == len(values):
                    return pd.DataFrame({
                        "Feature": [str(n) for n in feature_names],
                        "Importance": list(map(float, values)),
                    })
    return pd.DataFrame(columns=["Feature", "Importance"])
