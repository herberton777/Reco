"""Standalone read-only Streamlit explorer for the Reco benchmark registry.

Run: streamlit run benchmark_dashboard.py --server.address 0.0.0.0 --server.port 8502
Set RECO_BENCHMARK_DB to a trusted local SQLite path if using another location.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from src.benchmark_dashboard_data import (
    DashboardDataError,
    available_metrics,
    chart_rows,
    leaderboard_rows,
    load_snapshot,
    metric_label,
    protocol_runs,
    training_history,
)

st.set_page_config(page_title="Reco · Benchmark Lab", page_icon="📊", layout="wide")
st.title("📊 Reco · Benchmark Lab")
st.caption("Movie recommendation research · reproducible experiments · measured results only")

DB_PATH = Path(os.environ.get("RECO_BENCHMARK_DB", "artifacts/benchmarks.sqlite"))
st.sidebar.header("Benchmark database")
st.sidebar.caption("Read-only SQLite · data remains on the machine running Streamlit")
st.sidebar.code(str(DB_PATH), language=None)
if st.sidebar.button("Refresh database", use_container_width=True):
    st.rerun()

try:
    snapshot = load_snapshot(DB_PATH)
except DashboardDataError as exc:
    st.error(str(exc))
    st.markdown("Open a compatible database created by the benchmark registry.")
    st.stop()

if snapshot is None:
    st.info("No local benchmark database was found. The dashboard doesn't create data or scores.")
    st.code(
        "python -m src.benchmark_registry init\n"
        "python -m src.benchmark_registry models\n"
        "python -m src.hofm_experiments list",
        language="bash",
    )
    st.caption("Then refresh this page. For actual scores, import an evaluation report.")
    st.stop()

runs = snapshot["runs"]
attempts = snapshot["training_attempts"]
datasets = snapshot["datasets"]
models = snapshot["models"]
protocols = snapshot["protocols"]
linked_ids = {item["benchmark_id"] for item in snapshot["training_benchmarks"]}

top = st.columns(4)
top[0].metric("Cataloged models", len(models))
top[1].metric("Evaluated runs", len(runs))
top[2].metric("Training attempts", len(attempts))
top[3].metric("Stored metric values", len(snapshot["metrics"]))

st.caption(
    "Model catalog entries are not experiments. Training attempts are not ranking scores. "
    "Each leaderboard compares only the same dataset, protocol, split, source type, metric and K."
)
leader_tab, experiments_tab, protocols_tab = st.tabs(
    ["Leaderboard & charts", "Experiment history", "Protocols & catalog"]
)

with leader_tab:
    st.subheader("Protocol-controlled leaderboard")
    if not runs:
        st.info(
            "No measured or literature evaluation runs have been imported yet. "
            "The model catalog is populated, but there are no valid NDCG/Recall results to plot."
        )
        st.code(
            "python -m src.benchmark_registry import-baseline artifacts/metrics.json",
            language="bash",
        )
        st.caption("For MovieLens 32M, attach a real evaluation report via src.hofm_experiments.")
    else:
        counts = {}
        for item in runs:
            counts[item["dataset_id"]] = counts.get(item["dataset_id"], 0) + 1
        ordered_datasets = sorted(counts, key=lambda key: (-counts[key], key))
        dataset_names = {d["id"]: d["name"] for d in datasets}
        selection = st.columns([1, 2])
        with selection[0]:
            dataset_id = st.selectbox(
                "Dataset",
                ordered_datasets,
                format_func=lambda key: dataset_names.get(key, key),
            )
        candidate_protocols = [p for p in protocols if p["dataset_id"] == dataset_id
                               and any(r["protocol_id"] == p["id"] for r in runs)]
        with selection[1]:
            protocol_id = st.selectbox(
                "Evaluation protocol",
                [p["id"] for p in candidate_protocols],
            )
        protocol = next(p for p in candidate_protocols if p["id"] == protocol_id)
        with st.expander("Evaluation rules · read before comparing", expanded=False):
            st.write("**Split:**", protocol["split_method"])
            st.write("**Candidates:**", protocol["candidate_policy"])
            st.write("**Labels:**", protocol["label_rule"])
            st.write("**Cohort:**", protocol["cohort_signature"])
            st.write("**Time cutoff:**", protocol["cutoff"])

        scoped = [r for r in runs if r["dataset_id"] == dataset_id
                  and r["protocol_id"] == protocol_id]
        controls = st.columns([1, 1, 1])
        with controls[0]:
            split = st.selectbox(
                "Evaluation split", sorted({r["evaluation_split"] for r in scoped}),
                index=0,
            )
        matching_split = [r for r in scoped if r["evaluation_split"] == split]
        source_types = sorted({r["source_type"] for r in matching_split})
        with controls[1]:
            source_type = st.selectbox(
                "Result provenance", source_types,
                index=source_types.index("measured") if "measured" in source_types else 0,
            )
        comparable_runs = protocol_runs(snapshot, dataset_id, protocol_id, split, source_type)
        metric_pairs = available_metrics(snapshot, comparable_runs)
        with controls[2]:
            chosen = st.selectbox(
                "Ranking metric and cutoff",
                metric_pairs,
                format_func=lambda pair: metric_label(*pair),
            )

        if chosen is not None:
            best_only = st.toggle(
                "Show best individual run per model", value=True,
                help="Selects a single best recorded run for each model. "
                     "It does not average results across runs.",
            )
            ranking = leaderboard_rows(
                snapshot, comparable_runs, chosen[0], chosen[1],
                best_per_model=best_only,
            )
            if ranking:
                st.caption(
                    f"{len(comparable_runs)} recorded runs · "
                    f"{len(ranking)} entries displayed · "
                    f"{metric_label(*chosen)} · "
                    + ("lower is better" if chosen[0] in
                       {"rmse", "mae", "loss", "mse", "logloss"} else "higher is better")
                )
                ranked_df = pd.DataFrame(ranking).rename(columns={
                    "run_id": "Run ID", "model": "Model", "value": metric_label(*chosen),
                    "users": "Users", "git_commit": "Commit", "source_uri": "Evidence",
                    "source_type": "Provenance", "model_id": "Model ID",
                })
                chart_df = ranked_df.copy()
                chart_df["Model / run"] = (
                    chart_df["Model"] + " · " + chart_df["Run ID"].str[-6:]
                )
                st.bar_chart(
                    chart_df.set_index("Model / run")[[metric_label(*chosen)]],
                    use_container_width=True,
                )
                st.dataframe(
                    ranked_df, hide_index=True, use_container_width=True,
                )
                st.download_button(
                    "Export current leaderboard (CSV)",
                    data=ranked_df.to_csv(index=False).encode("utf-8"),
                    file_name="reco_leaderboard.csv", mime="text/csv",
                )
                sizes = {row["users"] for row in ranking}
                if len(sizes) > 1:
                    st.warning(
                        "These runs have different numbers of evaluated users. "
                        "Confirm the cohort and exclusions match before interpreting ranks."
                    )
                st.caption(
                    "Offline scores are protocol-specific. 'Best' uses the highest "
                    "(or lowest for error metrics) observed *single run* and "
                    "does not establish statistical significance."
                )
            else:
                st.info("No scores are stored for this metric and scope.")

            st.divider()
            st.subheader("Compare metrics across the same evaluated runs")
            default_pairs = [
                pair for pair in metric_pairs
                if pair[1] == chosen[1] and pair[0] in ("ndcg", "recall")
            ]
            if not default_pairs:
                default_pairs = [chosen]
            selected_pairs = st.multiselect(
                "Metrics to chart (K is shown explicitly)",
                metric_pairs, default=default_pairs,
                format_func=lambda pair: metric_label(*pair),
            )
            if selected_pairs:
                values = chart_rows(snapshot, comparable_runs, selected_pairs)
                if values:
                    frame = pd.DataFrame(values)
                    columns = [metric_label(*m) for m in selected_pairs]
                    st.bar_chart(
                        frame.set_index("Run")[columns],
                        use_container_width=True,
                    )
                    st.dataframe(
                        frame, hide_index=True, use_container_width=True,
                    )
                else:
                    st.info("No values are available for the selected metrics.")

with experiments_tab:
    st.subheader("Training history and evaluated runs")
    dataset_names = {d["id"]: d["name"] for d in datasets}
    history_datasets = ["All datasets"] + [d["id"] for d in datasets]
    history_dataset = st.selectbox(
        "Filter history by dataset", history_datasets,
        format_func=lambda key: dataset_names.get(key, key),
        key="history_dataset",
    )
    filtered_attempts = training_history(
        snapshot, dataset_id=None if history_dataset == "All datasets" else history_dataset
    )
    if filtered_attempts:
        status_options = sorted({a["status"] for a in filtered_attempts})
        statuses = st.multiselect(
            "Training status", status_options, default=status_options,
        )
        filtered_attempts = [a for a in filtered_attempts if a["status"] in statuses]
        attempts_df = pd.DataFrame([
            {
                "Label": a["run_label"], "Model": a["model_name"],
                "Dataset": a["dataset_name"], "Status": a["status"],
                "Evaluations": a["evaluation_count"],
                "Started (UTC)": a["started_at_utc"],
                "Exit code": a["exit_code"], "Attempt ID": a["id"],
            } for a in filtered_attempts
        ])
        st.dataframe(attempts_df, hide_index=True, use_container_width=True)
        if filtered_attempts:
            by_id = {a["id"]: a for a in filtered_attempts}
            chosen_attempt = st.selectbox(
                "Inspect training attempt",
                list(by_id),
                format_func=lambda key: by_id[key]["run_label"] + " · " + key[-8:],
            )
            attempt = by_id[chosen_attempt]
            st.write("**Execution status:**", attempt["status"])
            if attempt["status"] == "imported":
                st.caption(
                    "Imported checkpoint: original training outcome and command are unknown."
                )
            st.write("**Checkpoint / configuration details**")
            try:
                st.json(json.loads(attempt["configuration_json"]))
            except (ValueError, TypeError):
                st.code(str(attempt["configuration_json"]))
            with st.expander("Recorded execution command"):
                try:
                    st.code(" ".join(json.loads(attempt["command_json"])), language="bash")
                except (ValueError, TypeError):
                    st.code(str(attempt["command_json"]), language="text")
            linked = [edge["benchmark_id"] for edge in snapshot["training_benchmarks"]
                      if edge["attempt_id"] == chosen_attempt]
            if linked:
                st.caption("Attached evaluated run IDs: " + ", ".join(linked))
            else:
                st.info("No measured evaluation report has been linked to this attempt.")
    else:
        st.info("No training attempts have been registered yet.")
        st.code(
            "python -m src.hofm_experiments list\n"
            "python -m src.hofm_experiments register-existing --help",
            language="bash",
        )

    st.divider()
    st.subheader("Evaluated experiment records")
    history_runs = [r for r in runs if
                    history_dataset == "All datasets" or
                    r["dataset_id"] == history_dataset]
    if history_runs:
        detail_df = pd.DataFrame([
            {
                "Run ID": r["id"], "Model": r["model_name"],
                "Dataset": r["dataset_name"], "Protocol": r["protocol_id"],
                "Split": r["evaluation_split"], "Source": r["source_type"],
                "Users": r["sample_users"],
                "Linked to training": r["id"] in linked_ids,
            } for r in history_runs
        ])
        st.dataframe(detail_df, hide_index=True, use_container_width=True)
        by_id = {r["id"]: r for r in history_runs}
        chosen_run = st.selectbox(
            "Inspect evaluated run", list(by_id),
            format_func=lambda key: by_id[key]["model_name"] + " · " + key[-8:],
        )
        selected_run = by_id[chosen_run]
        metric_data = {
            metric_label(m["metric"], m["k"]): m["value"]
            for m in snapshot["metrics"] if m["run_id"] == chosen_run
        }
        left, right = st.columns(2)
        with left:
            st.write("**Measured metrics**")
            st.json(metric_data)
            st.write("**Source:**", selected_run["source_uri"])
            st.write("**Git commit:**", selected_run["git_commit"] or "Not recorded")
            st.write("**Source SHA256:**", selected_run["source_sha256"] or "Not recorded")
        with right:
            st.write("**Hyperparameters / configuration**")
            try:
                st.json(json.loads(selected_run["config_json"]))
            except (ValueError, TypeError):
                st.code(str(selected_run["config_json"]))
        if selected_run["notes"]:
            st.caption("Research notes: " + selected_run["notes"])
    else:
        st.info("There are no evaluated runs for the selected dataset.")

with protocols_tab:
    st.subheader("Datasets and evaluation protocols")
    st.dataframe(pd.DataFrame(datasets), hide_index=True, use_container_width=True)
    st.dataframe(pd.DataFrame(protocols), hide_index=True, use_container_width=True)
    st.caption(
        "NDCG computed with sampled negatives must never be compared directly "
        "to NDCG over the full unseen-item catalog."
    )
    st.divider()
    st.subheader("Model catalog")
    st.caption("Listed models are registered definitions, not claims of trained checkpoints.")
    st.dataframe(pd.DataFrame(models), hide_index=True, use_container_width=True)
    st.markdown(
        "Methodology: [benchmark registry](https://github.com/herberton777/Reco/blob/main/"
        "docs/BENCHMARK_REGISTRY.md) · "
        "[HOFM experiment tracking](https://github.com/herberton777/Reco/blob/main/"
        "docs/HOFM_EXPERIMENT_TRACKING.md)"
    )
