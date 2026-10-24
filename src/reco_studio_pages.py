"""Broad JSON-first Reco research studio pages.

No database setup and no automatic training/checkpoint mutations.
Everything displayed is based on existing model files, JSON or logs.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from .reco_studio_data import (
    MODEL_CATALOG, checkpoint_inventory, comparable_records, dataset_inventory,
    discover_logs, discover_reports, feature_importances, parse_training_log,
    training_configuration,
)
from .reco_visual_data import (
    metric_frame, sample_umap_source, umap_projection, pca_projection,
)


SEMANTIC_CACHE = Path("artifacts/semantic/ml32m_minilm_full_v1")
SEMANTIC_CACHE_1M = Path("artifacts/semantic/ml1m_minilm_full_v1")
DATA_MOVIES = [
    Path("data/raw/ml-32m/movies.csv"),
    Path("data/temporal/ml-32m/movies.csv"),
    Path("data/ml-32m/movies.csv"),
]


@st.cache_data(show_spinner=False)
def _movie_catalog(file_path, modification):
    return pd.read_csv(file_path, usecols=["movieId", "title", "genres"])


def _reports():
    return discover_reports("artifacts")


def _checkbox_frame(rows, empty):
    if not rows:
        st.info(empty)
        return
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _report_location_info():
    st.caption(
        "All comparisons originate in real JSON reports under artifacts/. "
        "No SQLite connection is used. A checkpoint is not proof of measured NDCG."
    )


def show_overview(engine):
    st.header("Research overview")
    st.caption(
        "One place for movie recommendation, dataset coverage, actual JSON "
        "evaluations, training logs and saved checkpoints."
    )
    reports = _reports()
    logs = discover_logs()
    checkpoints = checkpoint_inventory()
    datasets = dataset_inventory()
    a, b, c, d = st.columns(4)
    a.metric("JSON reports", len(reports))
    b.metric("Training logs", len(logs))
    c.metric("Checkpoint/config artifacts", len(checkpoints))
    d.metric("Datasets detected", sum(int(row["Found"]) for row in datasets))
    st.subheader("Data inventory")
    _checkbox_frame(datasets, "No MovieLens datasets found under known paths.")
    st.subheader("Available recommendation families")
    cat = pd.DataFrame(MODEL_CATALOG, columns=["Model ID", "Algorithm", "Family"])
    cat["Status"] = "Catalog entry only"
    if engine is not None:
        names = set(engine.models)
        cat.loc[cat["Model ID"].isin(names), "Status"] = "Baseline trained artifact"
    st.dataframe(cat, use_container_width=True, hide_index=True)
    st.caption(
        "The table is an algorithm catalog, not a list of measured models. "
        "Advanced HOFM/DeepFM/BPR research prototypes require real checkpoints "
        "and compatible inference adapters."
    )
    if reports:
        values = comparable_records(reports)
        if not values.empty:
            st.subheader("Reported metrics at a glance")
            st.dataframe(values[["Dataset", "Protocol", "Split", "Model", "Metric",
                                "Score", "JSON"]].head(80),
                         use_container_width=True, hide_index=True)
            st.caption("These rows are a catalog of reported scores, not a pooled leaderboard.")
    else:
        st.info("No supported JSON evaluation reports found yet.")
    st.subheader("Workspaces")
    st.markdown(
        "Use the **sidebar** to switch to Model comparisons, Training analytics, "
        "Retrieval & LambdaRank, UMAP, Checkpoints, or the JSON report explorer. "
        "The Recommender page remains available for existing users and cold-start favorites."
    )


def show_comparisons():
    st.header("Model comparisons")
    _report_location_info()
    reports = _reports()
    values = comparable_records(reports)
    if values.empty:
        st.info("No measured JSON metrics found. The catalog alone cannot provide model rankings.")
        st.code("python -m src.train", language="bash")
        return
    datasets = sorted(values["Dataset"].unique())
    dataset = st.selectbox("MovieLens dataset", datasets, key="cmp_dataset")
    df = values[values["Dataset"] == dataset]
    splits = sorted(df["Split"].unique())
    selected_split = st.selectbox(
        "Evaluation split", splits, index=splits.index("test") if "test" in splits else 0,
        key="cmp_split",
    )
    df = df[df["Split"] == selected_split]
    scopes = list(df[["Scope", "Protocol", "Provenance"]]
                  .drop_duplicates().itertuples(index=False, name=None))
    scope = st.selectbox(
        "Comparable protocol / cohort / candidate pool",
        [a for a, _, _ in scopes],
        format_func=lambda value: next(
            f"{p} · {source} · {value[:80]}" for k, p, source in scopes if k == value
        ),
        key="cmp_scope",
    )
    df = df[df["Scope"] == scope]
    st.caption(
        "Only records with the exact same validated dataset/protocol details "
        "and evaluation split, or records within one JSON experiment, are grouped."
    )
    keys = sorted(df["Metric"].unique(), key=lambda s: (
        not s.lower().startswith("ndcg"), s))
    metric = st.selectbox("Measured ranking metric", keys, key="cmp_metric")
    data = df[df["Metric"] == metric].copy()
    if data.empty:
        st.info("No data for the selected metric.")
        return
    data["Score"] = pd.to_numeric(data["Score"], errors="coerce")
    data = data.dropna(subset=["Score"])
    data = data.sort_values("Score", ascending=metric.lower() in ("rmse", "mae"))
    if not data.empty:
        plot = alt.Chart(data).mark_bar().encode(
            x=alt.X("Score:Q", title=metric),
            y=alt.Y("Model:N", sort="-x"),
            tooltip=["Model:N", alt.Tooltip("Score:Q", format=".5f"), "JSON:N"],
        ).properties(height=max(220, min(850, len(data) * 36)))
        st.altair_chart(plot, use_container_width=True)
    st.dataframe(
        data[["Model", "Metric", "Score", "Protocol", "JSON"]],
        use_container_width=True, hide_index=True,
    )
    st.download_button(
        "Export this comparable result set (JSON)",
        data=data[["Model", "Metric", "Score", "Protocol", "JSON"]].to_json(
            orient="records", indent=2),
        file_name="reco_model_comparison.json",
        mime="application/json",
    )
    st.caption(
        "The grouping never equates sampled-negative ranking to full-catalog "
        "ranking. For new metrics, generate a real held-out evaluation report."
    )


def show_training():
    st.header("Training analytics")
    st.write(
        "Inspect B+C+D preference and JointBCD epoch-by-epoch runs, "
        "validation curves, regularization settings and original logs."
    )
    logs = discover_logs()
    if not logs:
        st.info("No training logs found. Your runner can save logs using tee.")
        st.code(
            "python -u fast_unified_regularized.py ... 2>&1 | "
            "tee fast_bcd_training.log", language="bash",
        )
        return
    names = {str(path): path for path in logs}
    selected = st.selectbox("Training log", list(names), key="train_file")
    path = names[selected]
    size = path.stat().st_size
    st.caption(f"Log size: {size / (1024 * 1024):.2f} MiB")
    # Read only a bounded tail for very large training logs.
    with path.open("rb") as src:
        if size > 12_000_000:
            src.seek(-12_000_000, 2)
        text = src.read().decode("utf-8", errors="replace")
    if size > 12_000_000:
        st.warning("Showing only the last 12 MiB of this log to avoid large-page memory use.")
    frame = parse_training_log(text)
    if frame.empty:
        st.info(
            "No explicit epoch+metric pairs were recognized in this log. "
            "The original log remains available below; no values were inferred."
        )
    else:
        candidates = sorted(frame["Metric"].unique())
        selected_metrics = st.multiselect(
            "Plot recorded metrics", candidates,
            default=candidates[:min(3, len(candidates))], key="train_metrics",
        )
        if selected_metrics:
            plotted = frame[frame["Metric"].isin(selected_metrics)]
            chart = alt.Chart(plotted).mark_line(point=True).encode(
                x=alt.X("Epoch:Q", scale=alt.Scale(zero=False)),
                y=alt.Y("Value:Q", scale=alt.Scale(zero=False)),
                color="Metric:N",
                tooltip=["Epoch:Q", "Metric:N", alt.Tooltip("Value:Q", format=".6f")],
            ).properties(height=400).interactive()
            st.altair_chart(chart, use_container_width=True)
            st.dataframe(plotted, hide_index=True, use_container_width=True)
            st.download_button(
                "Export parsed training metrics (JSON)",
                data=plotted.to_json(orient="records", indent=2),
                file_name="reco_training_curve.json", mime="application/json",
            )
    with st.expander("View actual training log", expanded=False):
        st.code("\n".join(text.splitlines()[-400:]), language="text")
        st.download_button(
            "Download full inspected log text", data=text.encode("utf-8"),
            file_name=path.name, mime="text/plain",
        )
    st.caption("Metrics and epochs are extracted only when named explicitly in source logs.")


def show_retrieval():
    st.header("Retrieval and LambdaRank")
    st.write(
        "Evaluate candidate generation separately from reranking. "
        "Use the same held-out users, query timestamps, and eligible item pools."
    )
    reports = _reports()
    records = comparable_records(reports)
    if records.empty:
        st.info("No measured retrieval or reranker JSON results found.")
    else:
        metric_names = records["Metric"].str.lower()
        retrieval = records[metric_names.str.contains(
            r"recall(?:@|_at_)(?:200|500|1000)|hit_rate(?:@|_at_)(?:200|500|1000)",
            regex=True, na=False)]
        reranking = records[
            metric_names.str.contains(
                r"ndcg|precision|mrr|map|recall|coverage|diversity",
                regex=True, na=False,
            ) & records["Model"].str.lower().str.contains(
                r"lambda|rank|bpr|hofm|deepfm|fm", regex=True, na=False,
            )
        ]
        st.subheader("Candidate retrieval — measured Recall@200/500/1000")
        _checkbox_frame(
            retrieval[["Dataset", "Split", "Protocol", "Model", "Metric",
                       "Score", "JSON"]].to_dict("records"),
            "No large-K candidate-retrieval metrics have been exported to JSON yet.",
        )
        st.subheader("Reranking — measured test results")
        _checkbox_frame(
            reranking[["Dataset", "Split", "Protocol", "Model", "Metric",
                       "Score", "JSON"]].to_dict("records"),
            "No recognized LambdaRank/HOFM/BPR reranking metrics found yet.",
        )
    st.subheader("LambdaRank feature importance")
    choices = []
    for path, kind, report in reports:
        imp = feature_importances(report)
        if not imp.empty:
            choices.append((str(path), imp))
    if choices:
        option = st.selectbox(
            "Saved feature importance report", [name for name, _ in choices],
            key="importance_json",
        )
        frame = dict(choices)[option].sort_values(
            "Importance", ascending=False).head(60)
        st.bar_chart(frame.set_index("Feature")[["Importance"]])
        st.dataframe(frame, hide_index=True, use_container_width=True)
    else:
        st.info(
            "No actual LambdaRank feature-importance values are saved in the JSON "
            "reports. The UI will not invent them from checkpoint names."
        )
    st.subheader("Existing reranker command")
    st.code(
        "python -m src.lambdarank_rerank train \\\n"
        "  --train artifacts/rerank/train.csv \\\n"
        "  --valid artifacts/rerank/valid.csv \\\n"
        "  --test artifacts/rerank/test.csv \\\n"
        "  --manifest artifacts/rerank/evaluation_manifest.json \\\n"
        "  --ensemble --k 10 \\\n"
        "  --output artifacts/rerank/combined_v1",
        language="bash",
    )
    st.caption(
        "This command assumes genuine frozen candidate scores exist. "
        "Neither reranker training nor retrieval runs automatically inside the UI."
    )


def show_umap(engine):
    st.header("Movie embeddings: UMAP / PCA")
    st.write(
        "Interactive genre-colored movie maps. Choose SVD collaborative item "
        "factors or sentence embeddings from your saved recommender and MovieLens 32M cache."
    )
    sources = []
    if engine is not None:
        sources.append("SVD item factors")
        if engine.semantic_item is not None:
            sources.append("Semantic vectors (saved recommender)")
    if ((SEMANTIC_CACHE / "movie_ids.npy").is_file()
            and (SEMANTIC_CACHE / "title_embeddings.npy").is_file()):
        sources.append("MovieLens 32M title sentence embeddings")
    if ((SEMANTIC_CACHE_1M / "movie_ids.npy").is_file()
            and (SEMANTIC_CACHE_1M / "title_embeddings.npy").is_file()):
        sources.append("MovieLens 1M title sentence embeddings")
    if not sources:
        st.info(
            "No movie vectors yet. Encode MovieLens 32M titles or train "
            "the latest-small baseline recommender."
        )
        st.code(
            "python -m src.encode_ml32m_text encode \\\n"
            "  --data-dir data/raw/ml-32m \\\n"
            "  --output artifacts/semantic/ml32m_minilm_full_v1",
            language="bash",
        )
        return
    selector = st.columns([2, 1, 1])
    with selector[0]:
        source = st.selectbox("Dataset / vector representation", sources,
                              key="studio_umap_source")
    with selector[1]:
        n = st.select_slider(
            "Number of films", [250, 500, 1000, 2000, 3000, 5000, 10000],
            value=2000, key="studio_umap_count",
        )
    with selector[2]:
        neighbors = st.slider("Neighbors", 5, 60, 20, 5, key="studio_umap_neighbors")
    method = st.radio(
        "Projection method", ["UMAP", "PCA"], horizontal=True,
        key="studio_projection_method",
    )
    seed = 42
    settings = (source, n, neighbors if method == "UMAP" else None, seed, method)
    if st.button("Generate genre-colored projection", key="studio_umap_run"):
        try:
            with st.spinner("Projecting movie embeddings …"):
                cache = (SEMANTIC_CACHE_1M
                         if source.startswith("MovieLens 1M")
                         else SEMANTIC_CACHE)
                vectors, metadata = sample_umap_source(
                    source, engine=engine, cache_dir=cache,
                    count=n, seed=seed,
                )
                data = (umap_projection(
                    vectors, metadata, n_neighbors=neighbors, seed=seed,
                ) if method == "UMAP" else
                    pca_projection(vectors, metadata, seed=seed))
                st.session_state["studio_umap_result"] = {
                    "settings": settings, "data": data,
                }
        except (RuntimeError, ValueError, FileNotFoundError, ImportError) as exc:
            st.error(str(exc))
    saved = st.session_state.get("studio_umap_result")
    if not saved or saved["settings"] != settings:
        st.caption(
            "Click Generate to compute a real UMAP or PCA projection. "
            "Projection distances are not measured recommendation performance."
        )
        return
    data = saved["data"]
    x_axis, y_axis = f"{method} 1", f"{method} 2"
    if not data.empty:
        chart = alt.Chart(data).mark_circle(size=50, opacity=0.70).encode(
            x=alt.X(f"{x_axis}:Q", scale=alt.Scale(zero=False)),
            y=alt.Y(f"{y_axis}:Q", scale=alt.Scale(zero=False)),
            color=alt.Color("Primary genre:N"),
            tooltip=["title:N", alt.Tooltip("movieId:Q", format="d"), "genres:N"],
        ).properties(height=610).interactive()
        st.altair_chart(chart, use_container_width=True)
    st.dataframe(
        data[["movieId", "title", "genres", x_axis, y_axis]],
        use_container_width=True, hide_index=True,
    )
    st.download_button(
        "Export movie coordinates as JSON",
        data=data.to_json(orient="records", indent=2),
        file_name=f"reco_{method.lower()}_map.json", mime="application/json",
    )
    st.caption(
        "Full embedding arrays are preserved; only the visualization sample "
        "is capped for responsive interaction. Install optional UMAP with "
        "pip install -r requirements-umap.txt."
    )


def show_checkpoints():
    st.header("Checkpoint and experiment explorer")
    st.write(
        "Examine actual saved model files and regularization metadata without "
        "loading executable pickle/PyTorch checkpoint objects."
    )
    inventory = checkpoint_inventory()
    if not inventory:
        st.info("No checkpoint artifacts found under models/ or artifacts/.")
        return
    st.dataframe(pd.DataFrame(inventory), use_container_width=True, hide_index=True)
    folders = sorted({entry["Directory"] for entry in inventory})
    folder = st.selectbox("Inspect checkpoint folder", folders, key="ckpt_folder")
    folder_files = [row for row in inventory if row["Directory"] == folder]
    st.dataframe(pd.DataFrame(folder_files), hide_index=True, use_container_width=True)
    filename, config = training_configuration(folder)
    if config is not None:
        st.subheader("Stored training configuration")
        st.caption("Source: " + str(Path(folder) / filename))
        st.json(config)
        st.download_button(
            "Export checkpoint config (JSON)",
            data=json.dumps(config, indent=2), mime="application/json",
            file_name="reco_checkpoint_config.json",
        )
    else:
        st.info(
            "No recognized configuration JSON exists beside this checkpoint. "
            "Checkpoint names alone cannot establish actual hyperparameters."
        )
    st.caption(
        "This viewer is read-only. It never torch.load()s or unpickles "
        "untrusted checkpoint objects."
    )


def show_json_explorer():
    st.header("JSON report explorer")
    st.write(
        "Browse original test reports, validation results and experiment summaries "
        "without importing to SQLite."
    )
    reports = _reports()
    if not reports:
        st.info("No actual evaluation JSON reports found in artifacts/.")
    else:
        by_path = {str(path): (kind, doc, path) for path, kind, doc in reports}
        choice = st.selectbox("Saved JSON file", list(by_path), key="json_browser_file")
        kind, doc, path = by_path[choice]
        st.caption("Detected report type: " + kind)
        split = st.radio("Split", ["test", "validation"], horizontal=True,
                         key="json_browser_split") if kind != "evaluation" else "test"
        frame = metric_frame(kind, doc, split=split)
        if not frame.empty:
            st.dataframe(frame, hide_index=True, use_container_width=True)
        st.json(doc)
        st.download_button(
            "Download original JSON",
            data=path.read_bytes(), file_name=path.name,
            mime="application/json",
        )
    st.divider()
    st.subheader("Inspect a JSON report without saving it")
    uploaded = st.file_uploader(
        "Load a local report", type=["json"], key="json_upload",
        help="Processed only in this Streamlit session, without database writes.",
    )
    if uploaded is not None:
        try:
            if uploaded.size > 8_000_000:
                raise ValueError("File exceeds the 8 MiB report limit")
            doc = json.load(uploaded)
            st.json(doc)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            st.error("Could not parse the uploaded JSON: " + str(exc))


def show_movie_search():
    st.header("Find films and discover semantic neighbors")
    movie_file = next((p for p in DATA_MOVIES if p.is_file()), None)
    if movie_file is None:
        st.info("MovieLens 32M movies.csv is not found under the expected local paths.")
        return
    movies = _movie_catalog(str(movie_file), movie_file.stat().st_mtime_ns)
    text = st.text_input("Search film titles", placeholder="The Matrix, Casablanca, ...")
    if not text.strip():
        st.caption("Search the full local 32M catalog by movie title.")
        return
    matches = movies[movies["title"].str.contains(
        text.strip(), regex=False, case=False, na=False)].head(100)
    if matches.empty:
        st.info("No titles matched this search.")
        return
    st.dataframe(matches, use_container_width=True, hide_index=True)
    if not (SEMANTIC_CACHE / "movie_ids.npy").is_file():
        st.caption("Run the semantic encoder to enable similar-movie search.")
        return
    options = {
        str(row.title) + f" · ID {row.movieId}": int(row.movieId)
        for row in matches.itertuples(index=False)
    }
    selection = st.selectbox("Find semantically similar films to", list(options),
                             key="similar_movie_id")
    if st.button("Search semantic neighbors", key="semantic_neighbors"):
        try:
            ids = np.load(SEMANTIC_CACHE / "movie_ids.npy", mmap_mode="r")
            vectors = np.load(SEMANTIC_CACHE / "title_embeddings.npy", mmap_mode="r")
            if vectors.shape[0] != len(ids):
                raise ValueError("Semantic cache ID alignment mismatch")
            row = np.searchsorted(ids, options[selection])
            if row >= len(ids) or int(ids[row]) != options[selection]:
                raise ValueError("Selected movie is not in semantic cache")
            query = np.asarray(vectors[row], dtype="float32")
            # Sentence Transformer vectors are normalized by the encoder;
            # use cosine similarity, excluding the input movie itself.
            scores = np.asarray(vectors @ query, dtype="float32")
            scores[row] = -np.inf
            nearest = np.argsort(-scores, kind="stable")[:20]
            lookup = movies.drop_duplicates("movieId").set_index("movieId")
            info = lookup.reindex(ids[nearest]).reset_index()
            info["semantic_similarity"] = scores[nearest]
            st.dataframe(
                info[["movieId", "title", "genres", "semantic_similarity"]],
                hide_index=True, use_container_width=True,
            )
            st.download_button(
                "Export semantic neighbors JSON",
                data=info.to_json(orient="records", indent=2),
                file_name="reco_semantic_neighbors.json",
                mime="application/json",
            )
        except (OSError, ValueError, IndexError) as exc:
            st.error("Cannot search these saved embeddings: " + str(exc))
    st.caption(
        "This is unsupervised title/genre similarity, not personalized "
        "HOFM3/HOFM4 predictions."
    )


def show_methods():
    st.header("Evaluation protocol and model methodology")
    st.markdown(
        """
        ### Model families
        The research catalog includes popularity, temporal popularity, SVD,
        BPR/BPR-MF, classic FM, DeepFM, HOFM3, HOFM4, 15-feature
        JointBCD, temporal preferences B/C/D, Sentence Transformers,
        two-tower retrieval, LambdaRank/LambdaMART and experimental
        generative/slate variants.

        ### Comparability
        - Same dataset version, eligible users, chronological cutoff and
          item-candidate pool.
        - Same observed-positive definition, seen-item exclusions, split
          and exact K for NDCG, Recall, Precision, MRR, coverage and diversity.
        - Report Recall@500/1000 for retrieval **before** reranking NDCG@10.
        - Never interpret an unobserved movie as a verified user dislike.
        - Do not compare sampled-negative NDCG directly to full-catalog NDCG.
        - Use validation for hyperparameter selection; preserve untouched test.
        - Do not use tags, ratings or derived features that postdate a query.

        ### Data and saved files
        The studio reads local JSON and training log/checkpoint metadata, not
        the experiment database. The historical SQLite registry may be kept
        as an archival artifact. No file is deleted or overwritten.

        ### Limits
        The latest-small HybridRecommender is self-contained and can serve
        recommendations when its trusted `recommender.joblib` exists.
        MovieLens 32M HOFM3/HOFM4 research patches and checkpoints still need
        a separately validated inference adapter for direct recommendations.
        Training curves and metrics appear only when reported in real logs or JSON.

        [MovieLens / GroupLens dataset information](https://grouplens.org/datasets/movielens/)
        """
    )
