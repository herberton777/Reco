"""JSON-first Streamlit recommender studio, with UMAP and movie exploration.

Launch: streamlit run app.py --server.address 0.0.0.0 --server.port 8501
No SQLite database is read or written by this interface.
"""
from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.recommender import HybridRecommender
from src.reco_visual_data import (
    available_json_reports, metric_frame, read_ranking_json,
    sample_umap_source, umap_projection,
)

st.set_page_config(page_title="Reco Studio · MovieLens Research Lab", page_icon="🎬", layout="wide")
st.title("🎬 Reco Studio")
st.caption(
    "MovieLens research workbench · recommendations · model comparisons · "
    "training · retrieval · UMAP · checkpoint explorer · JSON results"
)

ARTIFACT = Path("artifacts/recommender.joblib")
SEMANTIC_CACHE = Path("artifacts/semantic/ml32m_minilm_full_v1")
JSON_RESULTS = Path("artifacts/metrics.json")


@st.cache_resource(show_spinner="Loading trained recommendation engine …")
def load_engine(filename):
    # joblib can execute arbitrary code; load only a file trained in this trusted Codespace.
    return HybridRecommender.load(filename)


engine = None
engine_failure = None
if ARTIFACT.exists():
    try:
        engine = load_engine(str(ARTIFACT))
    except Exception as exc:
        engine_failure = str(exc)

# Keep recommendations as the original five-tab view; expose the
# much broader research lab via a mobile-friendly single sidebar selector.
# All pages use existing JSON, log, embedding, or checkpoint files only.
pages = [
    "Overview", "Recommender", "Model comparisons", "Training analytics",
    "Retrieval & LambdaRank", "Movie search", "UMAP explorer",
    "Checkpoint explorer", "JSON reports", "Methodology",
]
st.sidebar.title("Reco Studio")
page = st.sidebar.radio("Workspace", pages, index=0)
st.sidebar.caption("JSON-first research · no benchmark database required")
if page != "Recommender":
    from src import reco_studio_pages as pages_ui
    handlers = {
        "Overview": lambda: pages_ui.show_overview(engine),
        "Model comparisons": pages_ui.show_comparisons,
        "Training analytics": pages_ui.show_training,
        "Retrieval & LambdaRank": pages_ui.show_retrieval,
        "Movie search": pages_ui.show_movie_search,
        "UMAP explorer": lambda: pages_ui.show_umap(engine),
        "Checkpoint explorer": pages_ui.show_checkpoints,
        "JSON reports": pages_ui.show_json_explorer,
        "Methodology": pages_ui.show_methods,
    }
    handlers[page]()
    st.stop()

recs_tab, favorites_tab, map_tab, results_tab, methodology_tab = st.tabs([
    "Recommendations", "Choose favorites", "UMAP movie map",
    "Results (JSON)", "Methodology",
])

with recs_tab:
    st.subheader("Recommend films to an existing user")
    if engine is None:
        st.info(
            "No usable trained recommender was found at artifacts/recommender.joblib. "
            "UMAP and JSON results remain accessible in the other tabs."
        )
        if engine_failure:
            st.error("Failed loading saved model: " + engine_failure)
        st.code(
            "python -m src.train\n"
            "streamlit run app.py --server.address 0.0.0.0 --server.port 8501",
            language="bash",
        )
        st.caption(
            "The self-contained recommender above is trained on MovieLens latest-small. "
            "It does not automatically load experimental HOFM3/HOFM4 checkpoints."
        )
    else:
        user_col, model_col, count_col = st.columns([1, 1, 1])
        with user_col:
            user_id = st.selectbox(
                "Anonymous MovieLens training user", engine.user_ids.tolist(),
                key="reco_user",
            )
        with model_col:
            model_names = engine.models
            model_name = st.selectbox(
                "Recommender model", model_names,
                index=model_names.index("hybrid") if "hybrid" in model_names else 0,
                key="reco_model",
            )
        with count_col:
            n = st.slider("Number of recommendations", 5, 30, 10, 5)
        hist_col, reco_col = st.columns([1, 2])
        with hist_col:
            st.markdown("**Previously seen movies (training only)**")
            seen_indexes = sorted(engine.seen[int(user_id)])
            history = engine.movies.iloc[seen_indexes][["title", "genres"]]
            st.dataframe(history.head(15), hide_index=True,
                         use_container_width=True)
        with reco_col:
            st.markdown("**Recommended movies**")
            recommendations = engine.recommend(user_id, k=n, model=model_name)
            st.dataframe(recommendations, hide_index=True, use_container_width=True)
            st.download_button(
                "Export recommendations as JSON",
                data=recommendations.to_json(orient="records", indent=2),
                file_name=f"reco_user_{user_id}_{model_name}.json",
                mime="application/json",
            )
        st.caption(
            "Ratings reserved for ranker training, validation and test are not "
            "included in the displayed training history."
        )

with favorites_tab:
    st.subheader("Find similar films from favorites")
    if engine is None:
        st.info("Train the baseline recommender to enable favorite-based recommendations.")
        st.code("python -m src.train", language="bash")
    else:
        st.write(
            "Select up to seven favorite movies. The model uses genre similarity, "
            "collaborative item factors and (when available) saved sentence embeddings."
        )
        names = {f"{row.title} · ID {row.movieId}": int(row.movieId)
                 for row in engine.movies.itertuples(index=False)}
        defaults = [name for name in names if name.startswith("Toy Story (1995)")][:1]
        selection = st.multiselect(
            "Movies you enjoyed", options=list(names),
            default=defaults, max_selections=7,
        )
        if selection:
            recommendations = engine.recommend_from_favorites(
                [names[name] for name in selection], k=15,
            )
            st.dataframe(recommendations, hide_index=True, use_container_width=True)
            st.download_button(
                "Export these favorites recommendations as JSON",
                data=recommendations.to_json(orient="records", indent=2),
                file_name="reco_from_favorites.json", mime="application/json",
            )
        else:
            st.info("Select one or more movie titles above.")

with map_tab:
    st.subheader("UMAP: explore the movie embedding space")
    st.write(
        "See how films are grouped by collaborative or semantic embeddings. "
        "Select a representation, then generate the map. "
        "UMAP is exploratory, not a ranking metric."
    )
    sources = []
    if engine is not None:
        sources.append("SVD item factors")
        if engine.semantic_item is not None:
            sources.append("Semantic vectors (saved recommender)")
    if (SEMANTIC_CACHE / "title_embeddings.npy").is_file() and (
            SEMANTIC_CACHE / "movie_ids.npy").is_file():
        sources.append("MovieLens 32M title sentence embeddings")

    if not sources:
        st.info(
            "No movie embedding vectors were found. Train the baseline recommender "
            "or complete the full-catalog Sentence Transformer encoding."
        )
        st.code(
            "python -m src.train\n"
            "# Or, for 32M semantic embeddings:\n"
            "python -m src.encode_ml32m_text encode \\\n"
            "  --data-dir data/raw/ml-32m \\\n"
            "  --output artifacts/semantic/ml32m_minilm_full_v1",
            language="bash",
        )
    else:
        sel_col, count_col, neighbors_col = st.columns([2, 1, 1])
        with sel_col:
            source = st.selectbox("Embedding source", sources, key="umap_source")
        with count_col:
            n_points = st.select_slider(
                "Number of sampled movies",
                options=[250, 500, 1000, 2000, 3000, 5000, 10000],
                value=2000,
            )
        with neighbors_col:
            neighbors = st.slider("UMAP neighbors", 5, 50, 20, step=5)
        st.caption(
            "UMAP uses a deterministic movie sample (seed 42), cosine distance "
            "and preserves the exact MovieLens movie-ID mapping."
        )
        setting = (source, n_points, neighbors)
        if st.button("Generate UMAP map", type="primary"):
            try:
                with st.spinner("Projecting selected movie vectors with UMAP …"):
                    vectors, movie_info = sample_umap_source(
                        source, engine=engine, cache_dir=SEMANTIC_CACHE,
                        count=n_points, seed=42,
                    )
                    mapped = umap_projection(
                        vectors, movie_info, n_neighbors=neighbors, seed=42,
                    )
                st.session_state["reco_umap"] = {
                    "settings": setting, "frame": mapped,
                }
            except (ValueError, FileNotFoundError, RuntimeError, ImportError) as exc:
                st.error(str(exc))
        current = st.session_state.get("reco_umap")
        if current and current["settings"] == setting:
            points = current["frame"]
            plot = alt.Chart(points).mark_circle(size=46, opacity=0.65).encode(
                x=alt.X("UMAP 1:Q", scale=alt.Scale(zero=False)),
                y=alt.Y("UMAP 2:Q", scale=alt.Scale(zero=False)),
                color=alt.Color("Primary genre:N", title="Primary genre"),
                tooltip=[
                    alt.Tooltip("title:N", title="Movie"),
                    alt.Tooltip("movieId:Q", title="Movie ID", format="d"),
                    alt.Tooltip("genres:N", title="Genres"),
                ],
            ).properties(height=580).interactive()
            st.altair_chart(plot, use_container_width=True)
            st.dataframe(
                points[["movieId", "title", "genres", "UMAP 1", "UMAP 2"]],
                hide_index=True, use_container_width=True,
            )
            st.download_button(
                "Export UMAP coordinates (JSON)",
                data=points.to_json(orient="records", indent=2),
                file_name="reco_umap_movies.json", mime="application/json",
            )
        else:
            st.caption("Click Generate UMAP map to calculate a projection.")
    st.caption(
        "Install optional UMAP support with "
        "`python -m pip install -r requirements-umap.txt`. "
        "The full 32M catalog is stored on disk; the visualization intentionally "
        "samples a smaller number of movies for responsiveness."
    )

with results_tab:
    st.subheader("Saved results — JSON, not a database")
    reports = available_json_reports("artifacts")
    if not reports:
        st.info(
            "No benchmark JSON reports found. Recommendations and UMAP do not "
            "require a benchmark database."
        )
        st.code("python -m src.train", language="bash")
        st.caption(
            "MovieLens latest-small training produces artifacts/metrics.json. "
            "Advanced HOFM reports must come from actual evaluation scripts."
        )
    else:
        by_name = {str(path): path for path in reports}
        default = str(JSON_RESULTS)
        chosen = st.selectbox(
            "Choose a saved JSON report", list(by_name),
            index=list(by_name).index(default) if default in by_name else 0,
        )
        try:
            kind, report = read_ranking_json(by_name[chosen])
            if kind == "baseline":
                split = st.radio("Evaluation split", ["test", "validation"],
                                 horizontal=True)
            elif kind == "rerank_summary":
                split = st.radio("Evaluation split", ["test", "validation"],
                                 horizontal=True)
            else:
                split = "test"
            frame = metric_frame(kind, report, split=split)
            if frame.empty:
                st.info("This JSON report contains no scored models for that split.")
            else:
                st.dataframe(frame, hide_index=True, use_container_width=True)
                scores = [
                    key for key in ("ndcg_at_k", "ndcg@10", "recall_at_k", "recall@10")
                    if key in frame and pd.api.types.is_numeric_dtype(frame[key])
                ]
                if scores:
                    chosen_metric = st.selectbox("Plot metric", scores)
                    st.bar_chart(frame.set_index("Model")[[chosen_metric]])
            if kind == "evaluation":
                protocol = report.get("protocol")
                if isinstance(protocol, dict):
                    st.caption(
                        "Dataset: " + str(report.get("dataset", {}).get("id", "unknown"))
                        + " · Protocol: " + str(protocol.get("id", "unknown"))
                        + " · Split: " + str(report.get("evaluation_split", "unknown"))
                    )
                st.caption("Source type: " + str(report.get("source_type", "unknown")))
            elif kind == "baseline":
                st.caption(
                    "Dataset: " + str(report.get("dataset", "unknown"))
                    + " · K: " + str(report.get("k", "unknown"))
                )
            st.download_button(
                "Export original report (JSON)",
                data=by_name[chosen].read_bytes(),
                file_name=by_name[chosen].name, mime="application/json",
            )
            with st.expander("Inspect original JSON"):
                st.json(report)
        except (OSError, ValueError, json.JSONDecodeError, KeyError) as exc:
            st.error("Cannot display this report: " + str(exc))
    st.caption(
        "Results are loaded from local JSON files. They are not averaged "
        "or compared across incompatible test protocols."
    )

with methodology_tab:
    st.markdown(
        """
        ### About the recommender
        **Known-user baseline:** popularity, TruncatedSVD collaborative filtering,
        genre TF-IDF, hybrid ranking and optional LambdaMART/two-tower models.

        **Movie exploration:** UMAP projects stored item vectors or full-catalog
        MovieLens 32M sentence embeddings into two dimensions. Genre labels
        are for inspection; UMAP neighborhoods are not proof of recommendation accuracy.

        **Evaluation:** the reproducible latest-small baseline uses each eligible
        user's last three positive events for ranker training, validation and test,
        then evaluates full-catalog top-K ranking with seen-item exclusions.
        HOFM3/HOFM4 32M experiments may follow different chronological and
        candidate protocols; their NDCG scores must not be pooled blindly.

        **Data and privacy:** everything displayed is read from local trusted model
        artifacts and JSON reports. This UI never opens or writes SQLite.
        Only load `.joblib` files you created or trust, because pickle-based
        deserialization can execute code.

        **Attribution:** [MovieLens / GroupLens](https://grouplens.org/datasets/movielens/)
        at the University of Minnesota. Respect the dataset's licensing restrictions.
        """
    )
