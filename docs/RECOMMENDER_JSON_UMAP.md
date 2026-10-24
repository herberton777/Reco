# Recommender studio — JSON reports, recommendations, UMAP

The default Reco application is again the **recommender studio**, not the
SQLite benchmark administration interface.

```bash
cd /workspaces/Reco
python -m pip install -r requirements.txt
python -m pip install -r requirements-umap.txt
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

In Codespaces, open forwarded port **8501**. Keep access **Private**
unless you explicitly intend to share research artifacts.

## Tabs

- **Recommendations:** choose a trained MovieLens user, compare SVD, genres,
  popularity, hybrid, and any trained optional models. See earlier films and
  export recommended movies as JSON.
- **Choose favorites:** select favorite movies for cold-start recommendations,
  compare genres and collaborative factors, and export as JSON.
- **UMAP movie map:** explore SVD item vectors, saved recommender sentence
  vectors, or your full MovieLens 32M title/genre Sentence Transformer
  embeddings. Hover to inspect movie names and genres. Choose a deterministic
  sample (250–10,000 films) and export UMAP coordinates as JSON.
- **Results (JSON):** read existing `artifacts/metrics.json` and JSON
  experiment/rerank reports. Model metrics and bar charts appear only when
  measured reports actually exist. There are no database queries.
- **Methodology:** data splits, evaluation rules, and scientific caveats.

UMAP itself is an **optional** package because it pulls in a relatively
heavy numba dependency. Other studio functions still work without it.
Install `requirements-umap.txt` to enable the projection.

## JSON formats

The historical MovieLens latest-small evaluation automatically writes:

```
artifacts/metrics.json
artifacts/recommender.joblib
```

For the baseline data:

```bash
python -m src.train
```

If you already have these artifacts, **don't retrain**; launch the app.

The JSON tab also recognizes actual `report*.json` from the HOFM/LambdaRank
reranking workflow and `summary.json` variant comparisons in
`artifacts/rerank/*/`. You can preserve research JSON reports in
`artifacts/reports/` or `artifacts/benchmarks/` for selection.

It does not convert or import an existing `benchmarks.sqlite`. That SQLite
file is **left untouched** for reproducibility, but is not a required input
to the visual recommender. The old optional `benchmark_dashboard.py`
remains in the repository for anyone who still needs its historic data.

JSON reports from incompatible datasets, candidate pools or evaluation
protocols are **not merged into a misleading comparison**.

## Full MovieLens 32M movie maps

The UMAP tab can read the semantic encoder's existing files:

```
artifacts/semantic/ml32m_minilm_full_v1/movie_ids.npy
artifacts/semantic/ml32m_minilm_full_v1/title_embeddings.npy
```

It aligns movie IDs to `data/raw/ml-32m/movies.csv` where available.
If the semantic encoder hasn't finished yet, the tab still works with
saved SVD recommender item vectors (if present).

If encoding has not yet run:

```bash
python -m pip install sentence-transformers
python -m src.encode_ml32m_text encode \
  --data-dir data/raw/ml-32m \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --device cpu \
  --output artifacts/semantic/ml32m_minilm_full_v1
```

The UMAP computation samples movies rather than running UMAP on all
87,585 points interactively. This does **not** truncate the underlying
embedding dataset.

## Important distinction

`app.py` can display MovieLens 32M **semantic UMAP maps** without a
trained recommender. Personalized recommendations currently use the
self-contained MovieLens latest-small `HybridRecommender` artifact.

The advanced HOFM3/HOFM4 models are still installed from separate
research patches and do not automatically become recommenders inside
the standalone app; this would require a versioned inference adapter.
No model checkpoints are deleted or overwritten by this UI change.
