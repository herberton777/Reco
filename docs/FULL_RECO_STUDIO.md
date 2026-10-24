# Reco Studio — complete JSON-first MovieLens research workbench

The central app is `app.py` on port **8501**. It replaces the narrower
recommendation demo as the default interface and does **not require SQLite**.

## Codespaces launch

```bash
cd /workspaces/Reco
git fetch origin main
git merge --ff-only origin/main

python -m pip install -r requirements-umap.txt
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Only merge if your Git working tree can fast-forward safely. The above
does not intentionally delete untracked research files. **Do not run
`git clean` or `git reset --hard`** on this research Codespace.
Keep Codespaces port visibility Private if your local checkpoint names,
training logs or configuration files are sensitive.

The sidebar has ten research workspaces:

| Workspace | What it includes |
| --- | --- |
| Overview | MovieLens Small/1M/32M path inventory, 15 model-family catalog, count of real checkpoint/config files and actual reports |
| Recommender | Existing known-user/algorithm-selector, favorites-based cold start, JSON recommendations export; baseline latest-small trained joblib |
| Model comparisons | Real NDCG, Recall, Precision, MRR etc. under the **same** dataset/protocol/split/candidate pool; bar charts, sortable data, JSON export |
| Training analytics | Existing epoch logs, explicit training loss and validation ranking metrics vs epoch, raw log inspection and JSON export |
| Retrieval & LambdaRank | Recall@200/500/1000 if measured, ranked/retrieval scores from JSON, saved LambdaRank feature importance, training command reference |
| Movie search | Find titles/genres across the actual ML-32M movie catalog and, with embeddings, cosine nearest-neighbor discovery |
| UMAP explorer | Interactive genre-colored UMAP **or PCA** from saved latest-small, MovieLens 1M or 32M embeddings, deterministic 250–10,000 film sample, title tooltips and JSON export |
| Checkpoint explorer | Read-only inventory of actual PyTorch/joblib files and saved training configuration JSON; no unsafe deserialization |
| JSON reports | Original baseline metrics.json, evaluation report JSON and rerank summary JSON, interactive comparison tables, optional JSON file upload |
| Methodology | Historical negative-sampling, time cutoff and dataset-split caveats and supported architecture families |

### Expected artifacts

- `artifacts/recommender.joblib`: **trusted** existing MovieLens
  latest-small trained recommender. This is the artifact used for
  **personalized recommendations**. Don't download or unpickle untrusted files.
- `artifacts/metrics.json`: measured latest-small benchmark JSON.
- `artifacts/reports/*.json`, `artifacts/benchmarks/*.json`,
  `artifacts/rerank/*/report*.json`, `artifacts/rerank/*/summary.json`:
  other verified evaluation reports. Files under `artifacts/` are explored
  without scanning the SQLite registry.
- `models/ml-32m/.../weights.pt` and saved
  `regularization_config.json`: actual research checkpoints and metadata.
- `artifacts/training_logs/*.log` and root `*.log`: source training logs
  parsed for explicitly recorded metrics; no made-up training curves.
- `artifacts/semantic/ml32m_minilm_full_v1/title_embeddings.npy` and
  `movie_ids.npy`: full-catalog semantic representations used for UMAP and
  similar-movie discovery.
- `data/raw/ml-32m/movies.csv`: actual ML-32M film titles.

### Restore baseline recommendations if needed

If you already have the trained latest-small `recommender.joblib`, **do not
retrain it**. If it doesn't exist and you want known-user recommendations
rather than just exploration:

```bash
python -m src.train
```

This baseline trains on the latest-small MovieLens dataset. It is **not**
the patch-based 32M HOFM3 or HOFM4 research model.

### 1M/32M embedding UMAP, PCA and similar movies

```bash
python -m pip install sentence-transformers
python -m src.encode_ml32m_text encode \
  --data-dir data/raw/ml-32m \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --output artifacts/semantic/ml32m_minilm_full_v1
```

This is needed only if you have not already completed encoding. The
studio loads your existing vectors when present and never re-encodes
automatically. The **1M** semantic representation is also supported when
`artifacts/semantic/ml1m_minilm_full_v1/{movie_ids,title_embeddings}.npy`
exists; missing sources are simply not listed. UMAP/PCA computation samples a
subset to remain responsive; underlying embedding caches stay complete.

### Evaluation boundaries and research limitations

This application is **broad in research inspection**, but that does
not make every algorithm directly usable for inference. Popularity/SVD,
genre, hybrid, optional two-tower and LambdaMART can be served from a
trained latest-small `HybridRecommender`. HOFM3/HOFM4/BPR/FM/DeepFM/TIGER
models from separate prototype patches are inventoried and compared only
when real reports/checkpoints exist. They need individual, validated
model adapters for direct personalized 32M recommendations.

The overview model catalog is **not** a list of trained models.
The app never equates model availability, a checkpoint's existence or a
saved command with a verified NDCG value. It never fabricates the user's
reported HOFM3 NDCG@10 ~0.13 as a persisted result.

For fair comparisons, candidate sets, split, chronological cutoff,
eligible user cohort, positive-label definition and metric K must be
aligned. If provenance is incomplete, reports remain isolated to the
individual JSON file, not merged under a guessed protocol.

The **old** `benchmark_dashboard.py` and `artifacts/benchmarks.sqlite`
remain available if previously created, but neither is needed for Reco
Studio. Nothing here deletes them.

### Tests

```bash
python -m pytest -q tests/test_full_reco_studio.py tests/test_reco_visual_studio.py
```
