# Benchmark dashboard (Streamlit)

The [Reco experiment registry](BENCHMARK_REGISTRY.md) can be browsed in a
**separate, read-only dashboard**. It does not depend on a trained
`recommender.joblib`, the MovieLens dataset, or any advanced research patches.
It reads the **same local SQLite file** used by training commands.

## Run in GitHub Codespaces

From the repository root:

```bash
cd /workspaces/Reco
python -m pip install -r requirements.txt
python -m src.benchmark_registry init
streamlit run benchmark_dashboard.py --server.address 0.0.0.0 --server.port 8502
```

In the Codespace **Ports** panel, open port **8502** in your browser. Keep the
port's visibility **Private**: your research database may contain local paths,
commands, checkpoints, and evaluation metadata. This is a local research UI,
not a public authenticated hosted service.

The preexisting recommender app uses port **8501**:
`streamlit run app.py --server.port 8501`.
The separate dashboard still works if the recommender model is absent.

To use an alternate database, set the path on the server side:

```bash
RECO_BENCHMARK_DB=/absolute/path/to/benchmarks.sqlite \
  streamlit run benchmark_dashboard.py --server.port 8502
```

The dashboard never asks a browser visitor to provide an arbitrary database
path. It opens its file with SQLite **read-only mode** and does not create,
import, train, delete, or mutate data.

## What's on the screen?

**Leaderboard & charts** lets you filter by dataset (MovieLens Small, 1M,
32M), **evaluation protocol**, validation/test split, measured/literature
provenance, and a specific metric **and cutoff K** (for example NDCG@10 vs
Recall@200). The chart and sortable table show the source run IDs, actual
scores, sample size and evidence references. You may inspect every run or
show the best *single* observed run per model. A CSV export is available.

Below the leaderboard, you can plot multiple metrics from the **same exact
dataset/protocol/split/source cohort selection**, always labeling the cutoff.

**Experiment history** displays tracked HOFM training attempts (including
failed, interrupted, imported historical checkpoints), their command/config,
status, checkpoint metadata and how many evaluation reports were attached.
It also lists evaluated runs and allows inspection of their hyperparameters,
source hashes, recorded scores, and Git revisions.

**Protocols & catalog** describes evaluation splits, candidates, labels,
cutoffs and which models/datasets are registered.

## No results yet?

An initialized registry has models and datasets but **zero evaluated runs**.
An empty leaderboard is therefore expected until actual evaluated metrics
are imported; the dashboard never supplies sample or invented scores.

For a *real* MovieLens latest-small run:

```bash
python -m src.train --no-ranker --registry-db artifacts/benchmarks.sqlite
```

For a previously generated `artifacts/metrics.json`:

```bash
python -m src.benchmark_registry import-baseline artifacts/metrics.json
```

For MovieLens 32M, you can register an existing checkpoint without restarting
training, and then attach **a real measured evaluation report**:

```bash
python -m src.hofm_experiments register-existing \
  --folder models/ml-32m/unified_hofm3_bcd/EXISTING_CHECKPOINT \
  --label historical_hofm
python -m src.hofm_experiments list
# See HOFM_EXPERIMENT_TRACKING.md for the required measured report format.
```

Return to the dashboard and click **Refresh database**. Training attempts
appear in experiment history; NDCG/Recall only appears when evaluated
metrics have been attached or directly imported.

## Scientific limitations

- Scores from sampled negatives and full-catalogue ranking are **not**
  directly comparable, even if both say NDCG@10.
- The interface refuses to pool different datasets/protocols/splits/sources
  in one leaderboard. Counts and the protocol's cohort signature should
  still be inspected for fairness. A protocol ID is not independent proof.
- Choosing "best run" is **descriptive**, not a validated model-selection
  procedure; do not tune on the final test split.
- Training success and checkpoint existence are not evidence of ranking
  accuracy.
- The dashboard is meant for private local research use; do not share local
  files, commands, or experiment details in a public deployment.

Run data-query regression tests:

```bash
python -m pytest -q tests/test_benchmark_dashboard_data.py
```
