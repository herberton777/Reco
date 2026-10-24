# Reco — Movie Recommendation Research Lab

[![Python tests](https://github.com/herberton777/Reco/actions/workflows/tests.yml/badge.svg)](https://github.com/herberton777/Reco/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/code-MIT-blue)](LICENSE)

**An end-to-end recommendation-systems portfolio:** reproducible offline evaluation, collaborative filtering, semantic content signals, learning to rank, and an interactive Streamlit demonstration. Python · pandas · SciPy · scikit-learn · LightGBM · optional PyTorch.

> **What works from a fresh clone:** the self-contained **MovieLens Small** pipeline in 'src/', its synthetic CI tests, and the Streamlit UI (the app shows an informative setup screen until a trusted model has been trained).
>
> **Experimental work:** TIGER-style generative retrieval, MovieLens 1M/32M preprocessing, temporal FM/DeepFM, BPR ranking and LambdaMART reranking are preserved as **versioned prototype patch archives** under 'research-patches/'. These are research iterations, not yet a fully integrated, CI-verified install. See [Research status](docs/RESEARCH_STATUS.md). Please do not compare metrics across their different evaluation protocols.

## Quick start: reproducible MovieLens Small baseline

Use Python 3.12; a GitHub Codespace works without a local installation.

```bash
git clone https://github.com/herberton777/Reco.git
cd Reco
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m src.train
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

The training command downloads the GroupLens MovieLens latest-small dataset, trains classical models and a LambdaMART ranker, and writes local files under 'artifacts/'. In Codespaces, open forwarded port **8501**. A clean checkout does **not** include a trained model or invented benchmark numbers.

Other entry points:

| Command | Purpose |
| --- | --- |
| 'python -m src.eda' | Download and explore MovieLens Small |
| 'python -m src.train --no-ranker' | Train classical baselines without LightGBM ranking |
| 'python -m pip install -r requirements-neural.txt' | Install optional PyTorch dependencies |
| 'python -m src.train --neural --epochs 4' | Train an experimental two-tower recommender |
| 'python -m pip install -r requirements-optional-nlp.txt' | Install optional Sentence Transformers |
| 'python -m src.semantic' | Add title/genre embeddings to a trained model |
| 'python -m src.benchmark_semantic' | Evaluate semantic models with the existing holdout |

## Experiment benchmark registry

A provenance-aware SQLite registry keeps models, datasets, evaluation protocols,
hyperparameters, run metadata, and actual NDCG/Recall/Coverage results together
without inventing scores or mixing incompatible evaluations.

```bash
python -m src.train --no-ranker --registry-db artifacts/benchmarks.sqlite
python -m src.benchmark_registry leaderboard --dataset ml-latest-small \
  --protocol ml-small-per-user-3-fullcatalog-v1 --metric ndcg --k 10
```

You can also import existing `artifacts/metrics.json` or externally verified
MovieLens 32M/literature runs. The database is generated locally (and ignored
by Git), while the schema, importer, tests, and documentation are versioned.
See [Benchmark registry](docs/BENCHMARK_REGISTRY.md).

### Visual benchmark dashboard

A standalone, **read-only Streamlit dashboard** visualizes evaluation runs
and tracked training attempts from `artifacts/benchmarks.sqlite`. It includes
protocol-scoped NDCG/Recall leaderboards, charts, run and hyperparameter
details, and the full model/dataset catalog. No trained recommender model is
required to open it.

```bash
streamlit run benchmark_dashboard.py --server.address 0.0.0.0 --server.port 8502
```

Open forwarded port **8502** in Codespaces and keep its visibility **Private**
(the database can contain research metadata). See the
[dashboard guide](docs/BENCHMARK_DASHBOARD.md) for setup and interpretation.

### MovieLens 32M / HOFM experiment tracking

The registry also tracks HOFM training attempts independently of accuracy results.
Existing checkpoint folders can be indexed **without retraining**, while new
runs can be launched through `python -m src.hofm_experiments run`. Verified
evaluation JSON reports can later be attached to the corresponding attempt.
A saved checkpoint or process exit never creates an invented NDCG score.

See [HOFM experiment tracking](docs/HOFM_EXPERIMENT_TRACKING.md).

### Full MovieLens 32M film-title and tag Sentence Transformer embeddings

A streaming, timestamp-safe encoder can embed **every movie title** and every
**distinct normalized user-generated tag phrase** with Sentence Transformers.
It stores `movie_ids.npy`, normalized title/tag vectors, a timestamped
SQLite tag-event index and a manifest. Movie-tag representations require
an explicit historical cutoff, preventing future tags from leaking into
past ranking evaluations.

```bash
python -m pip install sentence-transformers
python -m src.encode_ml32m_text encode --data-dir data/ml-32m \
  --output artifacts/semantic/ml32m_minilm_full_v1
```

See [Full-catalog MovieLens 32M text embeddings](docs/ML32M_SENTENCE_EMBEDDINGS.md)
for reproducible training-safe features, time-filtered tag pooling,
larger model variants and how to align movie IDs.

### HOFM4 + JointBCD: true fourth-order factorization

`fast_unified_hofm4.py` extends the pretrained HOFM3/JointBCD model with a
new **learnable, exact fourth-order ANOVA FM** over user, movie and 15
temporal preference fields. The existing HOFM3 scoring path is preserved,
while quartic parameters receive their own regularization. A separate
80-epoch recipe, run label and `hofm4_jointbcd` model ID keep existing
HOFM3 checkpoints and leaderboard results independent.

```bash
python fast_unified_hofm4.py --self-test
bash scripts/train_hofm4_jointbcd.sh --dry-run
bash scripts/train_hofm4_jointbcd.sh
```

This extension requires the advanced HOFM3 research patches installed in
your Codespace; the core trainer is **not** a standalone, CI-integrated
package on fresh clone. See the
[HOFM4 + JointBCD architecture and training guide](docs/HOFM4_JOINTBCD.md).

### Aggressive HOFM3 / JointBCD: 80-epoch experiment

A new, independent recipe tests **aggressive dropout, L1/L2, third-order
penalties and AdamW regularization** across 80 epochs on the MovieLens 32M
research installation. It preserves earlier runs, uses a new checkpoint
label and records attempts in the SQLite experiment registry.

```bash
bash scripts/train_hofm3_aggressive80.sh --dry-run
bash scripts/train_hofm3_aggressive80.sh
```

The second command **starts training** and requires the existing research
patches and pretrained parent checkpoint in your Codespace. See the
[80-epoch recipe](docs/AGGRESSIVE_HOFM3_80_EPOCHS.md) for hyperparameter
deltas and safety notes. It does not claim to improve NDCG before evaluation.

### LambdaRank reranking for MovieLens 32M

A standalone **LightGBM LambdaRank** stage accepts frozen `score_*`
predictions from HOFM3 + JointBCD, BPR, Temporal FM, DeepFM and other models.
It trains independent per-model rerankers and an optional score-stacked
ensemble, evaluating all models on the **same candidate pools**. Leakage
guards enforce disjoint decision/query IDs and a documented protocol.
Measured baseline and reranked NDCG@K results can be imported into the
SQLite dashboard; no benchmark is seeded without evaluated predictions.

```bash
python -m src.lambdarank_rerank --help
```

See [LambdaRank reranking](docs/LAMBDARANK_RERANKING.md) for the score
export schema, exact candidate joins, chronological holdout manifest and
the complete training/experiment command.

## Research questions

1. How much do collaborative factors improve full-catalogue top-K recommendation over popularity?
2. When does a content or semantic representation help over sparse interaction history?
3. Can learned ranking improve **NDCG@10** without hiding candidate-retrieval failures?
4. How do chronology, unavailable impression logs and missing-not-at-random ratings limit offline conclusions?

The implemented baseline includes popularity, TruncatedSVD, TF-IDF genre profiles, a fixed hybrid, optional two-tower embeddings, and LightGBM LambdaMART. TruncatedSVD is **not** implicit ALS. Title/genre embeddings are not equivalent to plot understanding. No production A/B uplift is claimed.

## Evaluation protocol

- Ratings **≥4 stars** are treated as observed positive interactions; missing ratings are **unknown**, not confirmed dislikes.
- For eligible users, the last three positive events are reserved in order for ranker training, validation, and test.
- The baseline evaluates full-catalogue recommendations with previously seen movies excluded, reporting Recall@K, NDCG@K and catalogue coverage.
- The split is **per user**, not a globally time-separated holdout. Cross-user time leakage and popularity bias remain limitations.
- Metrics are written to 'artifacts/metrics.json' after a real run. The UI does not invent scores. Avoid tuning on the test split.

See [Model card](reports/MODEL_CARD.md) and [Research status](docs/RESEARCH_STATUS.md) for limitations and the different protocols used in larger experiments.

## Repository layout

```text
app.py                    Streamlit known-user, cold-start, baseline demo
benchmark_dashboard.py    Read-only SQLite experiment leaderboard UI
src/                      Reproducible MovieLens Small implementations
tests/                    Data-free synthetic regression tests
notebooks/                Exploratory and evaluation notebooks
reports/MODEL_CARD.md     Methodology, assumptions, limitations
docs/RESEARCH_STATUS.md   1M/32M prototype inventory and integration status
research-patches/         Archived experimental source patches (not installed by default)
.github/workflows/        Automated baseline tests
```

## Engineering and reproduction notes

- Python 3.12 CI exercises synthetic tests without downloading external datasets.
- Large datasets, generated reports, checkpoints, caches and local secrets are kept out of Git.
- For an optional Docker-based UI, run 'docker build -t reco-demo .' and 'docker run -p 8501:8501 reco-demo'. A build without a trusted model intentionally opens the setup page.
- 'joblib' and PyTorch checkpoints must come from **trusted sources**, because deserialization can execute code.
- The experimental patch ZIPs are retained for provenance; they are **not a substitute for a clean, consolidated source release**. Consolidating and validating them is explicitly tracked as follow-up work.

## Data, ethics and licensing

Project **source code** is MIT licensed; MovieLens data are separately licensed by GroupLens Research (University of Minnesota) and are not included in this repository. Use and redistribution may require permission, particularly for commercial applications. [Dataset and terms](https://grouplens.org/datasets/movielens/). No affiliation with or endorsement by GroupLens is implied.

For a concise contributor workflow, see [CONTRIBUTING.md](CONTRIBUTING.md).