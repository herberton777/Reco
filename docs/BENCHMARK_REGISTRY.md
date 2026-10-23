# Benchmark registry: models, experiments and NDCG

This is a **SQLite metadata and result database**, not a claim that all models have
been trained or evaluated. Source code and tests are versioned in GitHub.
The actual database defaults to `artifacts/benchmarks.sqlite`, which is ignored
by Git because it may hold private training metadata or large experiment histories.

The catalog seeds the MovieLens Small models (Popularity, SVD, Genres, Hybrid,
Two-Tower, Neural Hybrid, LambdaMART), research families (BPR, DeepFM, Temporal
FM, third-order HOFM and TIGER-inspired retrieval) and the proposed
HOFM3 Semi25 Joint BCD variant. **Catalog entries have no scores until a
real experiment or independently sourced literature result is imported.**

## Start

From the repository root, using Python 3.12:

```bash
python -m src.benchmark_registry init
python -m src.benchmark_registry models
python -m src.benchmark_registry protocols
```

The CLI uses only the Python standard library; it does not download datasets.

## Import an actual local MovieLens Small evaluation

Train as normal, then load **the real metrics file written by src.train**:

```bash
python -m src.train --no-ranker
python -m src.benchmark_registry import-baseline artifacts/metrics.json
python -m src.benchmark_registry leaderboard \
  --dataset ml-latest-small \
  --protocol ml-small-per-user-3-fullcatalog-v1 \
  --split test --metric ndcg --k 10
```

The import includes test **and** validation metrics for each evaluated model,
plus evaluated-user counts, catalog size, split description, source file digest,
the original timestamp and source file path. Reimporting exactly the same file
is idempotent. A new result is a new run, not an overwrite.

## Import a MovieLens 32M or literature experiment

For custom experiments, create a JSON file with these required fields (numbers
below are **synthetic illustrative values; NOT measured MovieLens scores**):

```json
{
  "model": {
    "id": "hofm3_semi25_jointbcd",
    "name": "HOFM3 Semi25 Joint BCD",
    "family": "higher-order factorization",
    "stage": "ranking",
    "maturity": "research proposal"
  },
  "dataset": {"id": "ml-32m", "name": "MovieLens 32M", "version": "32m"},
  "protocol": {
    "id": "ml32m-chronological-full-v1",
    "dataset_id": "ml-32m",
    "split_method": "global chronological holdout",
    "candidate_policy": "full catalogue excluding previously seen interactions",
    "label_rule": "rating >=4 is an observed positive",
    "cohort_signature": "replace-with-actual-evaluation-cohort-hash",
    "cutoff": "replace-with-actual-timestamp"
  },
  "evaluation_split": "test",
  "source_type": "measured",
  "source_uri": "replace-with-actual-saved-evaluation-report-path",
  "sample_users": 20,
  "seed": 42,
  "git_commit": "replace-with-actual-training-commit-sha",
  "config": {"regularization": 0.1, "third_order_weight_decay": 0.01},
  "metrics": {"ndcg@10": 0.32, "recall@10": 0.4, "recall@200": 0.7},
  "notes": "synthetic schema example only"
}
```

Do **not** import those sample scores as a benchmark. Replace all illustrative
values with actual output and set a new accurate protocol ID for any different
split, candidate pool, negative sampling policy, cohort, or cutoff.

```bash
python -m src.benchmark_registry import-run /path/to/real-run.json
python -m src.benchmark_registry leaderboard \
  --dataset ml-32m --protocol ml32m-chronological-full-v1 \
  --metric ndcg --k 10
python -m src.benchmark_registry export-json > artifacts/benchmark_export.json
```

A literature result uses `"source_type": "literature"` and must cite its real
paper/report in `source_uri`. To see literature-only entries, add
`--source-type literature` to `leaderboard`. Record the paper's *actual*
protocol definition, not merely the same dataset name.

## Data model

- `models`: unique model IDs, family, stage, maturity/status.
- `datasets`: dataset ID, name, version.
- `protocols`: dataset, split, candidates, labels, cohort signature, cutoff.
- `runs`: content-addressed experiment ID; model, dataset, protocol, split,
  measured/literature, sample users, source location/hash, seed, Git revision,
  full JSON hyperparameters and notes.
- `metrics`: value per run, metric name and cutoff K, such as NDCG@10,
  Recall@200, or RMSE.

The registry checks foreign keys, ensures finite metric values, validates rates
within [0,1], rejects reuse of a protocol ID with incompatible definitions,
and keeps literature results separate from local measured runs.

## Scientific comparison rules

A leaderboard is **always scoped** to one dataset, one protocol ID, one
evaluation split, one metric and K, and one source type. Never compare
sampled-negative NDCG@10 with full-catalogue NDCG@10. Separately measure
retrieval Recall@200 / Recall@500 and post-retrieval NDCG@10. A reranker
cannot retrieve positives missing from candidate generation.

Even within one protocol, check the displayed `sample_users` and the
`cohort_signature` before treating differences as significant. Matching
counts alone do not prove an identical test cohort. Protocols must describe
candidate definitions, exclusion rules and train/test cutoff accurately.
For literature, use its authors' exact dataset release, split and candidate
definition; if those differ, give it a distinct protocol ID.

**No fabricated, guessed, or cross-protocol leaderboard numbers are seeded.**
The advanced MovieLens 32M work is still in archived prototype snapshots.
See [Research status](RESEARCH_STATUS.md) for that caveat.

Run tests with `python -m pytest -q tests/test_benchmark_registry.py`.
