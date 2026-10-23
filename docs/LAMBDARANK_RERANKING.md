# LambdaRank reranking for MovieLens 32M and other models

This implementation uses **LightGBM LambdaRank / LambdaMART** to *rerank*
candidates from the existing models. It is separate from the `src.train`
MovieLens Small LambdaMART baseline.

Supported score columns include `score_hofm3`,
`score_hofm3_semi25_jointbcd`, `score_bpr`, `score_temporal_fm`,
`score_deepfm`, `score_svd`, etc. You can compare one model at a time,
then train a **score-stacked ensemble** on all selected models. Additional
`feature_*` columns may carry strictly past-only features such as log
popularity, item age as of the prediction cutoff, historical user counts,
genre similarity and tag-similarity features.

**Important:** the advanced HOFM3 trainer in the Codespace comes from research
patches and is **not yet exported as a unified checked-in Python package**.
This PR implements the complete score-table-to-reranker and evaluation stage,
not an automatic inference adapter that can load arbitrary research-patch
checkpoints. Export the *real* base-model scores first.

## Scientific objective

Your reported HOFM3 + JointBCD NDCG@10 = **0.13** is a reference, **not**
a score we can insert in the registry without its corresponding candidate
pool, holdout manifest and evidence. A higher sampled-negative NDCG is
not proof of improvement over full-catalogue NDCG.

The reranker **cannot recover** positive items absent from candidate retrieval.
Use the same candidate set and the same evaluation holdout for every model.
If you want genuinely full-catalogue NDCG, your test file must include every
eligible unseen item (or have an exact, independently verified matching
full-catalogue protocol). Ranking 100 or 500 candidates is **candidate-conditional**
NDCG, not full-catalogue NDCG.

## 1. Prepare frozen score tables

Create the following files from your actual precomputed ranking pipeline:

```
artifacts/rerank/train.csv
artifacts/rerank/valid.csv
artifacts/rerank/test.csv
```

Each contains one row per **query and candidate** with columns:

| Column | Meaning |
| --- | --- |
| `query_id` | Unique recommendation decision ID, e.g. user + timestamp + holdout stage |
| `item_id` | Candidate movie ID |
| `label` | Binary held-out observed positive (1) or unlabeled/other candidate (0) |
| `total_relevant` | Optional if all positives are present; required when any known positive was not retrieved (same positive count repeated across each query's candidate rows) |
| `score_hofm3` | Frozen HOFM3/JointBCD score available at this decision |
| `score_bpr` | Frozen BPR score for the **same** query/movie |
| `score_temporal_fm` | Optional frozen Temporal FM score |
| `score_deepfm` | Optional frozen DeepFM score |
| `feature_historical_popularity` | Optional positive-history-only numerical feature |
| `feature_...` | Other numeric point-in-time features |

All three files must contain the **same feature/score columns**. Query IDs
must not overlap between train/validation/test. Different chronological
requests from the *same user* need distinct query IDs. Every query must have
at least two candidates, at least one positive, and at least one unlabeled
candidate in **ranker training**; duplicate (query_id,item_id) pairs are rejected.
Validation and test may include queries with **zero retrieved positives** when
`total_relevant` is provided, so missed positives count as NDCG/Recall **zero**
instead of silently disappearing from the test cohort. Set `total_relevant`
to the actual number of held-out relevant items for that query; it is never
given to the ranking model as an input feature.

Rows are sorted and passed with true query-group sizes to LightGBM.

**Leakage warnings:** no future ratings, validation/test-derived counts, test
ranks, target leakage, or candidate retrieval trained on later labels may
enter any `feature_*` or `score_*` column. `label=0` means *not observed
positive in this dataset*, **not necessarily a real dislike or impression**.
The program validates schema but cannot prove historical feature provenance:
you are required to certify it in the manifest.

### Score-table join utility

If each model exports only `query_id,item_id,score`, prepare one base file
with `query_id,item_id,label`, optionally `total_relevant`,
and any `feature_*` columns. Combine scores
by **exactly matching keys**; mismatches, duplicates and missing scores fail:

```bash
python -m src.lambdarank_rerank join \
  --base artifacts/rerank/train_base.csv \
  --score hofm3=artifacts/rerank/train_hofm3.csv \
  --score bpr=artifacts/rerank/train_bpr.csv \
  --score temporal_fm=artifacts/rerank/train_temporal_fm.csv \
  --output artifacts/rerank/train.csv
```

Repeat the same join for `valid` and `test`. Only include base models
you actually scored. Never fill unavailable scores with invented values.
Parquet is also supported when `pyarrow` is installed (already installed
by Streamlit dependencies).

## 2. Make the evaluation manifest

Write `artifacts/rerank/evaluation_manifest.json`. The following placeholders
**must be replaced with the actual protocol and provenance**:

```json
{
  "dataset": {
    "id": "ml-32m",
    "name": "MovieLens 32M",
    "version": "32m"
  },
  "protocol": {
    "id": "REPLACE_WITH_YOUR_ACTUAL_PROTOCOL_ID",
    "dataset_id": "ml-32m",
    "split_method": "REPLACE_WITH_ACTUAL_TEMPORAL_SPLIT",
    "candidate_policy": "REPLACE_WITH_ACTUAL_POOL_AND_EXCLUSION_RULE",
    "label_rule": "REPLACE_WITH_OBSERVED_POSITIVE_DEFINITION",
    "cohort_signature": "REPLACE_WITH_ACTUAL_HASH_OF_EVALUATED_QUERIES",
    "cutoff": "REPLACE_WITH_ACTUAL_CUTOFF"
  },
  "features_from_past_only": true,
  "frozen_base_models": true,
  "test_untouched_for_training": true,
  "query_id_is_user": false
}
```

Set `query_id_is_user=true` **only** when each test query is a distinct
user; otherwise the registry will store the real query count separately
and will not misrepresent it as a user count.

The positive labels in validation and test files are evaluation-only.
Do not tune the base model on its own test labels or train candidate
retrieval using holdout labels. The first-stage model must be trained before
the ranker-train targets, or use out-of-fold scores.

## 3. Train LambdaRank with each model and the ensemble

```bash
cd /workspaces/Reco
python -m pip install -r requirements.txt

OMP_NUM_THREADS=4 python -m src.lambdarank_rerank train \
  --train artifacts/rerank/train.csv \
  --valid artifacts/rerank/valid.csv \
  --test artifacts/rerank/test.csv \
  --manifest artifacts/rerank/evaluation_manifest.json \
  --models hofm3,bpr,temporal_fm,deepfm \
  --ensemble \
  --k 10 --trees 300 --early-stop 30 \
  --leaves 15 --min-leaf 20 --l2 10 \
  --threads 4 --seed 42 \
  --output artifacts/rerank/lambdarank_run01 \
  --db artifacts/benchmarks.sqlite
```

Use only model names present in your input score columns; you can omit
`--models` to train **every** supplied score column. If just HOFM3
is available, specify `--models hofm3_semi25_jointbcd` and leave off
`--ensemble`.

Each model gets its own `lambdarank_<name>.txt` (native LightGBM
booster, not untrusted pickle), test prediction CSV, baseline metric report,
and LambdaRank test report. The `ensemble` variant uses **all selected
frozen scores** plus the same historical metadata features.

Early stopping uses **validation NDCG@K including unretrieved positives**,
computed against each query's total relevant count when present. Test is
evaluated only after fitting. Hyperparameters are controlled on validation,
not test.

Run reports and metrics are imported into the same experiment SQLite database
with a **specific protocol ID**. The dashboard at port 8502 then shows
the LambdaRank variants and their frozen baselines together under that protocol.

Run folders must be new; existing output directories are never overwritten.

## 4. Inspect measured results

```bash
python -m src.benchmark_registry leaderboard \
  --dataset ml-32m \
  --protocol REPLACE_WITH_YOUR_ACTUAL_PROTOCOL_ID \
  --metric ndcg --k 10

streamlit run benchmark_dashboard.py --server.port 8502
```

The output reports `ndcg@10`, `recall@10`, `hit_rate@10`, and evaluated
query counts, all **conditional on exactly the candidates in the file**.
It also records SHA256 hashes of train/validation/test/manifest inputs,
feature names, seeds, Git revision, source paths and training parameters.

## 5. What is not included yet?

There is no claim that the patch-based HOFM3 model can be loaded in a
fresh clone, or that score files already exist. To obtain real test scores
for all 32M models, integrate each checkpoint's point-in-time `score_user`
export into the shared candidate schema. This stage is intentionally
separate from the reranking implementation so no research patches or
checkpoints are overwritten.

For a first controlled test, export **HOFM3 score + historical popularity**
on the same train/valid/test query pools as your original 0.13 run, then
add BPR/Temporal FM/DeepFM scores after validating that the candidate
IDs and labels match exactly.
