# Aggressive regularization: 80-epoch HOFM3 + JointBCD

This is an experimental **much-stronger-regularization ablation** for the
existing MovieLens 32M HOFM3 / Semi25 / JointBCD training environment.
It is intentionally separate from the 15-epoch `fast_bcd_strongreg_v1`
experiment (user-reported NDCG@10 ≈ 0.13). No results are claimed before
a genuinely held-out evaluation.

## Regularization schedule and controls

| Parameter | 15-epoch strongreg | 80-epoch aggressive |
| --- | ---: | ---: |
| Epochs | 15 | **80** |
| Minimum epochs | 15 | **80** |
| Steps per epoch | 800 | 800 |
| BCD temporal-input dropout | 0.15 | **0.30** |
| Sampled factor L2 | 0.0025 | **0.0100** |
| Dense content L2 | 0.0005 | **0.0020** |
| BCD coefficient L1 | 0.0010 | **0.0040** |
| BCD weight L2 | 0.0020 | **0.0080** |
| AdamW weight decay | 0.08 | **0.15** |
| Existing third-order L2 | 0.0005 | **0.0020** |
| General learning rate | 0.00008 | **0.00006** |
| New-feature learning rate | 0.00035 | **0.00020** |

Unchanged: batch size 128, 2 negatives, hard-negative share 0.25,
hard pool 8, 100 validation users every 2 epochs, full candidate
evaluation requested via `--max-candidates 0`, 4 CPU threads.

The **80 epochs × 800 steps = 64,000 optimizer steps** is a training budget,
not a measured outcome. `--min-epochs 80 --patience 999` prevents ordinary
early stopping from ending the requested run prematurely. Training may still
stop on errors or resource limits.

A high L1 can drive 15 BCD linear coefficients close to zero. Combining
strong dropout, weight decay and large L2 may also produce **underfitting**:
a lower NDCG is possible. We intentionally reduce the learning rates
somewhat to avoid highly variable updates at the larger training budget.
No mathematical guarantee of improvement is implied.

## Run it in an already prepared Codespace

Do not launch two processes against the same checkpoint label. Stop an
earlier run cleanly first. The advanced temporal_fm research patches must
already be installed and usable in the Codespace; they are **not bundled
as a clean install** in GitHub main.

```bash
cd /workspaces/Reco
git fetch origin main
git merge --ff-only origin/main

# Print the EXACT command without starting training or changing the DB:
bash scripts/train_hofm3_aggressive80.sh --dry-run

# Start the full run, writing the SQLite training-attempt metadata and log:
bash scripts/train_hofm3_aggressive80.sh
```

A *new* label is required whenever the old checkpoint path already exists:

```bash
bash scripts/train_hofm3_aggressive80.sh --label fast_bcd_aggressive80_v2
```

The script deliberately refuses to overwrite existing checkpoint
directories or log files. This remains true even though it passes
`--force-retrain` so the underlying trainer starts at epoch 1. The
**parent pretrained checkpoint** named `fm_all_hofm3semi25_bpr` is still
used through `--init-from`. It does not randomly initialize the full
model. It does not mutate the parent checkpoint.

Checkpoint: `models/ml-32m/unified_hofm3_bcd/fm_all_hofm3semi25_bpr_fast_bcd_aggressive80_v1`

Training log: `artifacts/training_logs/fast_bcd_aggressive80_v1_training.log`

Experiment metadata: `artifacts/benchmarks.sqlite` (table
`training_attempts`).

## Inspect training and compare NDCG

```bash
python -m src.hofm_experiments list

streamlit run benchmark_dashboard.py --server.port 8502
```

Training attempts appear in **Experiment history**, not the performance
leaderboard. An NDCG@10 chart appears **only after attaching a valid
measured evaluation report** to the corresponding attempt, with the exact
same chronological test cohort, candidate pool and exclusion rules as
the 0.13 reference. Follow
[HOFM experiment tracking](HOFM_EXPERIMENT_TRACKING.md) and the
[LambdaRank reranker guide](LAMBDARANK_RERANKING.md).

For a fair controlled comparison, use identical user cohort and K,
record validation curves through 80 epochs, and report the best
*validation-selected* checkpoint once on the held-out test set.
Do not compare scores from different candidate sampling setups.
