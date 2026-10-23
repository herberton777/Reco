# Tracking MovieLens 32M HOFM experiments

The benchmark registry stores **training attempts** separately from **measured evaluation results**. This matters for the existing `fast_unified_regularized.py` launcher: a checkpoint or successful process exit does not, by itself, prove a particular NDCG@10.

All records are local in `artifacts/benchmarks.sqlite`, excluded from Git. The scripts use only Python's standard library (the training process still needs its separate scientific dependencies). The tracking wrapper does **not** modify the research trainer or checkpoint files.

## 1. Import an existing checkpoint (no training)

If you have already trained HOFM3, register the folder without rerunning:

```bash
cd /workspaces/Reco
python -m src.benchmark_registry init
python -m src.hofm_experiments register-existing \
  --folder models/ml-32m/unified_hofm3_bcd/YOUR_CHECKPOINT_DIRECTORY \
  --label existing_hofm_run
python -m src.hofm_experiments list
```

Replace `YOUR_CHECKPOINT_DIRECTORY` with a *real* directory containing `weights.pt` or `regularization_config.json`. An imported checkpoint has status `imported`: its original command and training exit status are **unknown**, not invented. A repeated import of the same path/label is idempotent.

## 2. Track a new training attempt

Call the launcher through `src.hofm_experiments` rather than calling the existing Python trainer directly. The `--` separates tracking options from training options:

```bash
python -m src.hofm_experiments run \
  --db artifacts/benchmarks.sqlite \
  --script fast_unified_regularized.py -- \
  --dataset ml-32m \
  --backend sentence \
  --init-from fm_all_hofm3semi25_bpr \
  --preference-checkpoint auto \
  --run-label fast_bcd_reg_db01 \
  --epochs 15 --min-epochs 8 \
  --steps-per-epoch 800 \
  --batch-size 128 \
  --negatives 2 \
  --hard-negative-share 0.25 \
  --hard-pool 8 \
  --learning-rate 0.00008 \
  --new-feature-lr 0.0005 \
  --weight-decay 0.01
```

Use a new label that does not collide with existing saved checkpoints.
The exact model dependencies and trainer arguments must be supported by
your installed research-patch version.

The wrapper records the execution attempt ID, command and parameters, dataset,
model, Python script SHA256, Git SHA (if available), checkpoint directory,
regularization settings if present, timestamps and completion/failed exit code.
It forwards the child process's console output and exit code. If the launcher
fails, the attempt stays visible with status `failed` and **no benchmark metrics**.

```bash
python -m src.hofm_experiments list
```

## 3. Attach a **real** NDCG report

Your evaluator must produce a JSON report matching the documented benchmark
schema in [BENCHMARK_REGISTRY.md](BENCHMARK_REGISTRY.md), including a real model ID,
dataset version, explicit split, candidate/negative-sampling definition, cohort
signature, time cutoff, evaluated-user count, hyperparameters, and actual numeric
metrics. In particular, `ndcg@10` must come from a completed evaluation on the
claimed protocol. There is deliberately no scraping/guessing of values from
training logs, no assumed NDCG for a saved checkpoint, and no direct comparison
between sampled-negative and full-catalogue ranking.

Once you have a fully specified real report:

```bash
python -m src.hofm_experiments attach \
  --label fast_bcd_reg_db01 \
  --report /path/to/actual_evaluation.json

python -m src.benchmark_registry leaderboard \
  --dataset ml-32m --protocol YOUR_ACTUAL_PROTOCOL_ID \
  --metric ndcg --k 10
```

The attach command uses the newest attempt with that label; you can instead
choose an exact `--attempt-id`. It rejects failed or interrupted attempts,
literature reports, model/dataset mismatches and (when known) Git SHA
mismatches. It computes the SHA256 of the actual report file; the source
path and digest are recorded automatically. A measured evaluation can also
be attached to a historical `imported` checkpoint, but its **original training
status remains unknown**.

To inspect all experiments and results, use
`python -m src.hofm_experiments list` and
`python -m src.benchmark_registry export-json`.

## Notes on research integrity

- Training metrics (BPR loss, per-epoch validation) are not interchangeable
  with full-catalogue test NDCG@10 or Recall@200.
- Never select a best model on the final test split.
- Use the same eligible users, timestamp cutoff, catalogue and candidate
  pool when comparing model variants.
- Existing research-patch archives must be installed appropriately; this
  tracker does not claim that the advanced MovieLens 32M trainer is integrated
  into a fresh clone.
- Do not commit your private experiment database, large checkpoint files,
  licensed datasets, or raw logs to GitHub.
