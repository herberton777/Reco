# HOFM4 + JointBCD (fourth-order factorization)

## Architecture

This is a **new, trainable fourth-order FM extension** of your existing
HOFM3/Semi25/JointBCD trainer. It is **not just renaming HOFM3**, nor
is it simply raising the HOFM3 score to the fourth power.

The original 1st–3rd order HOFM3 + JointBCD forward score is preserved.
A new rank-24 quartic interaction is added:

\[
s_{HOFM4}(u,i,t)
 = s_{HOFM3+BCD}(u,i,t)
 + \lambda_4 \sum_{d=1}^{R_4}\sum_{a<b<c<e}
    x_{a,d}\,x_{b,d}\,x_{c,d}\,x_{e,d}.
\]

In the quartic branch the distinct fields are:

- One user-ID embedding
- One movie-ID embedding
- Fifteen **separate** JointBCD temporal preference features, each with
  its own learned rank-24 factor vector

Consequently a degree-4 term may combine four temporal fields,
a user plus three temporal fields, an item plus three temporal fields,
or user × item × two different temporal fields. The first-order through
third-order score still uses the original HOFM3 architecture including its
title/genre/tag/time content fields. **The new quartic branch does not add
a separate fourth-order title/genre/tag static-content tensor**: content
enters the new fourth-order interactions through the BCD features.
This distinction matters when comparing architecture families.

The degree-4 sum is computed *exactly* with a four-stage dynamic program
in O(batch × fields × rank), rather than enumerating all F⁴ combinations.
This is mathematically the degree-4 elementary symmetric polynomial over
the factorized fields; each field is used at most once.

The new parameters are registered as a PyTorch submodule before the
original trainer's optimizer is constructed. The launcher adds an optimizer
guard: if a standard PyTorch optimizer includes the existing HOFM3
embeddings but omits fourth-order parameters, training errors instead
of silently claiming the fourth-order field was learned.

## Existing checkpoint compatibility

This launcher is an **in-process patch** built over the version of
`temporal_fm.unified_hofm3_bcd` already installed in your Codespace.
The current repository contains the HOFM3 *launcher* but **not a
self-contained audited copy of the core advanced trainer**, which
still comes from the research ZIP installation.

The patch expects:

- `UnifiedHOFM3BCD` has `user` and `movie` PyTorch embeddings;
- `bcd_linear` is a `torch.nn.Linear` with 15 inputs;
- `forward(batch,bcd)` and
  `third_regularization(positive,negative,coefficient)` are called
  during training;
- The training optimizer normally sees all `model.parameters()`.

If your installed patch differs, the launcher is designed to fail
instead of silently training the wrong architecture. The model
compatibility **cannot be fully verified in GitHub CI without your
installed research-patch implementation**. Test in the Codespace
before committing substantial CPU time.

The parent `fm_all_hofm3semi25_bpr` is **not overwritten**. This run
starts from the pretrained HOFM3 checkpoint, adding randomly initialized
fourth-order factors. Missing quartic tensors are supplied on loading an
HOFM3 parent state dictionary; original parent weights are retained.
Optimizer resume across different architectures is **not supported**.

## Run the 80-epoch HOFM4 training recipe

After checking that your worktree can be fast-forwarded:

```bash
cd /workspaces/Reco
git fetch origin main
git merge --ff-only origin/main

# Independent algebra, dropout and gradient self-test (requires PyTorch):
python fast_unified_hofm4.py --self-test

# Print the full trainer command without running any training:
bash scripts/train_hofm4_jointbcd.sh --dry-run

# Start training, record the attempt in SQLite, save an independent log:
bash scripts/train_hofm4_jointbcd.sh
```

The script sets **80 epochs × 800 steps = 64,000 optimizer steps**,
batch size 128, 2 negatives, semi-hard share 0.25, full-candidate
evaluation flag `--max-candidates 0`, and four CPU threads.

A new label avoids overwriting your earlier HOFM3 checkpoints. To
start another experiment use:

```bash
bash scripts/train_hofm4_jointbcd.sh --label hofm4_jointbcd_80_v2
```

Default checkpoint folder:

```text
models/ml-32m/unified_hofm3_bcd/
  fm_all_hofm3semi25_bpr_hofm4_jointbcd_80_v1
```

The directory remains under `unified_hofm3_bcd` because this is the
**original trainer's checkpoint naming convention**, not because the
model is still third-order. The SQLite model ID is
`hofm4_jointbcd`, distinct from the previous HOFM3 entry.

Log file: `artifacts/training_logs/hofm4_jointbcd_80_v1.log`.

### Regularization and optimizer hyperparameters

| Parameter | New HOFM4 recipe |
| --- | ---: |
| Quartic latent rank | 24 |
| Quartic score multiplier | 1.0 |
| Quartic BCD input dropout | 0.20 |
| Quartic sampled embedding/field L2 | 0.001 |
| Existing BCD input dropout | 0.25 |
| Existing factor L2 | 0.010 |
| Existing content L2 | 0.002 |
| Existing BCD L1 | 0.003 |
| Existing BCD L2 | 0.005 |
| Existing HOFM3 cubic L2 | 0.002 |
| AdamW weight decay | 0.12 |
| Overall learning rate | 0.00006 |
| New-feature LR | 0.00030 |

These are **starting hyperparameters**, not measured optimal values.
Fourth-order interactions have more capacity and can severely overfit;
strong L1/L2 and dropout can conversely make the quartic contribution
vanish. Compare several seeds and log training/validation curves.

## Compare fairly to HOFM3

Your current reference is **reported NDCG@10 ≈ 0.13** from HOFM3+JointBCD.
A proper HOFM4-versus-HOFM3 comparison requires **exactly the same**
MovieLens 32M users, cutoff, candidate construction, observed-positive
definition, exclusion rules, metric and K.

Only measured evaluation results should enter the SQLite leaderboard,
not an assumed score simply because a checkpoint was written. The
training-attempt tracker logs model ID, command, checkpoints,
regularization and execution status. Actual NDCG/Recall reports can
be linked with the existing registry tools:

```bash
python -m src.hofm_experiments list
streamlit run benchmark_dashboard.py --server.port 8502
```

The dashboard leaderboard populates after attaching a valid, measured
evaluation JSON report. The same new fourth-order scorer can then be
exported to `score_hofm4_jointbcd` for the LambdaRank second stage,
but **the patch-specific candidate score exporter is a separate
integration task and is not implemented by this launcher**.

## Tests and limitations

- `tests/test_hofm4_interaction.py`: quartic formula against explicit
  enumeration, gradient propagation, parent state loading, optimizer
  parameter-group inclusion, dropout.
- `tests/test_hofm4_recipe.py`: CLI dry run, 80-epoch flags, safe labels.
- PyTorch-dependent tests skip on CI without Torch; the dependency-free
  CLI tests always execute.
- Actual training on MovieLens 32M has not run in GitHub CI and remains
  a Codespace integration check before claims of working end-to-end metrics.
