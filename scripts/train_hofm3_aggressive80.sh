#!/usr/bin/env bash
# Aggressive 80-epoch HOFM3 + JointBCD experiment for the ML-32M research install.
#
# This is a NEW run initialized from the existing pretrained parent checkpoint.
# It never overwrites an earlier experiment or replaces the parent model.
#
# Usage:
#   bash scripts/train_hofm3_aggressive80.sh
#   bash scripts/train_hofm3_aggressive80.sh --label aggressive80_v2
#   bash scripts/train_hofm3_aggressive80.sh --dry-run
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

label=fast_bcd_aggressive80_v1
dry_run=0
while (( "$#" )); do
  case "$1" in
    --label)
      if (( "$#" < 2 )); then
        echo "Error: --label needs a value" >&2
        exit 2
      fi
      label="$2"
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    --help|-h)
      echo "Usage: bash scripts/train_hofm3_aggressive80.sh [--label UNIQUE_LABEL] [--dry-run]"
      echo "80 epochs, 800 steps/epoch, aggressive dropout + L1/L2 + AdamW decay."
      echo "Uses an existing pretrained HOFM3 parent and creates a new run."
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ ! "$label" =~ ^[A-Za-z0-9_]+$ ]]; then
  echo "Error: label must contain only letters, digits and underscores." >&2
  exit 2
fi
if [[ "$label" == fast_bcd || "$label" == bcd_hofm3_research ]]; then
  echo "Error: refusing to use reserved historical run label $label" >&2
  exit 2
fi

parent=fm_all_hofm3semi25_bpr
run_dir="models/ml-32m/unified_hofm3_bcd/${parent}_${label}"
log_file="artifacts/training_logs/${label}_training.log"

# No existing experiment is silently overwritten, even though the trainer
# receives --force-retrain. Use another label to run another 80-epoch attempt.
command=(
  python -u -m src.hofm_experiments run
  --db artifacts/benchmarks.sqlite
  --script fast_unified_regularized.py --
  --dataset ml-32m
  --backend sentence
  --init-from "$parent"
  --preference-checkpoint auto
  --run-label "$label"
  --force-retrain
  --epochs 80
  --min-epochs 80
  --steps-per-epoch 800
  --batch-size 128
  --negatives 2
  --hard-negative-share 0.25
  --hard-pool 8
  --learning-rate 0.00006
  --new-feature-lr 0.00020
  --weight-decay 0.15
  --third-l2 0.0020
  --bcd-dropout 0.30
  --factor-l2 0.0100
  --content-l2 0.0020
  --bcd-l1 0.0040
  --bcd-l2 0.0080
  --validation-users 100
  --validate-every 2
  --patience 999
  --max-candidates 0
  --progress-every 50
  --threads 4
  --device cpu
)

echo "HOFM3 + JointBCD aggressive regularization / 80 epochs"
echo "Experiment label: $label"
echo "Checkpoint: $run_dir"
echo "Log: $log_file"
printf 'Command: '
printf '%q ' "${command[@]}"
printf '\n'

if (( dry_run )); then
  echo "DRY RUN ONLY: no training process, checkpoints, logs or database changes."
  exit 0
fi

if [[ -e "$run_dir" ]]; then
  echo "Error: checkpoint path already exists: $run_dir" >&2
  echo "Preserving existing run. Pass --label UNIQUE_NEW_LABEL." >&2
  exit 3
fi
if [[ -e "$log_file" ]]; then
  echo "Error: training log already exists: $log_file" >&2
  echo "Preserving existing log. Pass --label UNIQUE_NEW_LABEL." >&2
  exit 3
fi
if [[ ! -f fast_unified_regularized.py ]]; then
  echo "Error: fast_unified_regularized.py is missing." >&2
  exit 3
fi

# The exact training dependency graph is only installed in the research
# Codespace; do not imply the advanced ML-32M patch suite works on fresh clones.
python -c 'import torch; from temporal_fm import unified_hofm3_bcd'
mkdir -p "$(dirname "$log_file")" "$(dirname "$run_dir")"
# Claim the intended checkpoint path before launching, avoiding accidental
# concurrent experiments with the same label.
if ! mkdir "$run_dir"; then
  echo "Error: another process claimed $run_dir" >&2
  exit 3
fi

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
echo "Launching 80-epoch experiment. Baseline target: independently measured NDCG@10 ~0.13."
# pipefail propagates trainer failures while tee preserves full stdout/stderr.
"${command[@]}" 2>&1 | tee "$log_file"
