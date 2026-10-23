#!/usr/bin/env bash
# New HOFM4 + JointBCD training attempt: exact 4-way FM branch + existing HOFM3.
# Requires the original research patch trainer and HOFM3 pretrained parent.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

label=hofm4_jointbcd_80_v1
dry=0
while (( $# )); do
  case "$1" in
    --label)
      [[ $# -ge 2 ]] || { echo "--label requires a value" >&2; exit 2; }
      label="$2"
      shift 2
      ;;
    --dry-run)
      dry=1
      shift
      ;;
    --help|-h)
      echo "Usage: bash scripts/train_hofm4_jointbcd.sh [--label NEW_LABEL] [--dry-run]"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

if [[ ! "$label" =~ ^[A-Za-z0-9_]+$ ]]; then
  echo "Invalid run label (letters/digits/underscore only)" >&2
  exit 2
fi
parent=fm_all_hofm3semi25_bpr
out="models/ml-32m/unified_hofm3_bcd/${parent}_${label}"
log="artifacts/training_logs/${label}.log"

cmd=(
  python -u -m src.hofm_experiments run
    --db artifacts/benchmarks.sqlite
    --model-id hofm4_jointbcd
    --script fast_unified_hofm4.py --
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
    --new-feature-lr 0.00030
    --weight-decay 0.12
    --third-l2 0.002
    --bcd-dropout 0.25
    --factor-l2 0.01
    --content-l2 0.002
    --bcd-l1 0.003
    --bcd-l2 0.005
    --fourth-rank 24
    --fourth-strength 1.0
    --fourth-dropout 0.20
    --fourth-factor-l2 0.001
    --validation-users 100
    --validate-every 2
    --patience 999
    --max-candidates 0
    --progress-every 50
    --threads 4
    --device cpu
)

echo "HOFM4 + JointBCD: new 80-epoch fourth-order factorization experiment"
echo "Model: hofm4_jointbcd"
echo "Run label: $label"
echo "Training directory: $out"
echo "Log path: $log"
printf 'Command: '
printf '%q ' "${cmd[@]}"
printf '\n'

if (( dry )); then
  echo "DRY RUN: no checkpoint, training or SQLite changes."
  exit 0
fi

if [[ -e "$out" || -e "$log" ]]; then
  echo "Refusing to overwrite an existing HOFM4 checkpoint or log. Choose --label NEW_LABEL." >&2
  exit 3
fi
python -c 'import torch; from temporal_fm import unified_hofm3_bcd'
# The HOFM4 model class has a standalone algebra/gradient test.
python fast_unified_hofm4.py --self-test
mkdir -p "$(dirname "$out")" "$(dirname "$log")"
# Claim a separate launch lock while leaving the new checkpoint
# directory absent: the Python launcher refuses to overwrite existing runs.
lock="${out}.launch_lock"
mkdir "$lock" || { echo "Checkpoint label already reserved" >&2; exit 3; }
trap 'rmdir "$lock" 2>/dev/null || true' EXIT

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
"${cmd[@]}" 2>&1 | tee "$log"
