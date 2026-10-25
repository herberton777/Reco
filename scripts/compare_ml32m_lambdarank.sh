#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
python -u -m src.lambdarank_compare --top 4 --k 10 \
  --trees 500 --early-stop 40 --leaves 15 --min-leaf 50 \
  --lr 0.03 --l2 10 --threads 4 --bootstrap 2000 \
  --output "artifacts/rerank/ml32m_compare_$(date -u +%Y%m%dT%H%M%SZ)" "$@"
