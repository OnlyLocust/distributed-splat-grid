#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_full_pipeline.sh 4
#   ./run_full_pipeline.sh 4 basic
#   ./run_full_pipeline.sh 4 full
#
# Args:
#   $1 -> N chunks (grid_x)
#   $2 -> mode: basic|full (default: basic)

if [ "${1:-}" = "" ]; then
  echo "Usage: $0 <N> [basic|full]"
  exit 1
fi

N="$1"
MODE="${2:-basic}"

if ! [[ "$N" =~ ^[0-9]+$ ]] || [ "$N" -lt 1 ]; then
  echo "Error: N must be an integer >= 1"
  exit 1
fi

if [ "$MODE" != "basic" ] && [ "$MODE" != "full" ]; then
  echo "Error: mode must be 'basic' or 'full'"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

echo "[1/3] Generating chunks: N=$N"
python splitter.py --data_dir data --tasks_dir tasks --grid_x "$N"

echo "[2/3] Processing chunks with worker (mode=$MODE)"
for i in $(seq 0 $((N - 1))); do
  echo "  -> chunk_$i"
  if [ "$MODE" = "basic" ]; then
    python worker.py \
      --task_dir "tasks/chunk_$i" \
      --results_dir results \
      --basic_mode \
      --basic_max_points 1000
  else
    python worker.py \
      --task_dir "tasks/chunk_$i" \
      --results_dir results \
      --iterations 500 \
      --batch_size 1 \
      --image_downscale 2
  fi
done

echo "[3/3] Stitching results -> output.ply"
python stitcher.py --results_dir results --output output.ply

echo "Done. Final output: $ROOT_DIR/output.ply"
