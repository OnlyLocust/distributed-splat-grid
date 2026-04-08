# Python Runbook: Split -> Worker -> Stitch

This runbook explains how to run the Python pipeline and generate `chunk_i` for `i = 0..n-1`.

## 1) Activate environment

```bash
conda activate gaussian-splat
cd /mnt/d/Development/Projects/DC50/distributed-gaussian-splat-rendering
```

## 2) Generate chunks (`chunk_0` to `chunk_n-1`)

Use `grid_x = n` (V1 supports `n x 1 x 1` splitting).

```bash
python splitter.py --data_dir data --tasks_dir tasks --grid_x 4
```

This creates:
- `tasks/chunk_0`
- `tasks/chunk_1`
- `tasks/chunk_2`
- `tasks/chunk_3`

## 3) Run one worker on a specific chunk (`chunk_i`)

### Basic low-memory mode (recommended for quick validation)

```bash
python worker.py --task_dir tasks/chunk_1 --results_dir results --basic_mode --basic_max_points 1000
```

You can replace `chunk_1` with any `chunk_i`.

### Full training mode (requires gsplat CUDA backend)

```bash
python worker.py --task_dir tasks/chunk_1 --results_dir results --iterations 500 --batch_size 1 --image_downscale 2
```

## 4) Run workers for all chunks (`i = 0..n-1`)

Example for `n=4`:

```bash
python worker.py --task_dir tasks/chunk_0 --results_dir results --basic_mode --basic_max_points 1000
python worker.py --task_dir tasks/chunk_1 --results_dir results --basic_mode --basic_max_points 1000
python worker.py --task_dir tasks/chunk_2 --results_dir results --basic_mode --basic_max_points 1000
python worker.py --task_dir tasks/chunk_3 --results_dir results --basic_mode --basic_max_points 1000
```

Each command writes one output file:
- `results/chunk_0.ply`
- `results/chunk_1.ply`
- ...

## 5) Stitch all chunk outputs

```bash
python stitcher.py --results_dir results --output stitched_output.ply
```

Final merged output:
- `stitched_output.ply`

## 6) Single-command pipeline (split + one worker + stitch)

This runs only the first chunk returned by splitter (current `main.py` behavior).

```bash
python main.py --data_dir data --grid_x 4 --basic_mode --basic_max_points 1000
```

## Quick reference

- Create `n` chunks: `python splitter.py --grid_x n`
- Run one chunk `i`: `python worker.py --task_dir tasks/chunk_i ...`
- Merge all results: `python stitcher.py --results_dir results --output stitched_output.ply`

## 7) One executable file for full run (takes N)

You can run the whole flow from one script:

```bash
chmod +x run_full_pipeline.sh
./run_full_pipeline.sh 4 basic
```

- `4` = number of chunks (`chunk_0` to `chunk_3`)
- `basic` = low-memory mode (`basic` or `full`)

Examples:

```bash
./run_full_pipeline.sh 2 basic
./run_full_pipeline.sh 3 full
```
