import argparse

from splitter import Splitter
from stitcher import Stitcher
from worker import Worker


def main() -> None:
    parser = argparse.ArgumentParser(description="Distributed Gaussian Splatting V1 (local single-worker simulation).")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--tasks_dir", type=str, default="tasks")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--stitched_output", type=str, default="stitched_output.ply")
    parser.add_argument("--grid_x", type=int, default=1, help="Number of chunks along X axis for splitter (V1 supports Xx1x1).")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--image_downscale", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--basic_mode", action="store_true", help="Run minimal pipeline without gsplat rasterization training.")
    parser.add_argument("--basic_max_points", type=int, default=2000, help="Point cap for basic mode memory control.")
    args = parser.parse_args()

    print("[Main] Step 1/3: Splitter")
    splitter = Splitter(data_dir=args.data_dir, tasks_dir=args.tasks_dir, grid_shape=(args.grid_x, 1, 1))
    chunk_dir = splitter.run()
    print(f"[Main] created task chunk: {chunk_dir}")

    print("[Main] Step 2/3: Worker")
    worker = Worker(
        task_dir=chunk_dir,
        results_dir=args.results_dir,
        iterations=args.iterations,
        batch_size=args.batch_size,
        image_downscale=args.image_downscale,
        lr=args.lr,
        basic_mode=args.basic_mode,
        basic_max_points=args.basic_max_points,
    )
    chunk_result = worker.run()
    print(f"[Main] worker result: {chunk_result}")

    print("[Main] Step 3/3: Stitcher")
    stitcher = Stitcher(results_dir=args.results_dir, output_path=args.stitched_output)
    out = stitcher.run()
    print(f"[Main] stitched output: {out}")


if __name__ == "__main__":
    main()
