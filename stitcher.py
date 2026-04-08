import argparse
import os
from typing import List

import numpy as np

from ply_io import read_gaussian_ply, write_gaussian_ply


class Stitcher:
    def __init__(self, results_dir: str = "results", output_path: str = "stitched_output.ply") -> None:
        self.results_dir = results_dir
        self.output_path = output_path

    def merge_ply_files(self, ply_paths: List[str]) -> str:
        if not ply_paths:
            raise ValueError("No PLY files provided for stitching.")

        means_all = []
        sh_dc_all = []
        opacities_all = []
        scales_all = []
        quats_all = []

        for p in ply_paths:
            data = read_gaussian_ply(p)
            means_all.append(data["means"])
            sh_dc_all.append(data["sh_dc"])
            opacities_all.append(data["opacities"])
            scales_all.append(data["scales"])
            quats_all.append(data["quats"])

        write_gaussian_ply(
            self.output_path,
            means=np.concatenate(means_all, axis=0),
            sh_dc=np.concatenate(sh_dc_all, axis=0),
            opacities=np.concatenate(opacities_all, axis=0),
            scales=np.concatenate(scales_all, axis=0),
            quats=np.concatenate(quats_all, axis=0),
        )
        return self.output_path

    def run(self) -> str:
        ply_paths = [
            os.path.join(self.results_dir, n)
            for n in sorted(os.listdir(self.results_dir))
            if n.lower().endswith(".ply")
        ]
        return self.merge_ply_files(ply_paths)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stitch chunk PLY files into global output.")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output", type=str, default="stitched_output.ply")
    args = parser.parse_args()

    stitcher = Stitcher(results_dir=args.results_dir, output_path=args.output)
    out = stitcher.run()
    print(f"[Stitcher] wrote stitched output: {out}")


if __name__ == "__main__":
    main()
