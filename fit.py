"""Fit scene flow models."""

import argparse
import importlib
from pathlib import Path
from random import sample, seed
from typing import List, Optional, Tuple

import numpy as np
import torch
from nntime import export_timings
from torch.utils.data import Dataset
from tqdm import tqdm

import models
import models.base
from utils import options
from utils.torch import move_to_device


def fit(
    model: models.base.SceneFlow,
    data_loader: Dataset,
    output_root: Path,
    subset_size: int = 0,
    chunk: Tuple[int, int] = (1, 1),
    files: Optional[List[Path]] = None,
) -> None:
    """Fit a scene flow model.

    Args:
        model: The scene flow class to fit parameters with.
        data_loader: Loader for the data
        output_root: Root directory to save output files in,
        subset_size: Fit a random subset of the examples for faster testing.
                     If 0 use the whole dataset. Always uses seed 0 and random.shuffle to
                     get a consistent but arbitrarty ordering.
        chunk: Tuple of (N, M) for splitting the input into N
               non-overlapping chunks and only evaluating the Mth chunk.
               Useful for running multiple jobs in parallel.
    """

    if files is not None:
        inds = [data_loader.example_id_to_index(data_loader.file_to_id(f)) for f in files]
    else:
        inds = list(range(len(data_loader)))
        if subset_size > 0:
            seed(0)
            inds = sample(inds, subset_size)
    inds = np.array_split(inds, chunk[0])[chunk[1] - 1]

    for i in tqdm(inds):
        datapoint = data_loader[i]

        pcl_0, pcl_1, ego1_SE3_ego0, flow = (
            datapoint["pcl_0"],
            datapoint["pcl_1"],
            datapoint["ego1_SE3_ego0"],
            datapoint["flow"],
        )

        example_name = data_loader.example_id(i)
        model.fit(pcl_0, pcl_1, ego1_SE3_ego0, flow, example_name=example_name)
        pred_flow, is_dynamic = model(pcl_0, ego1_SE3_ego0)
        model.save_parameters(output_root / example_name)
    export_timings(model.flow, output_root / "timing.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="fit", description="Fit scene flow models to pairs of lidar scans")

    parser.add_argument("name", type=str, help="name to save results under")
    parser.add_argument("model", help="which type of model to fit")
    parser.add_argument(
        "model_args",
        nargs="*",
        type=str,
        help="override model configuration with opt.subopt.key=val",
    )
    parser.add_argument("--dataset", type=str, default="argoverse2", choices=["argoverse2", "nuscenes"])
    parser.add_argument("--outputs", type=str, default="outputs", help="place to store outputs")
    parser.add_argument(
        "--inputs",
        type=str,
        default="inputs",
        help="place to find inputs, looks for <inputs>/av2/sensor/<split>",
    )
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--mask-file", type=str, default="val-masks.zip", help="mask archive file location")
    parser.add_argument(
        "--chunks",
        type=int,
        default=1,
        help="split the job into N deterministic chucnks and only process one",
    )
    parser.add_argument("--chunk_number", type=int, default=1, help="which chunk to process")
    parser.add_argument("--subset", type=int, default=0, help="If >0 only use SUBSET random examples from the dataset.")
    parser.add_argument("--files", nargs="*", type=str, default=None, help="explicit list of files to process")

    args = parser.parse_args()

    output_root = Path(args.outputs) / args.name
    output_root.mkdir(exist_ok=True, parents=True)

    cli_args = options.parse_arguments([args.model] + args.model_args)
    model_cfg = options.set(cli_args)

    m = importlib.import_module(f"models.{model_cfg.model_name}")
    model = m.SceneFlow(model_cfg, output_root=output_root)

    options.save_options_file(model.opt, output_root / "options.yaml")

    if args.dataset == "argoverse2":
        import data.argoverse2

        data_loader = data.argoverse2.Dataloader(data_root=args.inputs, split=args.split, mask_file=args.mask_file)
    elif args.dataset == "nuscenes":
        import data.nuscenes

        data_loader = data.nuscenes.Dataloader(data_root=args.inputs)

    fit(
        model,
        data_loader,
        subset_size=args.subset,
        chunk=(args.chunks, args.chunk_number),
        output_root=output_root,
        files=args.files,
    )
