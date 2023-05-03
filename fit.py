"""Fit scene flow models."""

import argparse
import importlib
from pathlib import Path
from random import seed, shuffle
from typing import Tuple

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
    name: str,
    model: models.base.SceneFlow,
    data_loader: Dataset,
    output_root: str = "outputs",
    subset_size: int = 0,
    chunk: Tuple[int, int] = (1, 1),
) -> None:
    """Fit a scene flow model.

    Args:
        name: Name of the model to save output under.
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
    output_dir = Path(output_root) / name
    output_dir.mkdir(exist_ok=True, parents=True)
    options.save_options_file(model.opt, output_dir / "options.yaml")

    inds = np.arange(len(data_loader))
    if subset_size > 0:
        seed(0)
        shuffle(inds)
        inds = inds[:subset_size]
    inds = np.array_split(inds, chunk[0])[chunk[1] - 1]

    for i in tqdm(inds):
        datapoint = data_loader[i]

        pcl_0, pcl_1, ego1_SE3_ego0, flow = (
            datapoint["pcl_0"],
            datapoint["pcl_1"],
            datapoint["ego1_SE3_ego0"],
            datapoint["flow"],
        )

        model.fit(pcl_0, pcl_1, ego1_SE3_ego0, flow)
        pred_flow, is_dynamic = model(pcl_0)
        model.save_parameters(output_dir / data_loader.example_id(i))
    export_timings(model.flow, output_dir / "timing.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="fit", description="Fit scene flow models to pairs of lidar scans")

    parser.add_argument("name", type=str, help="name to save results under")
    parser.add_argument("model", choices=models.__all__, help="which type of model to fit")
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
    parser.add_argument("--mask-file", type=str, help="mask archive file location")
    parser.add_argument(
        "--chunks",
        type=int,
        default=1,
        help="split the job into N deterministic chucnks and only process one",
    )
    parser.add_argument("--chunk_number", type=int, default=1, help="which chunk to process")
    parser.add_argument("--subset", type=int, default=0, help="If >0 only use SUBSET random examples from the dataset.")

    args = parser.parse_args()

    cli_args = options.parse_arguments([args.model] + args.model_args)
    model_cfg = options.set(cli_args)

    m = importlib.import_module(f"models.{model_cfg.cfg_name}")
    model = m.SceneFlow(model_cfg)

    if args.dataset == "argoverse2":
        import data.argoverse2

        data_loader = data.argoverse2.Dataloader(args.inputs, args.split, args.mask_file)
    elif args.dataset == "nuscenes":
        raise NotImplementedError("No nuscenes yet")

    fit(
        args.name,
        model,
        data_loader,
        subset_size=args.subset,
        chunk=(args.chunks, args.chunk_number),
    )
