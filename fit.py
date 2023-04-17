"""Fit scene flow models."""

import argparse
import importlib
from pathlib import Path
from random import seed, shuffle
from typing import Tuple

import numpy as np
import torch
from av2.evaluation.scene_flow.utils import (
    get_eval_point_mask,
    get_eval_subset,
    write_output_file,
)
from av2.torch.data_loaders.scene_flow import SceneFlowDataloader
from tqdm import tqdm

import models
import models.base
from utils import options


def fit(
    name: str,
    model: models.base.SceneFlow,
    data_root: str = "inputs",
    split: str = "val",
    output_root: str = "outputs",
    mask_file: str = "val-masks.zip",
    subset_size: int = 0,
    chunk: Tuple[int, int] = (1, 1),
) -> None:
    """Fit a scene flow model.

    Args:
        name: Name of the model to save output under.
        model: The scene flow class to fit parameters with.
        data_root: Root directory containing dataset
                   (e.g. <data_root>/av2/sensor/val).
        split: Split to generate perdictions for (test, train or val),
        output_root: Root directory to save output files in,
        mask_file: Path to the appropriate mask file for choosing input points.
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

    data_loader = SceneFlowDataloader(data_root, "av2", split)
    inds = get_eval_subset(data_loader)
    if subset_size > 0:
        seed(0)
        shuffle(inds)
        inds = inds[:subset_size]
    inds = np.array_split(inds, chunk[0])[chunk[1] - 1]

    for i in tqdm(inds):
        s0, s1, ego1_SE3_ego0, flow_obj = data_loader[i]
        pcl_0 = s0.lidar.as_tensor()[:, :3]
        pcl_1 = s1.lidar.as_tensor()[:, :3]
        flow = flow_obj.flow

        mask0 = get_eval_point_mask(s0.sweep_uuid, Path(mask_file))
        mask1 = torch.logical_and(
            torch.logical_and((pcl_1[:, 0].abs() <= 50), (pcl_1[:, 1].abs() <= 50)).bool(),
            torch.logical_not(s1.is_ground),
        )

        pcl_1 = pcl_1[mask1]
        pcl_0 = pcl_0[mask0]
        if flow is not None:
            flow = flow[mask0]

        model.fit(pcl_0, pcl_1, ego1_SE3_ego0, flow)
        pred_flow = model(pcl_0)
        is_dynamic = pred_flow.norm(dim=-1) >= 0.05
        write_output_file(pred_flow, is_dynamic, s0.sweep_uuid, output_root)


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

    fit(
        args.name,
        model,
        split=args.split,
        data_root=args.inputs,
        output_root=args.outputs,
        subset_size=args.subset,
        chunk=(args.chunks, args.chunk_number),
    )
