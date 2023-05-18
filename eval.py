import argparse
from collections import defaultdict
from pathlib import Path
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    Final,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

import numpy as np
import pandas as pd
from av2.evaluation.scene_flow.constants import SceneFlowMetricType
from av2.evaluation.scene_flow.eval import compute_scene_flow_metrics
from av2.utils.typing import NDArrayBool, NDArrayFloat, NDArrayInt
from torch.utils.data import Dataset
from tqdm import tqdm

from models.base import load_model
from utils import geometry


def compute_metrics(
    pred_flow: NDArrayFloat,
    gts: NDArrayFloat,
    is_dynamic: NDArrayBool,
) -> Dict[str, List[Any]]:
    """Compute all the metrics for a given example and package them into a list to be put into a DataFrame.

    Args:
        pred_flow: (N,3) Predicted flow vectors.
        gts: (N,3) Ground truth flow vectors.
        is_dynamic: (N,) Ground truth dynamic labels.
    Returns:
        A dictionary of columns to create a long-form DataFrame of the results from.
        One row for each subset in the breakdown.
    """
    pred_flow = pred_flow.astype(np.float64)
    gts = gts.astype(np.float64)
    is_dynamic = is_dynamic.astype(bool)

    results: DefaultDict[str, List[Any]] = defaultdict(list)

    for motion, mask in [("Dynamic", is_dynamic), ("Static", ~is_dynamic)]:
        subset_size = mask.sum().item()
        gts_sub = gts[mask]
        pred_sub = pred_flow[mask]
        results["Motion"] += [motion]
        results["Count"] += [subset_size]

        # Check if there are any points in this subset and if so compute all the average metrics.
        if subset_size > 0:
            for flow_metric_type in SceneFlowMetricType:
                results[flow_metric_type] += [compute_scene_flow_metrics(pred_sub, gts_sub, flow_metric_type).mean()]
        else:
            for flow_metric_type in SceneFlowMetricType:
                results[flow_metric_type] += [np.nan]
    return results


def eval(weight_files: List[Path], data_loader: Dataset, override_args: List[str]):
    all_metrics: DefaultDict[str, List[Any]] = defaultdict(list)

    for wf in tqdm(weight_files):
        try:
            model, opt = load_model(wf, override_args)
        except FileNotFoundError:
            continue
        example_id = model.parameters_to_example(wf)
        datum = data_loader[data_loader.example_id_to_index(example_id)]

        pcl_0, e1_SE3_e0, flow = datum["pcl_0"], datum["ego1_SE3_ego0"], datum["flow"]
        pred, dynamic = model(pcl_0, e1_SE3_e0)
        is_dynamic = geometry.compute_dynamic_mask(pcl_0, flow, e1_SE3_e0)

        example_metrics = compute_metrics(pred.detach().numpy(), flow.numpy(), is_dynamic.numpy())
        num_subsets = len(list(example_metrics.values())[0])
        all_metrics["Example"] += [str(example_id) for _ in range(num_subsets)]
        for m in example_metrics:
            all_metrics[m] += example_metrics[m]

    df = pd.DataFrame(all_metrics, columns=["Example", "Motion", "Count"] + list(SceneFlowMetricType))
    return df


def weighted_average(x: pd.DataFrame, metric_type: SceneFlowMetricType) -> float:
    """Weighted average of metric m using the Count column.

    Args:
        x: Input data-frame.
        metric_type: Metric type.

    Returns:
        Weighted average over the metric_type;
    """
    total = cast(int, x["Count"].sum())
    if total == 0:
        return np.nan
    averages: float = (x[metric_type.value] * x.Count).sum() / total
    return averages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="eval", description="Do a basic local evaluation of flow models")
    parser.add_argument("dataset", type=str, choices=["argoverse2", "nuscenes"])
    parser.add_argument("weights_dir", type=str, help="path to directory of optimized model weights")
    parser.add_argument(
        "model_args",
        nargs="*",
        type=str,
        help="override model configuration with opt.subopt.key=val",
    )
    parser.add_argument(
        "--inputs", type=str, default="inputs", help="place to find inputs, looks for inputs/dataset directory"
    )
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--mask-file", type=str, default="val-masks.zip", help="mask archive file location")

    args = parser.parse_args()

    if args.dataset == "argoverse2":
        import data.argoverse2

        data_loader = data.argoverse2.Dataloader(data_root=args.inputs, split=args.split, mask_file=args.mask_file)
    elif args.dataset == "nuscenes":
        import data.nuscenes

        data_loader = data.nuscenes.Dataloader(data_root=args.inputs)

    weights_dir = Path(args.weights_dir)
    weight_files = list(weights_dir.glob("*.pt"))

    results_df = eval(weight_files, data_loader, args.model_args)
    results_df.to_parquet(weights_dir / "eval.parquet")

    grouped = results_df.groupby("Motion")
    avg = pd.DataFrame(
        {
            metric_type.value: grouped.apply(lambda x, m=metric_type: weighted_average(x, metric_type=m))
            for metric_type in SceneFlowMetricType
        }
    )
    print(avg)
