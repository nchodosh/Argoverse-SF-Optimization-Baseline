import argparse
from pathlib import Path

from models.base import load_model
from utils import vis3d

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="vis", description="Visualize continous range models of lidar scans")
    parser.add_argument("dataset", type=str, choices=["argoverse2", "nuscenes"])
    parser.add_argument("visualization", choices=list(vis3d.visualizations.keys()), help="what kind of visualization")
    parser.add_argument("weights_file", type=str, help="path to optimized model weights")
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

    weights_file = Path(args.weights_file)

    model, opt = load_model(weights_file, override=args.model_args)

    datum = data_loader[data_loader.example_id_to_index(model.parameters_to_example(weights_file))]

    vis3d.visualizations[args.visualization](model, datum)

    breakpoint()
