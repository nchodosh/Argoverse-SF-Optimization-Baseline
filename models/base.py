"""Base interface for an optimization base scene flow model."""

import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple

import torch
from kornia.geometry.liegroup import Se3

from utils import options


class SceneFlow:
    """Interface for a scene flow model.

    Args:
        opt: A namespace conftaining the model configuration.
    """

    def __init__(self, opt: SimpleNamespace, output_root: Path) -> None:
        """Create a scene flow model based on the configuration in opt.

        Args:
            opt: A nested namespace specificying the configuration.
            output_root: The root directory to save output files in.
        """
        self.opt = opt
        self.parameters_glob = "*.pkl"
        self.output_root = output_root

    def __call__(self, pcl_0: torch.Tensor, e1_SE3_e0: Se3) -> torch.Tensor:
        """Evaluate the model on a a set of points.

        Args:
            pcl_0: (N,3) tensor of locations to evaluate the flow at.
            e1_SE3_e0: Relative pose of the ego vehicle in the second frame.

        Raises:
            NotImplementedError: If the subclass has not implemented this.
        """
        raise NotImplementedError()

    def fit(
        self,
        pcl_0: torch.Tensor,
        pcl_1: torch.Tensor,
        e1_SE3_e0: Se3,
        flow: Optional[torch.Tensor] = None,
        example_name: Optional[str] = None,
    ) -> None:
        """Fit the model parameters on a a set of points.

        Args:
            pcl_0: (N,3) tensor containing the first point cloud.
            pcl_1: (M,3) tensor containing the second point cloud
            e1_SE3_e0: Relative pose of the ego vehicle in the second frame.
            flow: (N,3) optional tensor contating the ground truth flow
                   annotations. If supplied, the optimization can report
                progresso on thet test metrics.

        Raises:
            NotImplementedError: If the subclass has not implemented this.
        """
        raise NotImplementedError()

    def load_parameters(self, filename: Path) -> None:
        """Load saved parameters for the underlying model.

        Args:
            filename: Path to the parameters file produced by save_parameters.

        Raises:
            NotImplementedError: If the subclass has not implemented this.
        """
        raise NotImplementedError()

    def save_parameters(self, filename: Path) -> None:
        """Save parameters for the underlying model.

        Args:
            filename: Path to svae the parameters to.

        Raises:
            NotImplementedError: If the subclass has not implemented this.
        """
        raise NotImplementedError()

    def parameters_to_example(self, filename: Path) -> str:
        """Get the coresponding example for the given parameter output file.

        Args:
            filename: Path used to save parameters for the model.

        Raises:
            NotImplementedError: If the subclass has not implemented this.

        Returns:
            The example_id associated with the filename.
        """
        raise NotImplementedError()


def load_model(weights_file: Path, override: Optional[List[str]] = None) -> Tuple[SceneFlow, SimpleNamespace]:
    """From saved parameters, determine and import the class then load the model and weights.

    Args:
        weights_file: Path to the saved weights, the opions.yaml file should be in the same directory
                       or in the parent directory.
        override: CLI arguments to override configuration values.

    Raises:
        FileNotFoundError: If either the weights file or the options file couldn't be located.

    Returns:
        The loaded SceneFlow model and options.
    """
    options_file = weights_file.parent / "options.yaml"
    if not options_file.exists():
        options_file = weights_file.parent.parent / "options.yaml"
    if not options_file.exists():
        raise FileNotFoundError(
            "Corresponding options file needs to be located in the same directory or one directory up from the weights."
        )

    opt = options.load_saved_options(options_file, override)
    module = importlib.import_module(f"models.{opt.model_name}")
    model = module.SceneFlow(opt, output_root=None)

    model.load_parameters(weights_file)
    return model, opt
