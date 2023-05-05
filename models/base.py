"""Base interface for an optimization base scene flow model."""

from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
from kornia.geometry.liegroup import Se3


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
        self.output_root = output.root

    def __call__(self, pcl_0: torch.Tensor) -> torch.Tensor:
        """Evaluate the model on a a set of points.

        Args:
            pcl_0: (N,3) tensor of locations to evaluate the flow at.

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
