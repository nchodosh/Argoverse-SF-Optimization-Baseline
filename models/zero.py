"""Zero Flow."""
import re
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple

import torch
from kornia.geometry.liegroup import Se3
from kornia.geometry.linalg import transform_points

import models.base as base


class SceneFlow(base.SceneFlow):
    """Interface for a scene flow model.

    Args:
        opt: A namespace conftaining the model configuration.
    """

    def __init__(self, opt: SimpleNamespace, output_root: Path) -> None:
        """Create a scene flow model based on the configuration in opt.

        Args:
            opt: A nested namespace specificying the configuration.
        """
        self.opt = opt
        self.flow = torch.nn.Identity()

    def __call__(self, pcl_0: torch.Tensor, e1_SE3_e0: Se3) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the model on a a set of points.

        Args:
            pcl_0: (N,3) tensor of locations to evaluate the flow at.

        Raises:
            RuntimeError: If this method is called before calling fit().

        Returns:
            flow: (N,3) tensor of flow predictions.
            is_dynamic (N,) Dynamic segmentation predictions
        """
        pred = transform_points(e1_SE3_e0.matrix(), pcl_0) - pcl_0
        is_dynamic = torch.zeros(len(pcl_0), dtype=bool)

        return pred, is_dynamic

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
        """
        pass

    def load_parameters(self, filename: Path) -> None:
        """Load saved parameters for the underlying model.

        Args:
            filename: Path to the parameters file produced by save_parameters.

        Raises:
            NotImplementedError: If the subclass has not implemented this.
        """
        pass

    def save_parameters(self, filename: Path) -> None:
        """Save parameters for the underlying model.

        Args:
            filename: Path to svae the parameters to.

        Raises:
            NotImplementedError: If the subclass has not implemented this.
        """
        filename.with_suffix(".pt").touch()

    def parameters_to_example(self, filename: Path) -> str:
        """Get the coresponding example for the given parameter output file.

        Args:
            filename: Path used to save parameters for the model.

        Returns:
            The example_id associated with the filename.
        """
        if re.match("\d+", filename.stem):
            return filename.parent.stem
        return filename.stem
