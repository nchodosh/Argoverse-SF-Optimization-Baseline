"""Neural Scene Flow Prior model plus extensions."""

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional, Tuple

import numpy as np
import open3d as o3d
import torch
import tqdm
from kornia.geometry.liegroup import Se3
from kornia.geometry.linalg import transform_points
from nntime import export_timings, set_global_sync, time_this, timer_end, timer_start
from pytorch3d.ops import knn_points

import utils.refine


def homo(pc):
    return np.pad(pc.reshape(-1, pc.shape[-1]), ((0, 0), (0, 1)), constant_values=1)


class SceneFlow:
    """Interface for a scene flow model.

    Args:
        opt: A namespace conftaining the model configuration.
    """

    def __init__(self, opt: SimpleNamespace) -> None:
        """Create a scene flow model based on the configuration in opt.

        Args:
            opt: A nested namespace specificying the configuration.
        """
        self.opt = opt
        set_global_sync(False)
        self.R = None
        self.t = None
        self.flow = Flow(opt)

    def __call__(self, pcl_0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the model on a a set of points.

        Args:
            pcl_0: (N,3) tensor of locations to evaluate the flow at.

        Raises:
            RuntimeError: If this method is called before calling fit().

        Returns:
            flow: (N,3) tensor of flow predictions.
            is_dynamic (N,) Dynamic segmentation predictions
        """
        if self.R is None:
            raise RuntimeError("Model has not been fit")
        pred = pcl_0 @ self.R.T + self.t
        is_dynamic = torch.zeros(len(pcl_0), dtype=bool)

        return pred, is_dynamic

    def fit(
        self,
        pcl_0: torch.Tensor,
        pcl_1: torch.Tensor,
        e1_SE3_e0: Se3,
        flow: Optional[torch.Tensor] = None,
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
        _, trans = self.flow.icp_vanilla(pcl_0.numpy(), pcl_1.numpy())
        self.R = trans[:3, :3]
        self.t = trans[:3, 3]

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


class Flow(torch.nn.Module):
    """Dummy flow module."""

    def __init__(self, opt: SimpleNamespace) -> None:
        """Create a flow module."""
        super().__init__()

    @time_this()
    def icp_vanilla(self, src, tar, init=None):
        src_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src))
        src_pc.estimate_normals()
        tar_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tar))
        tar_pc.estimate_normals()
        p2l = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        trans_init = (
            np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0.0, 0.0, 0.0, 1.0]]) if init is None else init
        )
        reg_p2l = o3d.pipelines.registration.registration_icp(src_pc, tar_pc, 2, trans_init, p2l)
        src_def = (homo(src) @ reg_p2l.transformation.T)[:, :3]
        return src_def, reg_p2l.transformation
