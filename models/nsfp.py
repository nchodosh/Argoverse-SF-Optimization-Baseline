"""Neural Scene Flow Prior model plus extensions."""

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional

import numpy as np
import torch
import torch_geometric.nn as gnn
import tqdm
from kornia.geometry.liegroup import Se3
from kornia.geometry.linalg import transform_points


def trunc_nn(X: torch.Tensor, Y: torch.Tensor, r: float) -> torch.Tensor:
    """Compute the truncated nearest neighbor distnce from X to Y.

    Args:
        X: (N, D) tensor of points.
        Y: (M, D) tensor of points.
        r: The maximum nearest neighbor distance.

    Returns:
        (N,) tensor containing min(r, min_{y} ||x - y||).
    """
    if len(X) == 0:
        return torch.zeros(0).to(X.device)
    assigned = gnn.nearest(X, Y)
    errs = ((X - Y[assigned]) ** 2).sum(-1)
    errs = torch.clamp(errs, 0, r)
    return errs


def trunc_chamfer(X: torch.Tensor, Y: torch.Tensor, r: float) -> torch.Tensor:
    """Compute the a symmetric truncated nearest neighbor distnce between X to Y.

    Args:
        X: (N, D) tensor of points.
        Y: (M, D) tensor of points.
        r: The maximum nearest neighbor distance.

    Returns:
        (N,) tensor containing trunc_nn(X, Y, r) concatenate with trunc_nn(Y, X, r).
    """
    return torch.cat([trunc_nn(X, Y, r), trunc_nn(Y, X, r)], dim=0)


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
        self.e1_SE3_e0: Optional[Se3] = None

    def __call__(self, pcl_0: torch.Tensor) -> torch.Tensor:
        """Evaluate the model on a a set of points.

        Args:
            pcl_0: (N,3) tensor of locations to evaluate the flow at.

        Raises:
            RuntimeError: If this method is called before calling fit().

        Returns:
            flow: (N,3) tensor of flow predictions.
        """
        pred = self.graph(pcl_0)
        if self.opt.arch.motion_compensate:
            if self.e1_SE3_e0 is None:
                raise RuntimeError("Trying to evaluate a model that has not been fit!")
            rigid_flow = transform_points(self.e1_SE3_e0.matrix(), pcl_0) - pcl_0
            pred = pred + rigid_flow
        return pred

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
        early_stopping = EarlyStopping(patience=self.opt.optim.early_patience, min_delta=0.0001)
        self.graph = ImplicitFunction(self.opt).to(self.opt.device)
        self.e1_SE3_e0 = e1_SE3_e0
        if self.opt.arch.motion_compensate:
            pcl_1 = transform_points(e1_SE3_e0.inverse().matrix(), pcl_1)

        pcl_0 = pcl_0.to(self.opt.device)
        pcl_1 = pcl_1.to(self.opt.device)
        if flow is not None:
            flow = flow.to(self.opt.device)

        optim = torch.optim.Adam(
            [dict(params=self.graph.parameters(), lr=self.opt.optim.lr, weight_decay=self.opt.optim.weight_decay)]
        )

        pbar = tqdm.trange(self.opt.optim.iters, desc="optimizing...", dynamic_ncols=True)

        best_loss = float("inf")
        best_params = self.graph.state_dict()
        for _ in pbar:
            optim.zero_grad()

            flow_pred = self.graph(self.opt, pcl_0)

            loss = self.compute_loss(flow_pred, pcl_0, pcl_1)
            if best_loss > loss.detach().item():
                best_loss = loss.detach().item()
                best_params = copy.deepcopy(self.graph.state_dict())
            loss.backward()
            if early_stopping.step(loss):
                break

            optim.step()
            if flow is not None:
                pbar.set_postfix(loss=f"loss: {loss.detach().item():.3f}")
            else:
                epe = (flow_pred - flow).norm(dim=-1).mean().detach().item()
                pbar.set_postfix(loss=f"loss: {loss.detach().item():.3f} epe: {epe:.3f}")

        self.graph.load_state_dict(best_params)

    def compute_loss(self, flow_pred: torch.Tensor, pcl_0: torch.Tensor, pcl_1: torch.Tensor) -> torch.Tensor:
        """Compute the optimization loss.

        Args:
            flow_pred: The predicted flow vectors.
            pcl_0: First point cloud.
            pcl_1: Second point cloud.

        Returns:
            The total loss on the predictions.
        """
        return trunc_chamfer(pcl_0 + flow_pred, pcl_1, self.opt.optim.chamfer_radius).mean()

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


class ImplicitFunction(torch.nn.Module):
    """Coordinate Network for representing flow."""

    def __init__(self, opt: SimpleNamespace) -> None:
        """Create a coordinate network according to the configuration options.

        Args:
            opt: Configuration options.
        """
        super().__init__()
        self.define_network(opt)

    def define_network(self, opt: SimpleNamespace) -> None:
        """Set up the network.

        Args:
            opt: Configuration options.
        """
        # input layer (default: xyz -> 128)

        filter_size = opt.arch.filter_size
        dim_x = 3

        if opt.arch.pos_enc:
            dim_x = 2 * dim_x * opt.arch.L + dim_x

        # hidden layers (default: 128 -> 128)
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(dim_x, filter_size),
                torch.nn.Linear(filter_size, filter_size),
                torch.nn.Linear(filter_size, filter_size),
                torch.nn.Linear(filter_size, filter_size),
                torch.nn.Linear(filter_size, filter_size),
                torch.nn.Linear(filter_size, filter_size),
                torch.nn.Linear(filter_size, filter_size),
                torch.nn.Linear(filter_size, filter_size),
                # output layer (default: 128 -> 3)
                torch.nn.Linear(filter_size, 3),
            ]
        )

        # activation functions
        self.activ: Callable[[torch.Tensor], torch.Tensor] = lambda x: x
        if opt.arch.activ == "relu":
            self.activ = torch.nn.functional.relu
        elif opt.arch.activ == "sigmoid":
            self.activ = torch.nn.functional.sigmoid

    def positional_encoding(self, opt: SimpleNamespace, input: torch.Tensor, L: int) -> torch.Tensor:
        """Apply positional encoding to the input.

        Args:
            opt: Configuration options.
            input: (B,...,N) Tensor containing the input.
            L: Number of frequencies.

        Returns:
            The encoded input.
        """
        shape = input.shape
        freq = 2 ** torch.arange(L, dtype=torch.float32, device=opt.device) * np.pi / opt.arch.pos_scale
        spectrum = input[..., None] * freq  # [B,...,N,L]
        sin, cos = spectrum.sin(), spectrum.cos()  # [B,...,N,L]
        input_enc = torch.stack([sin, cos], dim=-2)  # [B,...,N,2,L]
        input_enc = input_enc.view(*shape[:-1], -1)  # [B,...,2NL]
        return input_enc

    def forward(self, opt: SimpleNamespace, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the network.

        Args:
            opt: Configuration options.
            x: (B,...,3) tensor of input points.

        Returns:
            (N,...,3) tensor of flow predictions.
        """
        # [B,...,2]
        if opt.arch.pos_enc:
            x = torch.cat((self.positional_encoding(opt, x, opt.arch.L), x), dim=-1)
        for layer in self.layers[:-1]:
            x = self.activ(layer(x))
        x = self.layers[-1](x)
        return x


class EarlyStopping(object):
    """Optimization progress tracker for deciding when to stop."""

    def __init__(self, mode: str = "min", min_delta: float = 0.0, patience: int = 10, percentage: bool = False) -> None:
        """Set up the tracker.

        Args:
            mode: Optimization mode (max or min).
            min_delta: The minimum improvment in the optimization metric to count.
            patience: How many iterations without improvement before stopping.
            percentage: If set, interpret pateince as a percentage of the best value.
        """
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best: Optional[torch.Tensor] = None
        self.num_bad_epochs = 0
        self.is_better = lambda a, b: False
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True

    def step(self, metrics: torch.Tensor) -> bool:
        """Record one step of the optimization.

        Args:
            metrics: The optimization value at this step.

        Returns:
            True if the optimization should stop, False otherwise.
        """
        if self.patience == 0:
            return False

        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode: str, min_delta: float, percentage: float) -> None:
        """Create the is_better function and save it.

        Args:
            mode: Optimization mode (max or min).
            min_delta: The minimum improvment in the optimization metric to count.
            percentage: If set, interpret pateince as a percentage of the best value.

        Raises:
            ValueError: If mode is not one of min or max.
        """
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if not percentage:
            if mode == "min":
                self.is_better = lambda a, best: a < best - min_delta
            if mode == "max":
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == "min":
                self.is_better = lambda a, best: a < best - (best * min_delta / 100)
            if mode == "max":
                self.is_better = lambda a, best: a > best + (best * min_delta / 100)
