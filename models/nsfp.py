"""Neural Scene Flow Prior model plus extensions."""

import copy
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import tqdm
from kornia.geometry.liegroup import Se3
from kornia.geometry.linalg import transform_points
from nntime import export_timings, set_global_sync, time_this, timer_end, timer_start
from pytorch3d.ops import knn_points
from torch.optim.lr_scheduler import ReduceLROnPlateau

import losses
import models.base as base
import utils
import utils.dotdict
import utils.options
import utils.refine

dummy_module = torch.nn.Linear(1, 1)


def inlier_loss(x, k=0.3):
    return 1 / (1 + torch.exp(-x.abs() / k)) - 1 / 2


def sheet_loss(model, xyz):
    ryp = utils.geometry.ryp(xyz)
    sheet_depth = model.depth_with_grad(ryp[:, 1:]).squeeze()
    invalid = torch.isnan(sheet_depth)
    sheet_depth[invalid] = 0
    err = ((sheet_depth - ryp[:, 0]) ** 2).clip(0, 2)
    err[invalid] = 0
    return err


def plane_loss(model, xyz):
    ryp = utils.geometry.ryp(xyz)
    n, d = model.planes(ryp[:, 1:])
    n = n.detach()
    d = d.detach()
    return (((n * xyz).sum(dim=-1, keepdim=True) + d) ** 2).clip(0, 2).squeeze()


class SceneFlow(base.SceneFlow):
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
        super().__init__(opt, output_root)
        self.e1_SE3_e0: Optional[Se3] = None
        if opt.timing:
            set_global_sync(True)
        else:
            set_global_sync(False)

        if opt.optim.loss.type == "sheet" or opt.optim.loss.type == "plane":
            import sheet_models.base

            self.sheet, self.sheet_cfg = sheet_models.base.load(
                Path(opt.optim.loss.models_root).parent, device=opt.device
            )

    @torch.no_grad()
    def __call__(self, pcl_0: torch.Tensor, e1_SE3_e0: Se3) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the model on a a set of points.

        Args:
            pcl_0: (N,3) tensor of locations to evaluate the flow at.
            e1_SE3_e0: Relative pose of the ego vehicle in the second frame.

        Returns:
            flow: (N,3) tensor of flow predictions.
            is_dynamic (N,) Dynamic segmentation predictions
        """
        if self.opt.arch.motion_compensate:
            pcl_input = transform_points(e1_SE3_e0.matrix(), pcl_0)
        else:
            pcl_input = pcl_0
        pred = self.flow.fw(self.opt, pcl_input.to(self.opt.device)).detach().cpu()
        if self.opt.arch.refine:
            pred = torch.from_numpy(utils.refine.refine_flow(pcl_0.numpy(), pred.numpy()))

        rigid_flow = transform_points(e1_SE3_e0.matrix(), pcl_0) - pcl_0
        if self.opt.arch.motion_compensate:
            is_dynamic = pred.norm(dim=-1) > 0.05
            pred = pred + rigid_flow
        else:
            is_dynamic = (pred - rigid_flow).norm(dim=-1) > 0.05

        return pred, is_dynamic

    def make_optimizer(self, parameters):
        optimizer = getattr(torch.optim, self.opt.optim.type)
        optim_args = utils.dotdict.namespace_to_kwargs(self.opt.optim.args, optimizer)
        return optimizer(parameters, **optim_args)

    def make_scheduler(self, optim):
        scheduler = getattr(torch.optim.lr_scheduler, self.opt.optim.sched.type)
        sched_args = utils.dotdict.namespace_to_kwargs(self.opt.optim.sched.args, scheduler)
        return scheduler(optim, **sched_args)

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
        if self.opt.optim.debug:
            self.log_dir = self.output_root / example_name
            if self.log_dir.exists():
                files = list(self.log_dir.iterdir())
                for f in files:
                    f.unlink()

            self.log_dir.mkdir(exist_ok=True)

        early_stopping = EarlyStopping(patience=self.opt.optim.early_patience, min_delta=0.0001)
        self.flow = Flow(self.opt).to(self.opt.device)

        self.e1_SE3_e0 = e1_SE3_e0
        rigid_flow = (transform_points(self.e1_SE3_e0.matrix(), pcl_0) - pcl_0).detach()
        non_rigid_flow = flow - rigid_flow
        dynamic_mask = non_rigid_flow.norm(dim=-1) > 0.05
        if self.opt.arch.motion_compensate:
            pcl_input = transform_points(e1_SE3_e0.matrix(), pcl_0).detach()
            flow = flow - rigid_flow
        else:
            pcl_input = pcl_0

        pcl_input = pcl_input.to(self.opt.device)
        pcl_1 = pcl_1.to(self.opt.device)
        if flow is not None:
            flow = flow.to(self.opt.device)

        optim = self.make_optimizer(self.flow.parameters())
        sched = self.make_scheduler(optim)

        pbar = tqdm.trange(self.opt.optim.iters, desc=f"{example_name}", dynamic_ncols=True)

        best_loss = float("inf")
        best_params = self.flow.state_dict()

        if self.opt.optim.loss.type == "sheet" or self.opt.optim.loss.type == "plane":
            self.flow.load_sheet(self.sheet, example_name)

        for self.it in pbar:
            timer_start(self.flow, "full_iteration")
            timer_start(self.flow, "opt_iteration")
            fw_flow_pred, bw_flow_pred, loss = self.optimization_iteration(optim, sched, pcl_input, pcl_1)
            timer_end(self.flow, "opt_iteration")

            if best_loss > loss.detach().item():
                best_loss = loss.detach().item()
                best_params = copy.deepcopy(self.flow.state_dict())

            if early_stopping.step(loss):
                self.flow.load_state_dict(best_params)
                fw_flow_pred = self.flow.fw(self.opt, pcl_input)
                epe = (fw_flow_pred - flow).norm(dim=-1)
                epe_d = epe[dynamic_mask].mean().detach().item()
                epe_s = epe[~dynamic_mask].mean().detach().item()
                pbar.set_postfix(loss=f"loss: {loss.detach().item():.3f} epe_s: {epe_s:.3f} epe_d: {epe_d:.3f}")
                break

            if flow is not None:
                epe = (fw_flow_pred - flow).norm(dim=-1)
                epe_d = epe[dynamic_mask].mean().detach().item()
                epe_s = epe[~dynamic_mask].mean().detach().item()
                pbar.set_postfix(
                    loss=f"loss: {loss.detach().item():.3f} epe_s: {epe_s:.3f} epe_d: {epe_d:.3f} lr: {sched._last_lr[0]:.2g}"
                )
            else:
                pbar.set_postfix(loss=f"loss: {loss.detach().item():.3f}")
            timer_end(self.flow, "full_iteration")

        self.flow.load_state_dict(best_params)

    def optimization_iteration(self, optim, sched, pcl_0, pcl_1):
        fw_flow_pred = self.flow.fw(self.opt, pcl_0)
        bw_flow_pred = self.flow.bw(self.opt, pcl_0 + fw_flow_pred)
        loss = self.flow.compute_loss(fw_flow_pred + pcl_0, bw_flow_pred, pcl_0, pcl_1)
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.flow.parameters(), self.opt.optim.grad_clip)
        optim.step()
        if isinstance(sched, ReduceLROnPlateau):
            sched.step(loss)
        else:
            sched.step()

        if self.opt.optim.debug and self.it % 10 == 0:
            self.save_parameters(self.log_dir / f"{self.it:04d}")
        return fw_flow_pred, bw_flow_pred, loss

    def point_gradients(self, pcl_0, pcl_1, e1_SE3_e0):
        if self.opt.arch.motion_compensate:
            pcl_input = transform_points(e1_SE3_e0.matrix(), pcl_0)
        else:
            pcl_input = pcl_0
        fw_flow_pred = self.flow.fw(self.opt, pcl_input)
        bw_flow_pred = self.flow.bw(self.opt, pcl_input + fw_flow_pred)
        pcl_0_def = (pcl_input + fw_flow_pred).detach().requires_grad_(True)
        loss = self.flow.compute_loss(pcl_0_def, bw_flow_pred, pcl_input, pcl_1)
        loss.backward()
        pcl_grad = pcl_0_def.grad.clone()
        return pcl_0_def.detach(), pcl_grad

    def point_loss(self, pcl_0, pcl_1, e1_SE3_e0):
        if self.opt.arch.motion_compensate:
            pcl_input = transform_points(e1_SE3_e0.matrix(), pcl_0)
        else:
            pcl_input = pcl_0
        fw_flow_pred = self.flow.fw(self.opt, pcl_input)
        bw_flow_pred = self.flow.bw(self.opt, pcl_input + fw_flow_pred)
        pcl_0_def = (pcl_input + fw_flow_pred).detach()
        loss = self.flow.compute_loss(pcl_0_def, bw_flow_pred, pcl_input, pcl_1, reduction="none").detach()
        return pcl_0_def, loss

    def load_parameters(self, filename: Path) -> None:
        """Load saved parameters for the underlying model.

        Args:
            filename: Path to the parameters file produced by save_parameters.
        """
        params = torch.load(filename, map_location=self.opt.device)["ckpt"]
        self.flow = Flow(self.opt).to(self.opt.device)
        self.flow.load_state_dict(params)
        if self.opt.optim.loss.type == "sheet" or self.opt.optim.loss.type == "plane":
            self.flow.load_sheet(self.sheet, self.parameters_to_example(filename))

    def save_parameters(self, filename: Path) -> None:
        """Save parameters for the underlying model.

        Args:
            filename: Path to save the parameters to. Suffix will be automatically converted to .pt
        """
        filename = filename.with_suffix(".pt")
        torch.save({"ckpt": self.flow.state_dict()}, filename)

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


class Flow(torch.nn.Module):
    """Flow module."""

    def __init__(self, opt: SimpleNamespace) -> None:
        """Create a flow module."""
        super().__init__()
        self.opt = opt
        self.fw = ImplicitFunction(self.opt).to(self.opt.device)
        self.bw = ImplicitFunction(self.opt).to(self.opt.device)
        self.bw.load_state_dict(self.fw.state_dict())

    def load_sheet(self, sheet, example_name):
        result_file = (Path(self.opt.optim.loss.models_root) / example_name).with_suffix(sheet.parameters_suffix)
        self.sheet = sheet
        self.sheet.load_parameters(result_file)

    @time_this()
    def compute_loss(
        self,
        pcl_0_def: torch.Tensor,
        bw_flow_pred: torch.Tensor,
        pcl_0: torch.Tensor,
        pcl_1: torch.Tensor,
        reduction="mean",
    ) -> torch.Tensor:
        """Compute the optimization loss.

        Args:
            flow_pred: The predicted flow vectors.
            flow_pred: The predicted backward flow vectors.
            pcl_0: First point cloud.
            pcl_1: Second point cloud.

        Returns:
            The total loss on the predictions.
        """
        if reduction == "mean":
            red = torch.mean
        elif reduction == "none":
            red = lambda x: x

        if self.opt.optim.loss.type == "chamfer":
            l = lambda x, y: losses.trunc_chamfer(x, y, 2)
            timer_start(self, "fw_chamf")
            fw_chamf = l(pcl_0_def, pcl_1)
            timer_end(self, "fw_chamf")
            timer_start(self, "bw_chamf")
            bw_chamf = l(pcl_0_def - bw_flow_pred, pcl_0)
            timer_end(self, "bw_chamf")
            full_loss = torch.cat((fw_chamf, bw_chamf))
            return red(full_loss)
        elif self.opt.optim.loss.type == "sheet":
            return red(sheet_loss(self.sheet, pcl_0_def).float())
        elif self.opt.optim.loss.type == "plane":
            return red(plane_loss(self.sheet, pcl_0_def))


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

    def _init_is_better(self, mode: str, min_delta: float, percentage: bool) -> None:
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
