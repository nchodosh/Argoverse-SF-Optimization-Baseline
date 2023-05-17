import pickle
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import NearestNDInterpolator
from tqdm import tqdm

from utils import geometry

from . import base


class WorldSheet(base.WorldSheet):
    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt

    def fit(self, pcl_train, pcl_test=None, quiet=False):
        x = pcl_train[:, 1:]
        if self.opt.inpt.type == "unit":
            x = geometry.yp_to_unit(x)
        x = x.numpy()
        d = pcl_train[:, 0].numpy()
        if self.opt.out.invert_depth:
            d = 1 / d

        self.interp = LinearNDInterpolator(x, d)
        self.nearest = NearestNDInterpolator(x, d)

    def forward(self, yp):
        x = yp
        if self.opt.inpt.type == "unit":
            x = geometry.yp_to_unit(yp)
        output = self.interp(x)
        missing = np.isnan(output)
        if np.any(missing):
            output[missing] = self.nearest(x[missing])
        return torch.from_numpy(output).reshape(-1, 1)

    def planes(self, coords):
        breakpoint()

    def depth_with_grad(self, coords):
        device = coords.device
        tri = self.interp.tri
        sinds = tri.find_simplex(coords.detach().cpu().numpy())
        valid = sinds > -1
        sinds = sinds[valid]
        simps = tri.simplices[sinds]

        homo_coords = torch.nn.functional.pad(coords[valid], (0, 1), value=1)
        coeff = self.eqs_inv[sinds] @ homo_coords[..., None]
        d = (coeff * self.values[simps]).squeeze().sum(-1)

        if self.opt.out.invert_depth:
            d = 1 / d

        depth = torch.zeros(len(coords), device=device)
        depth[valid] = d.float()
        depth[~valid] = np.nan
        return depth

    def depth(self, coords):
        d = self.forward(coords)
        if self.opt.out.invert_depth:
            d = 1 / d
        return d

    def load_parameters(self, filename):
        with open(Path(filename).with_suffix(".pkl"), "rb") as f:
            self.interp, self.nearest = pickle.load(f)
            self.values = torch.from_numpy(self.interp.values).to(self.opt.device)
            eqs = (
                torch.from_numpy(
                    np.pad(self.interp.points[self.interp.tri.simplices], ((0, 0), (0, 0), (0, 1)), constant_values=1)
                )
                .to(self.opt.device)
                .permute([0, 2, 1])
            )
            self.eqs_inv = torch.linalg.inv(eqs).float()

    def save_parameters(self, filename):
        with open(Path(filename).with_suffix(".pkl"), "wb") as f:
            pickle.dump((self.interp, self.nearest), f)
