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

    def fit(self, pcl_train, pcl_test=None):
        x = pcl_train[:, 1:]
        if self.opt.inpt.type == "unit":
            x = geometry.yp_to_unit(x)
        x = x.numpy()
        d = pcl_train[:, 0].numpy()
        if self.opt.out.invert_depth:
            d = 1 / d

        self.interp = NearestNDInterpolator(x, d)

    def forward(self, yp):
        x = yp
        if self.opt.inpt.type == "unit":
            x = geometry.yp_to_unit(yp)
        output = self.interp(x)
        return torch.from_numpy(output).reshape(-1, 1)

    def depth(self, coords):
        d = self.forward(coords)
        if self.opt.out.invert_depth:
            d = 1 / d
        return d

    def load_parameters(self, filename):
        with open(Path(filename).with_suffix(".pkl"), "rb") as f:
            self.interp = pickle.load(f)

    def save_parameters(self, filename):
        with open(Path(filename).with_suffix(".pkl"), "wb") as f:
            pickle.dump(self.interp, f)
