import pickle
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import RBFInterpolator
from tqdm import tqdm
from utils import geometry

from . import base

MAX_DEPTH = 80

class WorldSheet(base.WorldSheet):
    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        
    def fit(self, pcl_train, pcl_test=None):
        x = pcl_train[:, 1:]
        if self.opt.inpt.type == 'unit':
            x = geometry.yp_to_unit(x)
        x = x.numpy()
        d = pcl_train[:, 0].numpy()
        if self.opt.out.invert_depth:
            d = 1 / d

        self.interp = RBFInterpolator(x, d, **self.opt.rbf_args.__dict__)


    def forward(self, yp):
        x = yp
        if self.opt.inpt.type == 'unit':
            x = geometry.yp_to_unit(yp)
        try:
            d = self.interp(x.numpy())
        except np.linalg.LinAlgError:
            d = np.ones(len(x)) * np.nan
        return torch.from_numpy(d).reshape(-1, 1)
    
    def depth(self, coords):
        d = self.forward(coords)
        if self.opt.out.invert_depth:
            d = 1 / d
        d = np.clip(d, 0, MAX_DEPTH)
        return d

    def load_parameters(self, filename):
        with open(Path(filename).with_suffix('.pkl'), 'rb') as f:
            self.interp = pickle.load(f)

    def save_parameters(self, filename):
        with open(Path(filename).with_suffix('.pkl'), 'wb') as f:
            pickle.dump(self.interp, f)
        
