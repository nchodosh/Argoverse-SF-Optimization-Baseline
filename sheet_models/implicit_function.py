import inspect
from pathlib import Path

import numpy as np
import torch
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm, trange

import utils.options
from utils import geometry
from utils.geometry import yp_to_unit

from . import base

EVAL_CHUNK_SIZE = 20000
MAX_DEPTH = 55


class WorldSheet(base.WorldSheet):
    def __init__(self, opt):
        super().__init__(opt)
        self.parameters_suffix = ".pt"
        self.opt = opt
        self.graph = Graph(opt)
        self.graph.to(opt.device)
        self.scheduler = getattr(torch.optim.lr_scheduler, opt.optim.sched.type)

    def planes(self, coords):
        normals = self.normals(coords)
        ryp = self.points(coords)
        xyz = geometry.xyz(ryp)

        d = -(xyz * normals).sum(dim=-1, keepdim=True)
        return normals, d

    def depth_with_grad(self, coords):
        x = self.graph.depth(coords)
        if self.opt.out.invert_depth:
            depth = 1 / x
        else:
            depth = x
        depth = depth.clip(0, MAX_DEPTH)
        return depth

    def depth(self, coords):
        x = self.graph.depth(coords.float().to(self.opt.device))
        if self.opt.out.invert_depth:
            depth = 1 / x
        else:
            depth = x
        depth = depth.clip(0, MAX_DEPTH)
        return depth.detach().cpu()

    def normals(self, coords):
        if self.opt.out.type == "snorm":
            xyz = geometry.xyz(self.points(coords)).detach().requires_grad_(True)
            implicit_surface = lambda x: (self.graph._forward(geometry.ryp(x)[..., 1:]) * x).sum(dim=-1) - 1
            surf = implicit_surface(xyz)
            surf.sum().backward()
            n = self.graph.normals(coords)  # xyz.grad.clone()
            return n / n.norm(dim=-1, keepdim=True)

        if self.opt.out.invert_depth:
            to_depth = lambda x: (1 / x).clip(0, MAX_DEPTH)
        else:
            to_depth = lambda x: x.clip(0, MAX_DEPTH)
        implicit_surface = lambda x: (
            x.norm(dim=-1, keepdim=True) - to_depth(self.graph.depth(geometry.ryp(x)[..., 1:]))
        )

        xyz = geometry.xyz(self.points(coords)).detach().requires_grad_(True)
        surf = implicit_surface(xyz)
        surf.sum().backward()
        n = xyz.grad.clone()
        return n / n.norm(dim=-1, keepdim=True)

    def save_parameters(self, filename):
        g = self.graph.cpu()
        torch.save(g.state_dict(), Path(filename).with_suffix(".pt"))

    def load_parameters(self, filename):
        weights = torch.load(filename)
        self.graph.load_state_dict(weights)
        self.graph.to(self.opt.device)


class Graph(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        if self.opt.inpt.type == "polar":
            self.dim_x = 2
            self.L = opt.pos.freqs
        elif self.opt.inpt.type == "unit":
            self.dim_x = 3
            self.L = [opt.pos.freqs[0], opt.pos.freqs[0], opt.pos.freqs[1]]
        self.define_network()

    def define_network(self):
        # input layer (default: xyz -> 128)

        if self.opt.pos.use:
            dim_x = 2 * sum(self.L) + self.dim_x
        else:
            dim_x = self.dim_x

        if self.opt.out.type == "scalar":
            dim_out = 1
        else:
            dim_out = 3

        # hidden layers
        fs = self.opt.layers.nunits
        sizes = [dim_x] + self.opt.layers.n * [fs] + [dim_out]
        self.layers = torch.nn.ModuleList([torch.nn.Linear(i, o) for i, o in zip(sizes, sizes[1:])])

        # activation functions
        self.activ = torch.nn.functional.relu

    def positional_encoding(self, input):  # [B,...,N]
        shape = input.shape
        N = shape[-1]
        enc = []
        for i in range(N):
            freq = (
                2 ** torch.arange(self.L[i], dtype=torch.float32, device=input.device) * np.pi / self.opt.pos.scale
            )  # [L]
            spectrum = input[..., i, None] * freq  # [B,...,L]
            sin, cos = spectrum.sin(), spectrum.cos()  # [B,...,L]
            enc.append(torch.stack([sin, cos], dim=-2).view(*shape[:-1], -1))  # [B,...,2L]
        input_enc = torch.cat(enc, dim=-1).view(*shape[:-1], -1)  # [B, ..., 2L_1 + 2L_2 ...]
        return input_enc

    def _forward(self, x):  # [B,...,2]
        if self.opt.inpt.type == "unit":
            x = yp_to_unit(x)
        if self.opt.pos.use:
            x = torch.cat((self.positional_encoding(x), x), dim=-1)
        for layer in self.layers[:-1]:
            x = self.activ(layer(x))
        x = self.layers[-1](x)

        return x

    def forward(self, x):
        if len(x) <= EVAL_CHUNK_SIZE:
            return self._forward(x)
        else:
            inds = list(range(0, len(x), EVAL_CHUNK_SIZE)) + [len(x)]
            output = self._forward(x[inds[0] : inds[1]])
            for s, e in list(zip(inds[1:], inds[2:])):
                output = torch.cat((output, self._forward(x[s:e])))
            return output

    def normal(self, coords):
        if self.opt.out.type == "snorm":
            return self.forward(coords)
        else:
            raise NotImplementedError("Have not implemented normal calculation for scalar depth")

    def depth(self, coords):
        x = self.forward(coords)
        if self.opt.out.type == "snorm":
            x = (x * yp_to_unit(coords)).sum(dim=-1, keepdims=True)
        return x
