import importlib
from pathlib import Path

import numpy as np
import torch
from pytorch3d.structures import Meshes

from utils import geometry, options


class WorldSheet:
    def __init__(self, opt):
        self.parameters_suffix = ".pkl"

    def fit(self, pcl_train, pcl_test=None, quiet=False):
        raise NotImplementedError()

    def depth(self, coords):
        raise NotImplementedError()

    def points(self, coords):
        return torch.cat((self.depth(coords), coords), dim=-1)

    def to_mesh(self, coord_grid):
        H, W, _ = coord_grid.shape
        points = geometry.xyz(self.points(coord_grid.reshape(-1, 2))).float()
        faces_2d = []
        for i in range(H - 1):
            for j in range(W - 1):
                faces_2d.append([[i, j], [i + 1, j], [i, j + 1]])
                faces_2d.append([[i + 1, j + 1], [i + 1, j], [i, j + 1]])
        faces = np.ravel_multi_index(np.array(faces_2d).transpose([2, 0, 1]), (H, W)).reshape(-1, 3)
        mesh = Meshes(points[None], torch.from_numpy(faces)[None].int())
        normals = mesh.faces_normals_list()[0]

        face_verts = mesh.verts_packed()[mesh.faces_packed()]
        face_centers = face_verts.mean(dim=1)
        face_centers_ryp = geometry.ryp(face_centers)
        face_units = geometry.yp_to_unit(face_centers_ryp[:, 1:])
        dots = (face_units * normals).sum(-1)
        face_mask = dots.abs() > 0.15

        return Meshes(points[None], torch.from_numpy(faces[face_mask])[None].int())

    def load_parameters(self, filename):
        raise NotImplementedError()

    def save_parameters(self, filename):
        raise NotImplementedError()


def load(dir, device=None):
    path = Path(dir)
    opt_file = path / "options.yaml"
    if not opt_file.exists():
        raise FileNotFoundError(f"{str(opt_file)} not found")

    cfg = options.load_options(opt_file)
    if device is not None:
        cfg.device = device

    m = importlib.import_module(f"sheet_models.{cfg.cfg_name}")
    model = m.WorldSheet(cfg)

    return model, cfg
