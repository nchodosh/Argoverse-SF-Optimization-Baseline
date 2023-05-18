import numpy as np
import torch
from kornia.geometry.linalg import transform_points


def ego_to_sensor(pc, sensor):
    return (pc - sensor.t) @ sensor.rot


def filter_close(xyz, return_mask=False):
    mask = xyz[:, :3].norm(dim=-1) >= 1
    if return_mask:
        return xyz[mask], mask
    return xyz[mask]


def filter_far(xyz, return_mask=False):
    mask = xyz[:, :3].norm(dim=-1) < 35
    if return_mask:
        return xyz[mask], mask
    return xyz[mask]


def filter_range(xyz, return_mask=False):
    _, mc = filter_close(xyz, return_mask=True)
    _, mf = filter_far(xyz, return_mask=True)
    m = mc & mf
    if return_mask:
        return xyz[m], m
    return xyz[m]


def ryp(xyz):
    r = xyz.norm(dim=-1)
    yaw = -torch.atan2(xyz[:, 1], xyz[:, 0])
    pitch = torch.asin(xyz[:, 2] / r)
    return torch.stack((r, yaw, pitch), dim=-1)


def xyz(ryp):
    z = torch.sin(ryp[:, 2]) * ryp[:, 0]
    x = torch.cos(-ryp[:, 1]) * torch.cos(ryp[:, 2]) * ryp[:, 0]
    y = torch.sin(-ryp[:, 1]) * torch.cos(ryp[:, 2]) * ryp[:, 0]
    return torch.stack((x, y, z), dim=-1)


def yp_to_unit(yp):
    dz = torch.sin(yp[..., 1])
    dx = torch.cos(-yp[..., 0]) * torch.cos(yp[..., 1])
    dy = torch.sin(-yp[..., 0]) * torch.cos(yp[..., 1])
    unit = torch.stack((dx, dy, dz), dim=-1)
    return unit


def dense_grid_from_area(ryp, yaw_samples=5000, pitch_samples=5000):
    pitch_rng = (ryp[:, 2].min(), ryp[:, 2].max())
    beam_pitches = torch.linspace(pitch_rng[0], pitch_rng[1], pitch_samples)

    yaw_rng = (ryp[:, 1].min(), ryp[:, 1].max())
    beam_yaws = torch.linspace(yaw_rng[0], yaw_rng[1], yaw_samples)
    yp = torch.stack(torch.meshgrid(beam_yaws, beam_pitches, indexing="ij"), dim=-1)
    return yp


def dense_sample_area(ryp, yaw_samples=2000, pitch_samples=500):
    pitch_rng = (ryp[:, 2].min(), ryp[:, 2].max())
    beam_pitches = torch.linspace(pitch_rng[0], pitch_rng[1], pitch_samples)

    yaw_rng_1 = (ryp[:, 1].min(), ryp[:, 1].max())
    yaw_rng_2 = ((ryp[:, 1] % (2 * np.pi)).min(), (ryp[:, 1] % (2 * np.pi)).max())
    yaw_rng_2_normalized = (yaw_rng_2[0] - np.pi, yaw_rng_2[1] - np.pi)
    yaw_rng = yaw_rng_1 if yaw_rng_1[1] - yaw_rng_1[0] < yaw_rng_2[1] - yaw_rng_2[0] else yaw_rng_2
    print(yaw_rng_1)
    print(yaw_rng)

    def normalize_yaw(y):
        y[y > np.pi] = -(2 * np.pi - y[y > np.pi])
        return y

    beam_yaws = normalize_yaw(torch.linspace(yaw_rng[0], yaw_rng[1], yaw_samples))
    p, y = torch.meshgrid(beam_pitches, beam_yaws, indexing="ij")
    yp = torch.cat((y.reshape(-1, 1), p.reshape(-1, 1)), dim=-1)
    return yp


def compute_dynamic_mask(pcl_0, flow, e1_SE3_e0):
    rigid_flow = (transform_points(e1_SE3_e0.matrix(), pcl_0) - pcl_0).detach()
    non_rigid_flow = flow - rigid_flow
    dynamic_mask = non_rigid_flow.norm(dim=-1) > 0.05
    return dynamic_mask
