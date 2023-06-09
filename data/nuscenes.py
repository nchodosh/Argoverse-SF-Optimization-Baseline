import pickle
from pathlib import Path

import numpy as np
from kornia.geometry.liegroup import Se3, So3
from torch.utils.data import Dataset

from utils import dotdict, geometry
from utils.torch import numpy_to_torch


class Dataloader(Dataset):
    def __init__(self, data_root="inputs"):
        self.files = sorted(list((Path(data_root) / "nuscenes").rglob("*.pkl")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with open(self.files[index], "rb") as f:
            ex = dotdict.dict_to_namespace(numpy_to_torch(pickle.load(f)))

        not_ground1 = ex.pcl_t0[:, 2] > 0.3
        not_ground2 = ex.pcl_t1[:, 2] > 0.3

        pcl_1 = geometry.ego_to_sensor(ex.pcl_t0[:, :3], ex.sensor)
        flow = geometry.ego_to_sensor(ex.pcl_t0[:, :3] + ex.flow_t0_t1, ex.sensor) - geometry.ego_to_sensor(
            ex.pcl_t0[:, :3], ex.sensor
        )
        pcl_2 = geometry.ego_to_sensor(ex.pcl_t1[:, :3], ex.sensor)

        _, m1 = geometry.filter_range(pcl_1, return_mask=True)
        m1 = m1 & not_ground1
        pcl_1 = pcl_1[m1]
        flow = flow[m1]

        _, m2 = geometry.filter_range(pcl_2, return_mask=True)
        m2 = m2 & not_ground2
        pcl_2 = pcl_2[m2]

        annotations = ex.annotation_labels[m1]

        R = So3.from_matrix(ex.odom_t0_t1[:3, :3].clone()[None].float())
        t = ex.odom_t0_t1[:3, 3][None].float()
        ego1_SE3_ego0 = Se3(R, t)

        return {
            "pcl_0": pcl_1.float(),
            "pcl_1": pcl_2.float(),
            "flow": flow.float(),
            "ego1_SE3_ego0": ego1_SE3_ego0,
            "annotations": annotations,
        }

    def example_id(self, index):
        return self.files[index].stem
