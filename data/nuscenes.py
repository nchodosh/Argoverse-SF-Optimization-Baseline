import pickle
from pathlib import Path

import numpy as np
from kornia.geometry.liegroup import Se3, So3
from kornia.geometry.linalg import transform_points
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

        not_ground1 = (ex.pcl_t0[:, 2] > 0.3) & (ex.pcl_t0[:, 2] < 2.3)
        not_ground2 = (ex.pcl_t1[:, 2] > 0.3) & (ex.pcl_t1[:, 2] < 2.3)

        sensor_SE3_ego = Se3(So3.from_matrix(ex.sensor.rot.float()[None]), ex.sensor.t.float()[None]).inverse()

        pcl_1 = transform_points(sensor_SE3_ego.matrix(), ex.pcl_t0[:, :3]).detach()
        flow = transform_points(sensor_SE3_ego.matrix(), ex.pcl_t0[:, :3] + ex.flow_t0_t1).detach() - pcl_1
        pcl_2 = transform_points(sensor_SE3_ego.matrix(), ex.pcl_t1[:, :3]).detach()

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
        ego1_SE3_ego0 = Se3(R, t).inverse()
        sensor1_SE3_sensor0 = sensor_SE3_ego * ego1_SE3_ego0 * sensor_SE3_ego.inverse()

        return {
            "pcl_0": pcl_1.float(),
            "pcl_1": pcl_2.float(),
            "flow": flow.float(),
            "ego1_SE3_ego0": sensor1_SE3_sensor0,
            "annotations": annotations,
            "background_mask": ex.ego_flow_mask,
        }

    def file_to_id(self, file: Path) -> str:
        return file.stem

    def example_id_to_index(self, example_id: str) -> int:
        for i, file in enumerate(self.files):
            if file.stem == example_id:
                return i
        raise ValueError(f"{example_id} not found in the dataset")

    def example_id(self, index: int) -> str:
        return self.files[index].stem
