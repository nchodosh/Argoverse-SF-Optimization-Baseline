from pathlib import Path

import torch
from av2.evaluation.scene_flow.utils import (
    get_eval_point_mask,
    get_eval_subset,
    write_output_file,
)
from av2.torch.data_loaders.scene_flow import SceneFlowDataloader
from torch.utils.data import Dataset


class Dataloader(Dataset):
    def __init__(self, data_root="inputs", split="val", mask_file="val-mask.zip"):
        self.data_loader = SceneFlowDataloader(data_root, "av2", split)
        self.inds = get_eval_subset(self.data_loader)
        self.mask_file = Path(mask_file)

    def __len__(self):
        return len(self.inds)

    def __getitem__(self, index):
        s0, s1, ego1_SE3_ego0, flow_obj = self.data_loader[self.inds[index]]
        pcl_0 = s0.lidar.as_tensor()[:, :3]
        pcl_1 = s1.lidar.as_tensor()[:, :3]
        flow = flow_obj.flow if flow_obj is not None else None
        mask0 = get_eval_point_mask(s0.sweep_uuid, self.mask_file)
        mask1 = torch.logical_and(
            torch.logical_and((pcl_1[:, 0].abs() <= 50), (pcl_1[:, 1].abs() <= 50)).bool(),
            torch.logical_not(s1.is_ground),
        )

        pcl_1 = pcl_1[mask1]
        pcl_0 = pcl_0[mask0]
        if flow is not None:
            flow = flow[mask0]

        return {"pcl_0": pcl_0, "pcl_1": pcl_1, "flow": flow, "ego1_SE3_ego0": ego1_SE3_ego0}

    def example_id(self, index):
        index = self.inds[index]
        log = str(self.data_loader.file_index.loc[index, "log_id"])
        ts = self.data_loader.file_index.loc[index, "timestamp_ns"]
        return f"{log}-{ts}"
