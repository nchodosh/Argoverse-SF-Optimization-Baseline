import torch
from kornia.geometry.linalg import transform_points


def motion_compensate(datum):
    e1_SE3_e0 = datum["ego1_SE3_ego0"]
    pcl_0 = datum["pcl_0"]
    flow = datum["flow"]

    pcl_input = transform_points(e1_SE3_e0.matrix(), pcl_0).detach()
    rigid_flow = (transform_points(e1_SE3_e0.matrix(), pcl_0) - pcl_0).detach()
    flow = flow - rigid_flow

    return pcl_input, flow


def motion_compensate_prediction(datum, flow_pred):
    e1_SE3_e0 = datum["ego1_SE3_ego0"]
    pcl_0 = datum["pcl_0"]

    rigid_flow = (transform_points(e1_SE3_e0.matrix(), pcl_0) - pcl_0).detach()
    flow = flow_pred - rigid_flow

    return flow
