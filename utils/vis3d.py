import seaborn as sns
import torch
from mayavi import mlab

import data.utils

colors = sns.color_palette()


def gradients(model, datum):
    fig = mlab.figure(bgcolor=(0.0, 0.0, 0.0))
    pts, grad = model.point_gradients(datum["pcl_0"], datum["pcl_1"], datum["ego1_SE3_ego0"])
    grad_norm = grad.norm(dim=-1)
    mlab.quiver3d(*pts.T, *grad.T, scalars=grad_norm, scale_mode="scalar", scale_factor=1000, figure=fig)
    mlab.points3d(*datum["pcl_1"].T, color=colors[0], scale_factor=0.05, figure=fig)
    mlab.points3d(*pts.T, color=colors[1], scale_factor=0.05, figure=fig)


def flow_pred(model, datum):
    fig = mlab.figure(bgcolor=(0.0, 0.0, 0.0))
    emax = 0.3

    pred, dynamic = model(datum["pcl_0"], datum["ego1_SE3_ego0"])
    pcl_0, gt = data.utils.motion_compensate(datum)
    pred = data.utils.motion_compensate_prediction(datum, pred)
    error = torch.clip(torch.norm(gt - pred, dim=-1), 0, emax)

    mlab.points3d(*datum["pcl_1"].T, color=colors[0], scale_factor=0.05, figure=fig)
    mlab.points3d(*pcl_0.T, error, scale_factor=0.05, figure=fig, scale_mode="none")

    big_flows = torch.norm(pred, dim=-1) > 0.05

    plot_inds = torch.where(big_flows)[0]
    nbig_flows = len(plot_inds)
    print(nbig_flows)
    pts = mlab.pipeline.scalar_scatter(
        *torch.cat([pcl_0[plot_inds], pcl_0[plot_inds] + pred[plot_inds]]).T,
        torch.cat([error[plot_inds], error[plot_inds]]),
    )
    connections = torch.vstack([torch.tensor([i, i + nbig_flows]) for i in range(nbig_flows)])
    pts.mlab_source.dataset.lines = connections.numpy()
    tube = mlab.pipeline.tube(pts, tube_radius=0.01)
    tube_surf = mlab.pipeline.surface(tube, vmin=0, vmax=emax, opacity=1.0)
    tube.children[0].scalar_lut_manager.lut_mode = "bwr"
    return fig, pts, tube, tube_surf


visualizations = {"flow": flow_pred, "grad": gradients}
