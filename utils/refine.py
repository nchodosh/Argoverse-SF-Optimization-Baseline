import numpy as np
from sklearn.cluster import DBSCAN


def fit_rigid(A, B):
    """
    Fit Rigid transformation A @ R.T + t = B
    """
    assert A.shape == B.shape
    num_rows, num_cols = A.shape

    # find mean column wise
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # ensure centroids are 1x3
    centroid_A = centroid_A.reshape(1, -1)
    centroid_B = centroid_B.reshape(1, -1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am.T @ Bm
    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -centroid_A @ R.T + centroid_B

    return R, t


def ransac_fit_rigid(pts1, pts2, inlier_thresh, ntrials):
    best_R = np.eye(3)
    best_t = np.zeros((3,))
    best_inliers = np.linalg.norm(pts1 - pts2, axis=-1) < inlier_thresh
    best_inliers_sum = best_inliers.sum()
    for i in range(ntrials):
        choice = np.random.choice(len(pts1), 3)
        R, t = fit_rigid(pts1[choice], pts2[choice])
        inliers = np.linalg.norm(pts1 @ R.T + t - pts2, axis=-1) < inlier_thresh
        if inliers.sum() > best_inliers_sum:
            best_R = R
            best_t = t
            best_inliers = inliers
            best_inliers_sum = best_inliers.sum()
            if best_inliers_sum / len(pts1) > 0.5:
                break
    best_R, best_t = fit_rigid(pts1[best_inliers], pts2[best_inliers])
    return best_R, best_t


def refine_flow(pc0, flow_pred, eps=0.4, min_points=10, motion_threshold=0.05, inlier_thresh=0.2, ntrials=250):
    labels = DBSCAN(eps=eps, min_samples=min_points).fit_predict(pc0)
    max_label = labels.max()
    refined_flow = np.zeros_like(flow_pred)
    for l in range(max_label + 1):
        label_mask = labels == l
        cluster_pts = pc0[label_mask]
        cluster_flows = flow_pred[label_mask]
        R, t = ransac_fit_rigid(cluster_pts, cluster_pts + cluster_flows, inlier_thresh, ntrials)
        if np.linalg.norm(t) < motion_threshold:
            R = np.eye(3)
            t = np.zeros((3,))
        refined_flow[label_mask] = (cluster_pts @ R.T + t) - cluster_pts

    refined_flow[labels == -1] = flow_pred[labels == -1]
    return refined_flow
