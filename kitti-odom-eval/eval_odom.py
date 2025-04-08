import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# Hardcoded file paths
PRED_FILE = '/home/krkavinda/ProjectX-midAir/result/out_Kite_training_sunny_trajectory_0008.txt'
GT_FILE   = '/media/krkavinda/New Volume/Mid-Air-Dataset/MidAir_processed/Kite_training/sunny/poses/poses_0008.npy'
OUT_TRAJ_PLOT = '/home/krkavinda/ProjectX-midAir/comparison_plots/trajectory_comparison.png'
OUT_ERR_PLOT  = '/home/krkavinda/ProjectX-midAir/comparison_plots/per_frame_error.png'

def vector6_to_pose(vec):
    """
    Convert a 6-element vector [roll, pitch, yaw, t_x, t_y, t_z] to a 4x4 homogeneous transformation.
    Euler angles are assumed to be in radians with 'xyz' order.
    """
    angles = vec[:3]
    t = vec[3:]
    R = Rotation.from_euler('xyz', angles, degrees=False).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def load_predicted_poses_txt(pred_file):
    poses = []
    with open(pred_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                vals = [float(x) for x in line.split(',') if x.strip() != ""]
                if len(vals) != 6:
                    raise ValueError("Each line must have 6 values; got {}".format(len(vals)))
                poses.append(vals)
    return np.array(poses)

def load_ground_truth_poses_npy(gt_file):
    return np.load(gt_file)

def build_pose_dict(pose_array):
    pose_dict = {}
    N = pose_array.shape[0]
    for i in range(N):
        pose_dict[i] = vector6_to_pose(pose_array[i])
    return pose_dict

def umeyama_alignment(source, target):
    """Computes the similarity transform (scale, R, t) that maps source to target."""
    assert source.shape == target.shape, "Source and target must have the same shape"
    n, dim = source.shape
    mu_source = np.mean(source, axis=0)
    mu_target = np.mean(target, axis=0)
    source_centered = source - mu_source
    target_centered = target - mu_target
    H = source_centered.T @ target_centered / n
    U, D, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    var_source = np.sum(source_centered ** 2) / n
    s = np.sum(D) / var_source
    t = mu_target - s * (R @ mu_source)
    return s, R, t

def align_predictions(pred_poses, gt_poses):
    pred_xyz = pred_poses[:, 3:6]
    gt_xyz = gt_poses[:, 3:6]
    s, R, t = umeyama_alignment(pred_xyz, gt_xyz)
    pred_xyz_aligned = (s * (R @ pred_xyz.T)).T + t
    pred_poses_aligned = np.copy(pred_poses)
    pred_poses_aligned[:, 3:6] = pred_xyz_aligned
    return pred_poses_aligned, s, R, t

def compute_ATE(gt_dict, pred_dict):
    common_idxs = sorted(set(gt_dict.keys()) & set(pred_dict.keys()))
    errors = []
    for i in common_idxs:
        gt_t = gt_dict[i][:3, 3]
        pred_t = pred_dict[i][:3, 3]
        errors.append(np.linalg.norm(gt_t - pred_t))
    errors = np.array(errors)
    rmse = np.sqrt(np.mean(errors**2))
    return rmse, errors

def compute_RPE(gt_dict, pred_dict):
    common_idxs = sorted(set(gt_dict.keys()) & set(pred_dict.keys()))
    trans_errors = []
    rot_errors = []
    for i in common_idxs[:-1]:
        T_gt1 = gt_dict[i]
        T_gt2 = gt_dict[i+1]
        T_pred1 = pred_dict[i]
        T_pred2 = pred_dict[i+1]
        rel_gt = np.linalg.inv(T_gt1) @ T_gt2
        rel_pred = np.linalg.inv(T_pred1) @ T_pred2
        t_gt = rel_gt[:3, 3]
        t_pred = rel_pred[:3, 3]
        trans_errors.append(np.linalg.norm(t_gt - t_pred))
        R_err = np.linalg.inv(rel_pred[:3, :3]) @ rel_gt[:3, :3]
        angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0))
        rot_errors.append(angle)
    return np.mean(trans_errors), np.mean(rot_errors)

def plot_trajectories(gt_dict, pred_dict, out_file):
    gt_xyz = []
    pred_xyz = []
    common_idxs = sorted(set(gt_dict.keys()) & set(pred_dict.keys()))
    for i in common_idxs:
        gt_xyz.append(gt_dict[i][:3, 3])
        pred_xyz.append(pred_dict[i][:3, 3])
    gt_xyz = np.array(gt_xyz)
    pred_xyz = np.array(pred_xyz)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(gt_xyz[:,0], gt_xyz[:,1], gt_xyz[:,2], label="Ground Truth", color='green')
    ax.plot(pred_xyz[:,0], pred_xyz[:,1], pred_xyz[:,2], label="Aligned Prediction", color='red')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()
    plt.title("Trajectory Comparison (6DoF)")
    
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    plt.savefig(out_file)
    plt.close(fig)
    print("Saved trajectory plot to:", out_file)

def plot_per_frame_error(errors, out_file):
    """
    Plot per-frame translation error versus frame index.
    """
    plt.figure()
    plt.plot(errors, 'b-', label='Translation Error per Frame')
    plt.xlabel("Frame Index")
    plt.ylabel("Translation Error (m)")
    plt.title("Per-Frame Translation Error")
    plt.legend()
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    plt.savefig(out_file)
    plt.close()
    print("Saved per-frame error plot to:", out_file)

def main():
    print("Loading predicted data from:", PRED_FILE)
    pred_array = load_predicted_poses_txt(PRED_FILE)
    print("Predicted data shape:", pred_array.shape)
    
    print("Loading ground truth data from:", GT_FILE)
    gt_array = load_ground_truth_poses_npy(GT_FILE)
    print("Ground truth data shape:", gt_array.shape)
    
    N = min(pred_array.shape[0], gt_array.shape[0])
    pred_array = pred_array[:N]
    gt_array = gt_array[:N]
    
    print("First 5 predicted poses:")
    print(pred_array[:5])
    print("First 5 ground truth poses:")
    print(gt_array[:5])
    
    print("\nData ranges:")
    print(f"GT X: {gt_array[:,3].min()} to {gt_array[:,3].max()}")
    print(f"GT Y: {gt_array[:,4].min()} to {gt_array[:,4].max()}")
    print(f"GT Z: {gt_array[:,5].min()} to {gt_array[:,5].max()}")
    print(f"Pred X: {pred_array[:,3].min()} to {pred_array[:,3].max()}")
    print(f"Pred Y: {pred_array[:,4].min()} to {pred_array[:,4].max()}")
    print(f"Pred Z: {pred_array[:,5].min()} to {pred_array[:,5].max()}")
    
    # Align the predicted poses to GT.
    pred_aligned, s, R, t = align_predictions(pred_array, gt_array)
    print("\nAlignment parameters:")
    print("Scale =", s)
    print("Rotation matrix:\n", R)
    print("Translation vector:", t)
    
    # Build dictionaries.
    pred_dict = build_pose_dict(pred_aligned)
    gt_dict = build_pose_dict(gt_array)
    
    # Compute overall errors.
    ate, ate_errors = compute_ATE(gt_dict, pred_dict)
    rpe_trans, rpe_rot = compute_RPE(gt_dict, pred_dict)
    
    print("\nEvaluation Metrics:")
    print("ATE (RMSE): {:.4f} m".format(ate))
    print("RPE: translation = {:.4f} m, rotation = {:.4f} rad ({:.2f} deg)".format(
        rpe_trans, rpe_rot, np.degrees(rpe_rot)))
    
    # Plot and save trajectories.
    plot_trajectories(gt_dict, pred_dict, OUT_TRAJ_PLOT)
    
    # Additionally, plot per-frame translation error to see if there is drift.
    plt.figure()
    plt.plot(ate_errors, 'ro-')
    plt.xlabel("Frame Index")
    plt.ylabel("Translation Error (m)")
    plt.title("Per-Frame Translation Error")
    os.makedirs(os.path.dirname(OUT_ERR_PLOT), exist_ok=True)
    plt.savefig(OUT_ERR_PLOT)
    plt.close()
    print("Saved per-frame error plot to:", OUT_ERR_PLOT)

if __name__ == "__main__":
    main()
