import numpy as np
import torch
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import pandas as pd
from params import par

# Set Matplotlib backend to Agg for non-interactive plotting
plt.switch_backend('Agg')

# Define paths
pose_dir = par.pose_dir  # '/media/krkavinda/New Volume/Mid-Air-Dataset/MidAir_processed/'
predicted_result_dir = '/home/krkavinda/ProjectX-midAir/result/'  # Absolute path
trajectories_to_verify = [
    ('Kite_training/cloudy', 'trajectory_3008'),
    ('Kite_training/sunny', 'trajectory_0008'),
    ('Kite_training/foggy', 'trajectory_2008'),
    ('Kite_training/sunset', 'trajectory_1008')
]

# Utility functions
def euler_to_rotation_matrix(roll, pitch, yaw):
    """Convert Euler angles (roll, pitch, yaw) to a rotation matrix (XYZ order)."""
    return Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=False).as_matrix()

def rotation_matrix_to_euler(R):
    """Convert a rotation matrix to Euler angles (roll, pitch, yaw)."""
    return Rotation.from_matrix(R).as_euler('xyz', degrees=False)

def compute_relative_pose(pose_t, pose_t_minus_1):
    """Compute the relative pose between two consecutive absolute poses in NED."""
    roll_t, pitch_t, yaw_t = pose_t[:3]
    x_t, y_t, z_t = pose_t[3:]
    roll_tm1, pitch_tm1, yaw_tm1 = pose_t_minus_1[:3]
    x_tm1, y_tm1, z_tm1 = pose_t_minus_1[3:]

    roll_rel = roll_t - roll_tm1
    pitch_rel = pitch_t - pitch_tm1
    yaw_rel = yaw_t - yaw_tm1

    roll_rel = np.arctan2(np.sin(roll_rel), np.cos(roll_rel))
    pitch_rel = np.arctan2(np.sin(pitch_rel), np.cos(pitch_rel))
    yaw_rel = np.arctan2(np.sin(yaw_rel), np.cos(yaw_rel))

    t_rel = np.array([x_t - x_tm1, y_t - y_tm1, z_t - z_tm1])
    R_prev = euler_to_rotation_matrix(roll_tm1, pitch_tm1, yaw_tm1)
    t_rel_transformed = R_prev.T @ t_rel  # Transform to previous frame's Body frame

    return np.array([roll_rel, pitch_rel, yaw_rel, t_rel_transformed[0], t_rel_transformed[1], t_rel_transformed[2]])

def compute_absolute_poses(relative_poses):
    """Compute absolute poses from relative poses in NED."""
    seq_len = len(relative_poses)
    absolute_poses = np.zeros((seq_len + 1, 6))  # [roll, pitch, yaw, x, y, z]

    for t in range(seq_len):
        rel_pose = relative_poses[t]
        rel_angles = rel_pose[:3]  # [roll, pitch, yaw]
        rel_trans = rel_pose[3:]  # [x, y, z]

        prev_pose = absolute_poses[t]
        prev_angles = prev_pose[:3]
        prev_trans = prev_pose[3:]

        # Compute rotation matrix of previous pose
        R = euler_to_rotation_matrix(prev_angles[0], prev_angles[1], prev_angles[2])

        # Transform relative translation to World frame
        abs_trans = prev_trans + R @ rel_trans

        # Update absolute angles
        abs_angles = prev_angles + rel_angles

        absolute_poses[t + 1, :3] = abs_angles
        absolute_poses[t + 1, 3:] = abs_trans

    return absolute_poses

def plot_trajectories_2d(pred_traj, gt_traj, climate, traj_id, save_dir, suffix=""):
    """Plot 2D trajectories in the North-East plane."""
    pred_x, pred_y = pred_traj[:, 3], pred_traj[:, 4]  # x (North), y (East)
    gt_x, gt_y = gt_traj[:, 3], gt_traj[:, 4]  # x (North), y (East)

    plt.figure(figsize=(10, 8))
    plt.plot(pred_x, pred_y, label='Predicted Trajectory', color='blue', linestyle='--')
    plt.plot(gt_x, gt_y, label='Ground Truth Trajectory', color='green', linestyle='-')
    plt.xlabel('North (m)')
    plt.ylabel('East (m)')
    plt.title(f'Trajectory Comparison: {climate}/{traj_id}')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    climate_sanitized = climate.replace('/', '_')
    plt.savefig(f'{save_dir}/traj_2d_{climate_sanitized}_{traj_id}{suffix}.png')
    plt.close()

def plot_trajectories_3d(pred_traj, gt_traj, climate, traj_id, save_dir, suffix=""):
    """Plot 3D trajectories with NED convention."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Ground truth (X: North, Y: East, Z: Down)
    x_gt = gt_traj[:, 3]
    y_gt = gt_traj[:, 4]
    z_gt = gt_traj[:, 5]

    # Predicted (X: North, Y: East, Z: Down)
    x_out = pred_traj[:, 3]
    y_out = pred_traj[:, 4]
    z_out = pred_traj[:, 5]

    # Align predicted trajectory with ground truth
    x_out = x_out - (x_out[0] - x_gt[0])
    y_out = y_out - (y_out[0] - y_gt[0])
    z_out = z_out - (z_out[0] - z_gt[0])

    ax.plot(x_gt, y_gt, z_gt, color='green', label='Ground Truth')
    ax.plot(x_out, y_out, z_out, color='blue', linestyle='--', label='Predicted')
    ax.set_xlabel('X (North, meters)')
    ax.set_ylabel('Y (East, meters)')
    ax.set_zlabel('Z (Down, meters)')

    # Set equal aspect ratio
    x_range = np.ptp(x_gt)
    y_range = np.ptp(y_gt)
    z_range = np.ptp(z_gt)
    max_range = max(x_range, y_range, z_range) / 2
    x_mid = np.mean(x_gt)
    y_mid = np.mean(y_gt)
    z_mid = np.mean(z_gt)
    ax.set_xlim(x_mid - max_range, x_mid + max_range)
    ax.set_ylim(y_mid - max_range, y_mid + max_range)
    ax.set_zlim(z_mid - max_range, z_mid + max_range)

    plt.legend()
    plt.title(f'Trajectory Comparison (3D): {climate}/{traj_id}')
    climate_sanitized = climate.replace('/', '_')
    plt.savefig(f'{save_dir}/traj_3d_{climate_sanitized}_{traj_id}{suffix}.png')
    plt.close()

# Verification script
for climate, traj_id in trajectories_to_verify:
    print('=' * 50)
    print(f'Verifying: {climate}/{traj_id}')

    # Step 1: Load ground truth poses
    traj_num = traj_id.replace('trajectory_', '')
    gt_pose_path = f'{pose_dir}/{climate}/poses/poses_{traj_num}.npy'
    gt_pose = np.load(gt_pose_path)  # Shape: [n_images, 6] (roll, pitch, yaw, x, y, z)

    print("Ground Truth Poses (NED) - First 5:")
    print(gt_pose[:5])

    # Step 2: Compute relative poses from ground truth
    gt_relative_poses = []
    for i in range(len(gt_pose)):
        if i == 0:
            gt_relative_poses.append(np.zeros(6))
        else:
            rel_pose = compute_relative_pose(gt_pose[i], gt_pose[i-1])
            gt_relative_poses.append(rel_pose)
    gt_relative_poses = np.array(gt_relative_poses)

    print("Ground Truth Relative Poses (NED) - First 5:")
    print(gt_relative_poses[:5])

    # Step 3: Load IMU and GPS data for cross-checking
    imu_path = f'/media/krkavinda/New Volume/Mid-Air-Dataset/MidAir_processed/{climate}/{traj_id}/imu.npy'
    gps_path = f'/media/krkavinda/New Volume/Mid-Air-Dataset/MidAir_processed/{climate}/{traj_id}/gps.npy'

    if os.path.exists(imu_path):
        imu_data = np.load(imu_path)  # Shape: [n_images, 6] (ax, ay, az, wx, wy, wz)
        print("IMU Data (Body Frame, NED) - First 5:")
        print(imu_data[:5])

        # Verify relative rotations using IMU angular velocities
        imu_angular_vel = imu_data[:, 3:]  # [wx, wy, wz]
        dt = 1.0 / 5.0  # Mid-Air frame rate: 5 Hz
        imu_relative_angles = imu_angular_vel * dt  # Approximate angle change
        print("IMU Relative Angles (Body Frame, NED) - First 5:")
        print(imu_relative_angles[:5])

        # Compare with ground truth relative angles
        gt_relative_angles = gt_relative_poses[:, :3]
        print("Difference (GT - IMU) Relative Angles - Mean:")
        print(np.mean(np.abs(gt_relative_angles[1:] - imu_relative_angles[:-1]), axis=0))

    if os.path.exists(gps_path):
        gps_data = np.load(gps_path)  # Shape: [n_images, 6] (x, y, z, vx, vy, vz)
        print("GPS Data (World Frame, NED) - First 5:")
        print(gps_data[:5])

        # Compute relative positions from GPS
        gps_pos = gps_data[:, :3]  # [x, y, z]
        gps_relative_pos = gps_pos[1:] - gps_pos[:-1]
        gps_relative_pos = np.vstack([np.zeros(3), gps_relative_pos])  # Pad first frame
        print("GPS Relative Positions (World Frame, NED) - First 5:")
        print(gps_relative_pos[:5])

        # Compare with ground truth relative translations
        gt_relative_trans = gt_relative_poses[:, 3:]
        print("Difference (GT - GPS) Relative Translations - Mean:")
        print(np.mean(np.abs(gt_relative_trans[1:] - gps_relative_pos[:-1]), axis=0))

    # Step 4: Load model predictions
    climate_sanitized = climate.replace('/', '_')
    pose_result_path = f'{predicted_result_dir}/out_{climate_sanitized}_{traj_id}.txt'
    if not os.path.exists(pose_result_path):
        print(f"Predicted pose file not found: {pose_result_path}")
        continue
    with open(pose_result_path) as f_out:
        out = [l.strip() for l in f_out.readlines()]
        for i, line in enumerate(out):
            out[i] = [float(v) for v in line.split(',')]
        out = np.array(out)

    # Ensure lengths match
    min_len = min(len(gt_pose), len(out))
    gt_pose = gt_pose[:min_len]
    out = out[:min_len]

    print("Predicted Absolute Poses (NED, Before X Flip) - First 5:")
    print(out[:5])

    # Step 5: Apply X (North) flip correction to predicted poses
    out_flipped = out.copy()
    out_flipped[:, 3] *= -1  # Flip the X (North) direction

    print("Predicted Absolute Poses (NED, After X Flip) - First 5:")
    print(out_flipped[:5])

    # Step 6: Compute relative poses from predicted absolute poses (after X flip)
    pred_relative_poses = []
    for i in range(len(out_flipped)):
        if i == 0:
            pred_relative_poses.append(np.zeros(6))
        else:
            rel_pose = compute_relative_pose(out_flipped[i], out_flipped[i-1])
            pred_relative_poses.append(rel_pose)
    pred_relative_poses = np.array(pred_relative_poses)

    print("Predicted Relative Poses (NED, After X Flip) - First 5:")
    print(pred_relative_poses[:5])

    # Compare predicted relative poses with ground truth
    print("Difference (GT - Predicted) Relative Poses (After X Flip) - Mean:")
    print(np.mean(np.abs(gt_relative_poses - pred_relative_poses), axis=0))

    # Step 7: Recompute absolute poses from predicted relative poses
    pred_absolute_poses = compute_absolute_poses(pred_relative_poses)

    print("Recomputed Predicted Absolute Poses (NED, After X Flip) - First 5:")
    print(pred_absolute_poses[:5])

    # Step 8: Plot trajectories (before and after X flip for comparison)
    # Before X flip
    plot_trajectories_2d(out, gt_pose, climate, traj_id, predicted_result_dir, suffix="_before_xflip")
    plot_trajectories_3d(out, gt_pose, climate, traj_id, predicted_result_dir, suffix="_before_xflip")
    
    # After X flip
    plot_trajectories_2d(pred_absolute_poses, gt_pose, climate, traj_id, predicted_result_dir, suffix="_after_xflip")
    plot_trajectories_3d(pred_absolute_poses, gt_pose, climate, traj_id, predicted_result_dir, suffix="_after_xflip")

    # Step 9: Analyze dataset diversity
    gt_trans = gt_pose[:, 3:6]  # [x, y, z]
    gt_angles = gt_pose[:, :3]  # [roll, pitch, yaw]
    trans_range = np.ptp(gt_trans, axis=0)
    angle_range = np.ptp(gt_angles, axis=0)
    print(f"Dataset Diversity - {climate}/{traj_id}:")
    print(f"Translation Range (NED): North={trans_range[0]:.2f}, East={trans_range[1]:.2f}, Down={trans_range[2]:.2f} meters")
    print(f"Angle Range (NED): Roll={angle_range[0]:.2f}, Pitch={angle_range[1]:.2f}, Yaw={angle_range[2]:.2f} radians")