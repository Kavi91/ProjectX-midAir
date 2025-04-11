import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from params import par
from helper import to_ned_pose
import torch

# Directories
pose_GT_dir = f'{par.data_dir}'
predicted_result_dir = 'results/test_predictions/'
gradient_color = True
plot_3d = True

def plot_route_3d(gt, out, ax, c_gt='b', c_out='r', label_gt='Ground Truth', label_out='Predicted'):
    x_gt = gt[:, 3]  # North
    y_gt = gt[:, 4]  # East
    z_gt = gt[:, 5]  # Down
    x_out = out[:, 3]
    y_out = out[:, 4]
    z_out = out[:, 5]
    
    ax.plot(x_gt, y_gt, z_gt, color=c_gt, label=label_gt)
    ax.plot(x_out, y_out, z_out, color=c_out, label=label_out)
    
    ax.set_xlabel('North (m)')
    ax.set_ylabel('East (m)')
    ax.set_zlabel('Down (m)')

# Test trajectories from params.py
test_videos = []
for climate_set, traj_ids in par.test_traj_ids.items():
    test_videos.extend([(climate_set, traj) for traj in traj_ids])
video_list = [traj_id for _, traj_id in test_videos]

for climate_set, traj_id in test_videos:
    print('=' * 50)
    print(f'Trajectory {traj_id} in {climate_set}')
    
    # Load ground-truth poses
    traj_num = traj_id.split('_')[1]
    GT_pose_path = f'{pose_GT_dir}/{climate_set}/poses/poses_{traj_num}.npy'
    if not os.path.exists(GT_pose_path):
        print(f'Ground-truth poses not found at {GT_pose_path}. Skipping.')
        continue
    gt = np.load(GT_pose_path)
    gt = torch.tensor(gt, dtype=torch.float32)
    gt = to_ned_pose(gt, is_absolute=True).numpy()
    
    # Debug: Print raw ground truth before scaling
    print("Raw ground truth (first 5 poses, NED, before scaling):")
    print(gt[:5])
    
    # Scale ground truth translations to match GPS plot scale
    gt[:, 3:] /= 30  # Divide by 30 to match the GPS plot's scale
    
    # Load predicted poses
    pose_result_path = f'{predicted_result_dir}pred_{traj_id}.txt'
    if not os.path.exists(pose_result_path):
        print(f'Predicted poses not found at {pose_result_path}. Skipping.')
        continue
    with open(pose_result_path, 'r') as f_out:
        out = [line.strip().split(',') for line in f_out.readlines()]
        out = np.array([[float(v) for v in line] for line in out])
    
    # Ensure lengths match
    min_len = min(len(gt), len(out))
    gt = gt[:min_len]
    out = out[:min_len]
    print(f'Ground truth poses: {len(gt)}, Predicted poses: {len(out)}')
    
    # Print trajectory ranges
    print("Ground truth trajectory range (NED, after scaling):")
    print(f"North: {gt[:, 3].min()} to {gt[:, 3].max()}")
    print(f"East: {gt[:, 4].min()} to {gt[:, 4].max()}")
    print(f"Down: {gt[:, 5].min()} to {gt[:, 5].max()}")
    print("Predicted trajectory range (NED, after scaling):")
    print(f"North: {out[:, 3].min()} to {out[:, 3].max()}")
    print(f"East: {out[:, 4].min()} to {out[:, 4].max()}")
    print(f"Down: {out[:, 5].min()} to {out[:, 5].max()}")
    
    # Compute MSE
    mse_rotate = 100 * np.mean((out[:, :3] - gt[:, :3]) ** 2)
    mse_translate = np.mean((out[:, 3:] - gt[:, 3:]) ** 2)
    print(f'MSE Rotation (x100): {mse_rotate:.6f}')
    print(f'MSE Translation: {mse_translate:.6f}')
    
    # Plotting
    plt.clf()
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectories
    plot_route_3d(gt, out, ax, c_gt='b', c_out='r', label_gt='Ground Truth', label_out='Predicted')
    
    # Mark start point
    ax.scatter([gt[0, 3]], [gt[0, 4]], [gt[0, 5]], color='k', marker='s', label='Sequence Start')
    
    plt.title(f'Trajectory {traj_id} - 3D Path (NED)')
    plt.legend()
    save_name = f'{predicted_result_dir}route_{traj_id}_3d.png'
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f'Saved plot to {save_name}')

print('Visualization completed.')