import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting
import numpy as np
import time
from params import par

# Set Matplotlib backend to Agg for non-interactive plotting (important for servers)
plt.switch_backend('Agg')

pose_GT_dir = par.pose_dir  # '/media/krkavinda/New Volume/Mid-Air-Dataset/MidAir_processed/'
predicted_result_dir = './result/'
gradient_color = True

def plot_route(ax, gt, out, c_gt='g', c_out='r', pred_x_idx=3, pred_y_idx=4, pred_z_idx=5, label_gt='Ground Truth', label_out='DeepVO'):
    """
    Plot the ground truth and predicted trajectories in 3D.
    
    Args:
        ax (Axes3D): The 3D axis to plot on.
        gt (np.ndarray): Ground truth poses [N, 6] (roll, pitch, yaw, x, y, z).
        out (np.ndarray): Predicted poses [N, 6] (roll, pitch, yaw, x, y, z).
        c_gt (tuple or str): Color for ground truth trajectory.
        c_out (tuple or str): Color for predicted trajectory.
        pred_x_idx (int): Index for predicted X axis (default 3: North).
        pred_y_idx (int): Index for predicted Y axis (default 4: East).
        pred_z_idx (int): Index for predicted Z axis (default 5: Down).
        label_gt (str): Label for ground truth trajectory.
        label_out (str): Label for predicted trajectory.
    """
    # Ground truth (X: North, Y: East, Z: Down)
    gt_x_idx = 3  # X (North)
    gt_y_idx = 4  # Y (East)
    gt_z_idx = 5  # Z (Down)

    # Ground truth trajectory
    x_gt = gt[:, gt_x_idx]
    y_gt = gt[:, gt_y_idx]
    z_gt = gt[:, gt_z_idx]

    # Predicted trajectory with specified axis mapping
    x_out = out[:, pred_x_idx]
    y_out = out[:, pred_y_idx]
    z_out = out[:, pred_z_idx]

    # Align predicted trajectory with ground truth by subtracting initial offset
    x_out = x_out - (x_out[0] - x_gt[0])
    y_out = y_out - (y_out[0] - y_gt[0])
    z_out = z_out - (z_out[0] - z_gt[0])

    # Debug: Print trajectory values
    print(f"Ground Truth - X: min={x_gt.min():.2f}, max={x_gt.max():.2f}, mean={x_gt.mean():.2f}")
    print(f"Ground Truth - Y: min={y_gt.min():.2f}, max={y_gt.max():.2f}, mean={y_gt.mean():.2f}")
    print(f"Ground Truth - Z: min={z_gt.min():.2f}, max={z_gt.max():.2f}, mean={z_gt.mean():.2f}")
    print(f"Predicted - X: min={x_out.min():.2f}, max={x_out.max():.2f}, mean={x_out.mean():.2f}")
    print(f"Predicted - Y: min={y_out.min():.2f}, max={y_out.max():.2f}, mean={y_out.mean():.2f}")
    print(f"Predicted - Z: min={z_out.min():.2f}, max={z_out.max():.2f}, mean={z_out.mean():.2f}")

    # Check for NaNs or infinities
    if np.any(np.isnan(x_gt)) or np.any(np.isnan(y_gt)) or np.any(np.isnan(z_gt)):
        print("Warning: NaNs detected in ground truth trajectory! Skipping segment.")
        return
    if np.any(np.isnan(x_out)) or np.any(np.isnan(y_out)) or np.any(np.isnan(z_out)):
        print("Warning: NaNs detected in predicted trajectory! Skipping segment.")
        return
    if np.any(np.isinf(x_gt)) or np.any(np.isinf(y_gt)) or np.any(np.isinf(z_gt)):
        print("Warning: Infinities detected in ground truth trajectory! Skipping segment.")
        return
    if np.any(np.isinf(x_out)) or np.any(np.isinf(y_out)) or np.any(np.isinf(z_out)):
        print("Warning: Infinities detected in predicted trajectory! Skipping segment.")
        return

    # Plot in 3D
    ax.plot(x_gt, y_gt, z_gt, color=c_gt, label=label_gt)
    ax.plot(x_out, y_out, z_out, color=c_out, label=label_out)
    ax.set_xlabel('X (North, meters)')
    ax.set_ylabel('Y (East, meters)')
    ax.set_zlabel('Z (Down, meters)')

    # Compute ranges for equal aspect ratio based on ground truth only
    x_range = np.ptp(x_gt)
    y_range = np.ptp(y_gt)
    z_range = np.ptp(z_gt)
    max_range = max(x_range, y_range, z_range) / 2

    # Set axis limits to ensure equal aspect ratio
    x_mid = np.mean(x_gt)
    y_mid = np.mean(y_gt)
    z_mid = np.mean(z_gt)
    ax.set_xlim(x_mid - max_range, x_mid + max_range)
    ax.set_ylim(y_mid - max_range, y_mid + max_range)
    ax.set_zlim(z_mid - max_range, z_mid + max_range)

    # Debug: Print axis limits
    print(f"Axis Limits - X: {x_mid - max_range:.2f} to {x_mid + max_range:.2f}")
    print(f"Axis Limits - Y: {y_mid - max_range:.2f} to {y_mid + max_range:.2f}")
    print(f"Axis Limits - Z: {z_mid - max_range:.2f} to {z_mid + max_range:.2f}")

# Define the trajectories to plot (Mid-Air dataset)
trajectories_to_plot = [
    ('Kite_training/cloudy', 'trajectory_3008'),
    ('Kite_training/foggy', 'trajectory_2008'),
    ('Kite_training/sunny', 'trajectory_0008'),
    ('Kite_training/sunset', 'trajectory_1008')
]

for climate, traj_id in trajectories_to_plot:
    print('='*50)
    print(f'Trajectory: {climate}/{traj_id}')

    # Load ground truth poses
    traj_num = traj_id.replace('trajectory_', '')
    GT_pose_path = f'{pose_GT_dir}/{climate}/poses/poses_{traj_num}.npy'
    gt = np.load(GT_pose_path)

    # Load predicted poses
    climate_sanitized = climate.replace('/', '_')
    pose_result_path = f'{predicted_result_dir}/out_{climate_sanitized}_{traj_id}.txt'
    with open(pose_result_path) as f_out:
        out = [l.strip() for l in f_out.readlines()]
        for i, line in enumerate(out):
            out[i] = [float(v) for v in line.split(',')]
        out = np.array(out)

    # Ensure gt and out have the same length
    min_len = min(len(gt), len(out))
    gt = gt[:min_len]
    out = out[:min_len]

    # Debug: Print shapes
    print(f"Ground Truth Shape: {gt.shape}")
    print(f"Predicted Shape: {out.shape}")

    # Compute MSE
    mse_rotate = 10 * np.mean((out[:, :3] - gt[:, :3])**2)
    mse_translate = np.mean((out[:, 3:] - gt[:, 3:6])**2)
    print('mse_rotate: ', mse_rotate)
    print('mse_translate: ', mse_translate)

    if gradient_color:
        # Plot gradient color in 3D with Z (Down)_Y (East)_X (North) mapping for predicted poses
        step = max(1, len(out) // 10)  # Adaptive step size: ~10 segments
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')  # Create a 3D axis
        plt.scatter([gt[0][3]], [gt[0][4]], [gt[0][5]], label='Sequence Start', marker='s', color='k')
        
        for st in range(0, len(out), step):
            end = min(st + step, len(out))
            print(f"Plotting segment: {st} to {end}")
            g = max(0.2, st / len(out))
            c_gt = (0, g, 0)  # Green gradient for ground truth
            c_out = (1, g, 0)  # Red-to-yellow gradient for predicted
            # Predicted: Z (Down) as X (North), Y (East) as Y (East), X (North) as Z (Down)
            label_gt = 'Ground Truth' if st == 0 else None
            label_out = 'DeepVO' if st == 0 else None
            plot_route(ax, gt[st:end], out[st:end], c_gt, c_out, pred_x_idx=5, pred_y_idx=4, pred_z_idx=3, label_gt=label_gt, label_out=label_out)

        plt.legend()
        plt.title(f'Trajectory: {climate}/{traj_id}\n(Predicted in NED: X=North, Y=East, Z=Down)')
        climate_sanitized = climate.replace('/', '_')
        save_name = f'{predicted_result_dir}/route_3d_with_gt_{climate_sanitized}_{traj_id}_gradient_NED.png'
        plt.savefig(save_name)
        print(f"Saved plot to: {save_name}")
        plt.close(fig)