import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting
import numpy as np
import time
from params import par

pose_GT_dir = par.pose_dir  # '/media/krkavinda/New Volume/Mid-Air-Dataset/MidAir_processed/'
predicted_result_dir = './result/'
gradient_color = True

def plot_route(gt, out, c_gt='g', c_out='r', pred_x_idx=3, pred_y_idx=4, pred_z_idx=5):
    # Ground truth (X: North, Y: East, Z: Down)
    gt_x_idx = 3  # X (North)
    gt_y_idx = 4  # Y (East)
    gt_z_idx = 5  # Z (Down)

    # Ground truth trajectory
    x_gt = [v for v in gt[:, gt_x_idx]]
    y_gt = [v for v in gt[:, gt_y_idx]]
    z_gt = [v for v in gt[:, gt_z_idx]]

    # Predicted trajectory with specified axis mapping
    x_out = [v for v in out[:, pred_x_idx]]
    y_out = [v for v in out[:, pred_y_idx]]
    z_out = [v for v in out[:, pred_z_idx]]

    # Plot in 3D
    ax = plt.gca()
    ax.plot(x_gt, y_gt, z_gt, color=c_gt, label='Ground Truth')
    ax.plot(x_out, y_out, z_out, color=c_out, label='DeepVO')
    ax.set_xlabel('X (North, meters)')
    ax.set_ylabel('Y (East, meters)')
    ax.set_zlabel('Z (Down, meters)')
    # Set equal aspect ratio for 3D plot (approximation)
    ax.set_box_aspect([1, 1, 1])

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
        mse_rotate = 100 * np.mean((out[:, :3] - gt[:, :3])**2)
        mse_translate = np.mean((out[:, 3:] - gt[:, 3:6])**2)
        print('mse_rotate: ', mse_rotate)
        print('mse_translate: ', mse_translate)

    if gradient_color:
        # Plot gradient color in 3D with Z (Down)_Y (East)_X (North) mapping for predicted poses
        step = 200
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')  # Create a 3D axis
        plt.scatter([gt[0][3]], [gt[0][4]], [gt[0][5]], label='sequence start', marker='s', color='k')
        for st in range(0, len(out), step):
            end = st + step
            g = max(0.2, st/len(out))
            c_gt = (0, g, 0)
            c_out = (1, g, 0)
            # Predicted: Z (Down) as X (North), Y (East) as Y (East), X (North) as Z (Down)
            plot_route(gt[st:end], out[st:end], c_gt, c_out, pred_x_idx=5, pred_y_idx=4, pred_z_idx=3)
            if st == 0:
                plt.legend()
            plt.title(f'Trajectory: {climate}/{traj_id} (Predicted: Z(Down)_Y(East)_X(North))')
            climate_sanitized = climate.replace('/', '_')
            save_name = f'{predicted_result_dir}/route_3d_with_gt_{climate_sanitized}_{traj_id}_gradient_Z(Down)_Y(East)_X(North).png'
        plt.savefig(save_name)
        plt.close(fig)