from params import par
from model_2 import StereoAdaptiveVO
import numpy as np
from PIL import Image
import glob
import os
import time
import torch
from data_helper import get_data_info, ImageSequenceDataset
from torch.utils.data import DataLoader
from helper import eulerAnglesToRotationMatrix
import matplotlib.pyplot as plt

def compute_mse(pred_traj, gt_traj):
    """
    Compute Mean Squared Error (MSE) between predicted and ground truth trajectories.
    
    Args:
        pred_traj (np.ndarray): Predicted trajectory [N, 6] (roll, pitch, yaw, x, y, z).
        gt_traj (np.ndarray): Ground truth trajectory [N, 6] (roll, pitch, yaw, x, y, z).
    
    Returns:
        float: MSE value.
    """
    mse = np.mean((pred_traj - gt_traj) ** 2)
    return mse

def compute_ate(pred_traj, gt_traj):
    """
    Compute Absolute Trajectory Error (ATE) between predicted and ground truth trajectories.
    ATE is the RMSE of the translation components after alignment.
    
    Args:
        pred_traj (np.ndarray): Predicted trajectory [N, 6] (roll, pitch, yaw, x, y, z).
        gt_traj (np.ndarray): Ground truth trajectory [N, 6] (roll, pitch, yaw, x, y, z).
    
    Returns:
        float: ATE value.
    """
    # Extract translation components (x, y, z)
    pred_trans = pred_traj[:, 3:]
    gt_trans = gt_traj[:, 3:6]
    
    # Compute RMSE of translation errors
    trans_errors = pred_trans - gt_trans
    ate = np.sqrt(np.mean(np.sum(trans_errors ** 2, axis=1)))
    return ate

def plot_trajectories(pred_traj, gt_traj, climate, traj_id, save_dir):
    """
    Plot the predicted and ground truth trajectories and save the plot.
    
    Args:
        pred_traj (np.ndarray): Predicted trajectory [N, 6] (roll, pitch, yaw, x, y, z).
        gt_traj (np.ndarray): Ground truth trajectory [N, 6] (roll, pitch, yaw, x, y, z).
        climate (str): Climate set (e.g., 'Kite_training/sunny').
        traj_id (str): Trajectory ID (e.g., 'trajectory_0008').
        save_dir (str): Directory to save the plot.
    """
    # Extract x, y coordinates for 2D plotting (North-East plane)
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
    plt.axis('equal')  # Ensure equal scaling for x and y axes

    # Save the plot
    climate_sanitized = climate.replace('/', '_')
    plt.savefig(f'{save_dir}/traj_{climate_sanitized}_{traj_id}.png')
    plt.close()

if __name__ == '__main__':    
    # Define the trajectories to test (Mid-Air dataset)
    trajectories_to_test = [
        ('Kite_training/cloudy', 'trajectory_3008'),
        ('Kite_training/sunny', 'trajectory_0008'),
        ('Kite_training/foggy', 'trajectory_2008'),
        ('Kite_training/sunset', 'trajectory_1008')
    ]

    # Path
    load_model_path = par.load_model_path  # Use the Mid-Air model path
    save_dir = 'result/'  # Directory to save prediction answer and plots
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load model
    M_deepvo = StereoAdaptiveVO(par.img_h, par.img_w, par.batch_norm)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        M_deepvo = M_deepvo.cuda()
        M_deepvo.load_state_dict(torch.load(load_model_path, weights_only=True))
    else:
        M_deepvo.load_state_dict(torch.load(load_model_path, map_location={'cuda:0': 'cpu'}, weights_only=True))
    print('Load model from: ', load_model_path)

    # Set model to evaluation mode
    M_deepvo.eval()

    # Data
    n_workers = 1
    seq_len = int((par.seq_len[0] + par.seq_len[1]) / 2)
    overlap = seq_len - 1
    print('seq_len = {},  overlap = {}'.format(seq_len, overlap))
    batch_size = par.batch_size

    fd = open('test_dump.txt', 'w')
    fd.write('\n' + '=' * 50 + '\n')

    for climate, test_traj_id in trajectories_to_test:
        # Adjust the folder list to match Mid-Air structure
        # get_data_info returns a tuple (train_df, valid_df, test_df)
        train_df, valid_df, test_df = get_data_info(
            climate_sets=[climate],
            seq_len_range=[seq_len, seq_len],
            overlap=overlap,
            sample_times=1,
            shuffle=False,
            sort=False,
            include_test=True
        )
        # Use test_df for testing
        df = test_df
        df = df.loc[df.seq_len == seq_len]  # Drop last
        dataset = ImageSequenceDataset(
            df, 
            par.resize_mode, 
            (par.img_h, par.img_w), 
            par.img_means_03, 
            par.img_stds_03, 
            par.img_means_02, 
            par.img_stds_02, 
            par.minus_point_5
        )
        df.to_csv('test_df.csv')
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)
        
        # Load ground truth poses for the specific trajectory
        traj_num = test_traj_id.replace('trajectory_', '')
        gt_pose = np.load(f'{par.pose_dir}/{climate}/poses/poses_{traj_num}.npy')  # (n_images, 6)

        # Predict
        has_predict = False
        st_t = time.time()
        n_batch = len(dataloader)
        all_pred_poses = []

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                print('{} / {}'.format(i, n_batch), end='\r', flush=True)
                _, (x_03, x_02, x_depth, x_imu, x_gps), y = batch  # Mid-Air dataset includes IMU and GPS
                if use_cuda:
                    x_03 = x_03.cuda()
                    x_02 = x_02.cuda()
                    x_depth = x_depth.cuda()
                    x_imu = x_imu.cuda()
                    x_gps = x_gps.cuda()
                    y = y.cuda()
                x = (x_03, x_02, x_depth, x_imu, x_gps)  # Construct the input tuple for StereoAdaptiveVO
                batch_predict_pose = M_deepvo.forward(x)  # Shape: [batch_size, seq_len-1, 6]

                # Record answer
                fd.write('Batch: {}\n'.format(i))
                for seq, predict_pose_seq in enumerate(batch_predict_pose):
                    for pose_idx, pose in enumerate(predict_pose_seq):
                        fd.write(' {} {} {}\n'.format(seq, pose_idx, pose))

                # Convert to numpy for processing
                batch_predict_pose = batch_predict_pose.data.cpu()
                all_pred_poses.append(batch_predict_pose)

        # Concatenate all predicted relative poses
        all_pred_poses = torch.cat(all_pred_poses, dim=0)  # Shape: [total_frames-1, seq_len-1, 6]
        print("Shape of all_pred_poses before reshape:", all_pred_poses.shape)

        # Reshape to combine the first two dimensions into a single sequence dimension
        total_frames_minus_1 = all_pred_poses.shape[0] * all_pred_poses.shape[1]
        all_pred_poses = all_pred_poses.view(total_frames_minus_1, 6)  # Shape: [total_frames-1, 6]
        print("Shape of all_pred_poses after reshape:", all_pred_poses.shape)

        # Add batch dimension for compute_absolute_poses
        all_pred_poses = all_pred_poses.unsqueeze(0)  # Shape: [1, total_frames-1, 6]
        print("Shape of all_pred_poses after unsqueeze:", all_pred_poses.shape)

        # Compute absolute poses from relative poses
        absolute_poses = M_deepvo.compute_absolute_poses(all_pred_poses)  # Shape: [1, total_frames, 6]
        print("Shape of absolute_poses:", absolute_poses.shape)
        absolute_poses = absolute_poses.squeeze(0).numpy()  # Shape: [total_frames, 6]

        # Adjust lengths to match ground truth
        min_len = min(len(absolute_poses), len(gt_pose))
        absolute_poses = absolute_poses[:min_len]
        gt_pose = gt_pose[:min_len]

        print('len(absolute_poses): ', len(absolute_poses))
        print('expect len: ', len(glob.glob(f'{par.image_dir}/{climate}/{test_traj_id}/image_rgb/*.JPEG')))
        print('Predict use {} sec'.format(time.time() - st_t))

        # Save predicted absolute poses
        climate_sanitized = climate.replace('/', '_')
        with open(f'{save_dir}/out_{climate_sanitized}_{test_traj_id}.txt', 'w') as f:
            for pose in absolute_poses:
                f.write(', '.join([str(p) for p in pose]))
                f.write('\n')

        # Calculate loss
        loss = 0
        for t in range(min_len):
            angle_loss = np.sum((absolute_poses[t, :3] - gt_pose[t, :3]) ** 2)
            translation_loss = np.sum((absolute_poses[t, 3:] - gt_pose[t, 3:6]) ** 2)
            loss += (par.k_factor * angle_loss + translation_loss)
        loss /= min_len
        print('Loss = ', loss)

        # Compute MSE and ATE
        mse = compute_mse(absolute_poses, gt_pose)
        ate = compute_ate(absolute_poses, gt_pose)
        print(f'MSE = {mse:.4f}')
        print(f'ATE = {ate:.4f}')

        # Plot trajectories
        plot_trajectories(absolute_poses, gt_pose, climate, test_traj_id, save_dir)

        # Log metrics to file
        with open(f'{save_dir}/metrics_{climate_sanitized}_{test_traj_id}.txt', 'w') as f:
            f.write(f'Loss: {loss:.4f}\n')
            f.write(f'MSE: {mse:.4f}\n')
            f.write(f'ATE: {ate:.4f}\n')

        print('=' * 50)

    fd.close()