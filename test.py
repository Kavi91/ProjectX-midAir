import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
import glob
import time
import pandas as pd
from params import par
from model import StereoAdaptiveVO
from data_helper import get_data_info, ImageSequenceDataset, SortedRandomBatchSampler
from helper import integrate_relative_poses, to_ned_pose
import torch.nn.functional as F

if __name__ == '__main__':
    # Test trajectories from params.py
    test_videos = []
    for climate_set, traj_ids in par.test_traj_ids.items():
        test_videos.extend([(climate_set, traj) for traj in traj_ids])
    
    # Paths
    load_model_path = par.save_model_path + '.valid'
    save_dir = 'results/test_predictions/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Load model
    M_deepvo = StereoAdaptiveVO(
        img_h=par.img_h,
        img_w=par.img_w,
        batch_norm=par.batch_norm,
        input_channels=3,
        hidden_size=par.rnn_hidden_size,
        num_layers=2
    )
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('Using CUDA.')
        M_deepvo = M_deepvo.cuda()
        M_deepvo.load_state_dict(torch.load(load_model_path))
    else:
        print('Using CPU.')
        M_deepvo.load_state_dict(torch.load(load_model_path, map_location='cpu'))
    print(f'Loaded model from: {load_model_path}')
    
    # Data parameters
    n_workers = min(par.num_workers, 8)
    seq_len = par.seq_len
    overlap = par.overlap
    batch_size = par.batch_size
    print(f'seq_len = {seq_len}, overlap = {overlap}, batch_size = {batch_size}')
    
    # Open log file
    log_file = open(os.path.join(save_dir, 'test_log.txt'), 'w')
    log_file.write('=' * 50 + '\n')
    log_file.write(f'Testing on trajectories: {test_videos}\n')
    
    for climate_set, traj_id in test_videos:
        print(f'\nTesting on {climate_set}/{traj_id}')
        log_file.write(f'\nTesting on {climate_set}/{traj_id}\n')
        
        # Load test data
        train_df, valid_df, test_df = get_data_info(
            climate_sets=[climate_set],
            seq_len=seq_len,
            overlap=overlap,
            sample_times=1,
            shuffle=False,
            sort=False,
            include_test=True
        )
        test_df = test_df[test_df['image_path_03'].apply(lambda x: traj_id in x[0])]
        if test_df.empty:
            print(f'No data found for {traj_id}. Skipping.')
            log_file.write(f'No data found for {traj_id}. Skipping.\n')
            continue
        
        test_dataset = ImageSequenceDataset(
            test_df,
            resize_mode=par.resize_mode,
            new_size=(par.img_h, par.img_w),
            img_means_03=par.img_means_03,
            img_stds_03=par.img_stds_03,
            img_means_02=par.img_means_02,
            img_stds_02=par.img_stds_02,
            minus_point_5=par.minus_point_5,
            is_training=False
        )
        test_sampler = SortedRandomBatchSampler(test_df, batch_size, drop_last=False)
        test_dataloader = DataLoader(
            test_dataset,
            batch_sampler=test_sampler,
            num_workers=n_workers,
            pin_memory=par.pin_mem
        )
        
        # Load ground-truth poses
        traj_num = traj_id.split('_')[1]
        gt_pose_path = f'{par.data_dir}/{climate_set}/poses/poses_{traj_num}.npy'
        if not os.path.exists(gt_pose_path):
            print(f'Ground-truth poses not found at {gt_pose_path}. Skipping.')
            log_file.write(f'Ground-truth poses not found at {gt_pose_path}. Skipping.\n')
            continue
        gt_poses = np.load(gt_pose_path)
        gt_poses = torch.tensor(gt_poses, dtype=torch.float32)
        gt_poses = to_ned_pose(gt_poses, is_absolute=True)
        
        # Debug: Print raw ground truth before scaling
        print("Raw ground truth (first 5 poses, NED, before scaling):")
        print(gt_poses[:5])
        
        # Scale ground truth translations to match GPS plot scale
        gt_poses[:, 3:] /= 30  # Divide by 30 to match the GPS plot's scale
        
        # Print ground-truth trajectory range after scaling
        print("Ground truth trajectory range (NED, after scaling):")
        print(f"North: {gt_poses[:, 3].min().item():.3f} to {gt_poses[:, 3].max().item():.3f}")
        print(f"East: {gt_poses[:, 4].min().item():.3f} to {gt_poses[:, 4].max().item():.3f}")
        print(f"Down: {gt_poses[:, 5].min().item():.3f} to {gt_poses[:, 5].max().item():.3f}")
        log_file.write("Ground truth trajectory range (NED, after scaling):\n")
        log_file.write(f"North: {gt_poses[:, 3].min().item():.3f} to {gt_poses[:, 3].max().item():.3f}\n")
        log_file.write(f"East: {gt_poses[:, 4].min().item():.3f} to {gt_poses[:, 4].max().item():.3f}\n")
        log_file.write(f"Down: {gt_poses[:, 5].min().item():.3f} to {gt_poses[:, 5].max().item():.3f}\n")
        
        # Predict poses
        M_deepvo.eval()
        all_relative_poses = []
        start_time = time.time()
        n_batches = len(test_dataloader)
        
        with torch.no_grad():
            for batch_idx, (batch_seq_len, (x_03, x_02, x_depth, x_imu, x_gps), y) in enumerate(test_dataloader):
                print(f'Batch {batch_idx + 1}/{n_batches}', end='\r', flush=True)
                if use_cuda:
                    x_03 = x_03.cuda(non_blocking=par.pin_mem)
                    x_02 = x_02.cuda(non_blocking=par.pin_mem)
                    x_depth = x_depth.cuda(non_blocking=par.pin_mem)
                    x_imu = x_imu.cuda(non_blocking=par.pin_mem)
                    x_gps = x_gps.cuda(non_blocking=par.pin_mem)
                    y = y.cuda(non_blocking=par.pin_mem)
                
                batch_pred = M_deepvo.forward((x_03, x_02, x_depth, x_imu, x_gps))
                all_relative_poses.append(batch_pred.cpu())
        
        # Concatenate all relative poses
        all_relative_poses = torch.cat(all_relative_poses, dim=0)  # [total_sequences, seq_len, 6]
        print(f'\nPrediction took {time.time() - start_time:.2f} seconds')
        log_file.write(f'Prediction took {time.time() - start_time:.2f} seconds\n')
        
        # Adjust for overlap and create a continuous sequence of relative poses
        jump = seq_len - overlap
        continuous_relative_poses = []
        for i in range(len(all_relative_poses)):
            start_idx = 0 if i == 0 else 1  # Skip first pose of subsequent sequences to avoid overlap
            for j in range(start_idx, seq_len):
                continuous_relative_poses.append(all_relative_poses[i, j])
        continuous_relative_poses = torch.stack(continuous_relative_poses)  # [total_frames-1, 6]
        
        # Debug: Print number of relative poses
        print(f"Number of continuous relative poses: {len(continuous_relative_poses)}")
        log_file.write(f"Number of continuous relative poses: {len(continuous_relative_poses)}\n")
        
        # Compute absolute poses
        continuous_relative_poses = continuous_relative_poses.unsqueeze(0)  # [1, total_frames-1, 6]
        absolute_pred_poses = M_deepvo.compute_absolute_poses(continuous_relative_poses)  # [1, total_frames, 6]
        absolute_pred_poses = absolute_pred_poses[0]  # [total_frames, 6]
        
        # Scale predicted translations to match ground truth scale
        absolute_pred_poses[:, 3:] /= 30  # Divide by 30 to match the ground truth scale
        
        # Debug: Print number of absolute poses before alignment
        print(f"Number of absolute predicted poses (before alignment): {len(absolute_pred_poses)}")
        log_file.write(f"Number of absolute predicted poses (before alignment): {len(absolute_pred_poses)}\n")
        
        # Align the first predicted pose with the first ground-truth pose
        if len(absolute_pred_poses) > 0 and len(gt_poses) > 0:
            offset = gt_poses[0] - absolute_pred_poses[0]
            absolute_pred_poses += offset
        
        # Match length with ground truth
        min_len = min(len(absolute_pred_poses), len(gt_poses))
        absolute_pred_poses = absolute_pred_poses[:min_len].cpu().numpy()
        gt_poses = gt_poses[:min_len].cpu().numpy()
        print(f'Predicted {len(absolute_pred_poses)} poses, Ground truth has {len(gt_poses)} poses')
        log_file.write(f'Predicted {len(absolute_pred_poses)} poses, Ground truth has {len(gt_poses)} poses\n')
        
        # Debug: Print first few poses
        print("First 5 ground-truth poses (NED, after scaling):")
        print(gt_poses[:5])
        print("First 5 predicted absolute poses (NED, after scaling):")
        print(absolute_pred_poses[:5])
        log_file.write("First 5 ground-truth poses (NED, after scaling):\n")
        log_file.write(str(gt_poses[:5]) + '\n')
        log_file.write("First 5 predicted absolute poses (NED, after scaling):\n")
        log_file.write(str(absolute_pred_poses[:5]) + '\n')
        
        # Print predicted trajectory range
        print("Predicted trajectory range (NED, after scaling):")
        print(f"North: {absolute_pred_poses[:, 3].min():.3f} to {absolute_pred_poses[:, 3].max():.3f}")
        print(f"East: {absolute_pred_poses[:, 4].min():.3f} to {absolute_pred_poses[:, 4].max():.3f}")
        print(f"Down: {absolute_pred_poses[:, 5].min():.3f} to {absolute_pred_poses[:, 5].max():.3f}")
        log_file.write("Predicted trajectory range (NED, after scaling):\n")
        log_file.write(f"North: {absolute_pred_poses[:, 3].min():.3f} to {absolute_pred_poses[:, 3].max():.3f}\n")
        log_file.write(f"East: {absolute_pred_poses[:, 4].min():.3f} to {absolute_pred_poses[:, 4].max():.3f}\n")
        log_file.write(f"Down: {absolute_pred_poses[:, 5].min():.3f} to {absolute_pred_poses[:, 5].max():.3f}\n")
        
        # Save predictions
        output_file = os.path.join(save_dir, f'pred_{traj_id}.txt')
        with open(output_file, 'w') as f:
            for pose in absolute_pred_poses:
                f.write(','.join([str(x) for x in pose]) + '\n')
        print(f'Saved predictions to {output_file}')
        log_file.write(f'Saved predictions to {output_file}\n')
        
        # Calculate loss
        pred_tensor = torch.tensor(absolute_pred_poses, dtype=torch.float32)
        gt_tensor = torch.tensor(gt_poses, dtype=torch.float32)
        if use_cuda:
            pred_tensor = pred_tensor.cuda()
            gt_tensor = gt_tensor.cuda()
        
        angle_loss = F.mse_loss(pred_tensor[:, :3], gt_tensor[:, :3])
        translation_loss = F.mse_loss(pred_tensor[:, 3:], gt_tensor[:, 3:])
        total_loss = par.k_factor * angle_loss + par.translation_loss_weight * translation_loss
        
        print(f'Angle Loss (MSE): {angle_loss.item():.6f}')
        print(f'Translation Loss (MSE): {translation_loss.item():.6f}')
        print(f'Total Loss: {total_loss.item():.6f}')
        log_file.write(f'Angle Loss (MSE): {angle_loss.item():.6f}\n')
        log_file.write(f'Translation Loss (MSE): {translation_loss.item():.6f}\n')
        log_file.write(f'Total Loss: {total_loss.item():.6f}\n')
        
        log_file.write('=' * 50 + '\n')
    
    log_file.close()
    print('Testing completed.')