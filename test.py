import os
import time
import glob
import numpy as np
import torch
import argparse
from params import Parameters, par
from model_2 import StereoAdaptiveVO
from data_helper import get_data_info, ImageSequenceDataset
from torch.utils.data import DataLoader
from helper import normalize_angle_delta
from scipy.spatial.transform import Rotation

def euler_to_rotation_matrix(roll, pitch, yaw):
    return Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=False).as_matrix()

def rotation_matrix_to_euler(R):
    return Rotation.from_matrix(R).as_euler('xyz', degrees=False)

def compute_absolute_poses(relative_poses):
    """Integrate a sequence of relative poses starting from [0,0,0,0,0,0]."""
    absolute_poses = [[0.0]*6]  # Start at 0.
    for rel_pose in relative_poses:
        roll_rel, pitch_rel, yaw_rel = rel_pose[:3]
        t_rel = rel_pose[3:]
        last_pose = absolute_poses[-1]
        R_last = euler_to_rotation_matrix(last_pose[0], last_pose[1], last_pose[2])
        t_rel_global = R_last @ t_rel
        t_abs = np.array(last_pose[3:]) + t_rel_global
        R_rel = euler_to_rotation_matrix(roll_rel, pitch_rel, yaw_rel)
        R_abs = R_last @ R_rel
        roll_abs, pitch_abs, yaw_abs = rotation_matrix_to_euler(R_abs)
        roll_abs = normalize_angle_delta(roll_abs)
        pitch_abs = normalize_angle_delta(pitch_abs)
        yaw_abs = normalize_angle_delta(yaw_abs)
        absolute_pose = [roll_abs, pitch_abs, yaw_abs, t_abs[0], t_abs[1], t_abs[2]]
        absolute_poses.append(absolute_pose)
    return absolute_poses[1:]

def evaluate_ate(pred_poses, gt_poses):
    """Compute Absolute Trajectory Error (ATE) on the translation components."""
    min_len = min(len(pred_poses), len(gt_poses))
    pred_poses = np.array(pred_poses)[:min_len, 3:]  # Only x,y,z
    gt_poses = np.array(gt_poses)[:min_len, 3:6]
    errors = np.sqrt(np.sum((pred_poses - gt_poses)**2, axis=1))
    return np.mean(errors)

def test_model_aggregated(model, dataloader, use_cuda, output_file, pose_gt_file, debug_file, climate, seq_len):
    """
    Processes overlapping sequences, aggregates predictions for each unique frame, and evaluates against ground truth.
    
    Assumptions:
     - Test DataLoader returns sequences in temporal order with stride 1.
     - Each sequence covers exactly 'seq_len' frames.
     - For a sequence starting at frame index s, the integrated relative predictions correspond to frames:
           s+1, s+2, …, s+(seq_len-1)
    """
    model.eval()
    # Load full ground truth trajectory.
    gt_full = np.load(pose_gt_file)  # shape: (T, 6)
    T = gt_full.shape[0]
    # Create a dictionary to gather predictions for each frame index.
    frame_predictions = {i: [] for i in range(T)}
    
    st_t = time.time()
    seq_global_idx = 0  # Overall sequence counter.
    with open(debug_file, 'a') as fd:
        fd.write(f'\nTesting Climate: {climate}\n')
        fd.write('='*50 + '\n')
        for i, batch in enumerate(dataloader):
            print(f"Processing batch {i+1}/{len(dataloader)}", end='\r', flush=True)
            _, (x_03, x_02, x_depth, x_imu, x_gps), y = batch
            if use_cuda:
                x_03 = x_03.cuda(non_blocking=par.pin_mem)
                x_02 = x_02.cuda(non_blocking=par.pin_mem)
                x_depth = x_depth.cuda(non_blocking=par.pin_mem)
                x_imu = x_imu.cuda(non_blocking=par.pin_mem)
                x_gps = x_gps.cuda(non_blocking=par.pin_mem)
                y = y.cuda(non_blocking=par.pin_mem)
            x = (x_03, x_02, x_depth, x_imu, x_gps)
            with torch.no_grad():
                batch_predict_pose = model.forward(x)
            # batch_predict_pose: (batch_size, seq_len-1, 6)
            batch_predict_pose = batch_predict_pose.data.cpu().numpy()
            batch_gt_pose = y.data.cpu().numpy()  # (batch_size, seq_len, 6)
            for b in range(batch_predict_pose.shape[0]):
                rel_pred_seq = batch_predict_pose[b]  # length = seq_len-1
                gt_seq = batch_gt_pose[b]            # length = seq_len
                # Integrate the relative predictions.
                computed_abs = compute_absolute_poses(rel_pred_seq)
                # Align predictions to ground truth using the first predicted frame:
                offset = np.array(gt_seq[1]) - np.array(computed_abs[0])
                computed_abs_aligned = [list(np.array(p) + offset) for p in computed_abs]
                # For this sequence, assign predictions to frames:
                # If this sequence starts at frame index 's' (from your sampling order),
                # then the computed predictions correspond to frames s+1 ... s+(seq_len-1).
                start_frame = seq_global_idx
                for k, pred in enumerate(computed_abs_aligned):
                    frame_idx = start_frame + 1 + k  # prediction index in global trajectory.
                    if frame_idx < T:  # Only assign if within full trajectory.
                        frame_predictions[frame_idx].append(pred)
                fd.write(f"Sequence {seq_global_idx}: offset = {offset}\n")
                seq_global_idx += 1

    # Now aggregate: For each frame, average all predictions.
    aggregated_preds = []
    valid_indices = []  # To keep track of frames with predictions.
    for i in range(T):
        preds = frame_predictions[i]
        if len(preds) > 0:
            aggregated_preds.append(np.mean(preds, axis=0))
            valid_indices.append(i)
    aggregated_preds = np.array(aggregated_preds)
    gt_valid = gt_full[valid_indices]  # Corresponding ground truth poses.
    
    ate_error = evaluate_ate(aggregated_preds, gt_valid)
    with open(output_file, 'w') as f:
        for pose in aggregated_preds:
            f.write(', '.join([str(p) for p in pose]) + '\n')
    
    print(f"\nAggregated predictions computed for {len(valid_indices)} frames out of {T}")
    print(f"Total prediction time: {time.time()-st_t:.1f} sec")
    print(f"First 5 aggregated predictions:\n {aggregated_preds[:5]}")
    print(f"First 5 ground truth poses:\n {gt_valid[:5]}")
    print(f"ATE Error: {ate_error:.4f} meters")
    
    # Optionally compute additional losses.
    angle_loss_total = 0
    translation_loss_total = 0
    for pred_pose, gt_pose in zip(aggregated_preds, gt_valid):
        angle_loss_total += np.sum((np.array(pred_pose[:3]) - gt_pose[:3])**2)
        translation_loss_total += np.sum((np.array(pred_pose[3:]) - gt_pose[3:6])**2)
    angle_loss_avg = angle_loss_total / len(aggregated_preds)
    translation_loss_avg = translation_loss_total / len(aggregated_preds)
    combined_loss = 10*angle_loss_avg + translation_loss_avg
    print(f"Angle Loss (MSE) = {angle_loss_avg:.6f}")
    print(f"Translation Loss (MSE) = {translation_loss_avg:.6f}")
    print(f"Combined Loss = {combined_loss:.6f}")
    
    return angle_loss_avg, translation_loss_avg, combined_loss, ate_error, aggregated_preds, gt_valid

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepVO Testing with Aggregation')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for testing')
    args = parser.parse_args()
    
    par = Parameters(batch_size=args.batch_size)
    save_dir = 'result'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Load model
    M_deepvo = StereoAdaptiveVO(par.img_h, par.img_w, par.batch_norm)
    use_cuda = torch.cuda.is_available()
    model_path = '/home/krkavinda/ProjectX-midAir/models/midair_im184x608_s5x7_b24_rnn1000_optAdam_lr0.0001_weight_decay0.0001.model.valid'
    if use_cuda:
        M_deepvo = M_deepvo.cuda()
        M_deepvo.load_state_dict(torch.load(model_path, weights_only=True))
    else:
        M_deepvo.load_state_dict(torch.load(model_path, map_location={'cuda:0':'cpu'}, weights_only=True))
    print('Loaded model from:', model_path)
    
    # Use a fixed sequence length—if par.seq_len is (5,7), choose the average.
    seq_len = int((par.seq_len[0] + par.seq_len[1]) / 2)  # e.g., 6
    # For the test trajectory; assume par.test_traj_ids for 'Kite_training/sunny' is ['trajectory_0008'].
    climate = 'Kite_training/sunny'
    test_traj_id = par.test_traj_ids[climate][0]  # 'trajectory_0008'
    traj_num = test_traj_id.replace('trajectory_', '')
    pose_gt_file = f'{par.pose_dir}/{climate}/poses/poses_{traj_num}.npy'
    output_file = f'{save_dir}/out_{climate.replace("/", "_")}_{test_traj_id}.txt'
    debug_file = f'{save_dir}/aggregated_debug_{climate.replace("/", "_")}_{test_traj_id}.txt'
    
    # Get test dataframe (filtering for consistent sequence length)
    _, _, test_df = get_data_info(
        climate_sets=[climate],
        seq_len_range=[seq_len, seq_len],
        overlap=seq_len-1,   # stride=1
        sample_times=1,
        shuffle=False,
        sort=False,
        include_test=True
    )
    test_df = test_df.loc[test_df.seq_len == seq_len]
    dataset = ImageSequenceDataset(
        test_df, par.resize_mode, (par.img_h, par.img_w),
        par.img_means_03, par.img_stds_03, par.img_means_02, par.img_stds_02, par.minus_point_5
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
    
    # Run aggregated inference.
    test_model_aggregated(M_deepvo, dataloader, use_cuda, output_file, pose_gt_file, debug_file, climate, seq_len)
