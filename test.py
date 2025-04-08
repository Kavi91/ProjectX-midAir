from params import par
from model_2 import StereoAdaptiveVO  # Use the Mid-Air model
import numpy as np
from PIL import Image
import glob
import os
import time
import torch
from data_helper import get_data_info, ImageSequenceDataset
from torch.utils.data import DataLoader
from helper import eulerAnglesToRotationMatrix

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
    save_dir = 'result/'  # Directory to save prediction answer
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
        M_deepvo.eval()
        has_predict = False
        answer = [[0.0] * 6]  # Initialize with the first absolute pose
        st_t = time.time()
        n_batch = len(dataloader)

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
            batch_predict_pose = M_deepvo.forward(x)

            # Record answer
            fd.write('Batch: {}\n'.format(i))
            for seq, predict_pose_seq in enumerate(batch_predict_pose):
                for pose_idx, pose in enumerate(predict_pose_seq):
                    fd.write(' {} {} {}\n'.format(seq, pose_idx, pose))

            batch_predict_pose = batch_predict_pose.data.cpu().numpy()
            if i == 0:
                for pose in batch_predict_pose[0]:
                    # Use all predicted poses in the first prediction
                    for j in range(len(pose)):
                        # Convert predicted relative pose to absolute pose by adding last pose
                        pose[j] += answer[-1][j]
                    answer.append(pose.tolist())
                batch_predict_pose = batch_predict_pose[1:]

            # Transform from relative to absolute
            for predict_pose_seq in batch_predict_pose:
                ang = eulerAnglesToRotationMatrix([0, answer[-1][0], 0])
                location = ang.dot(predict_pose_seq[-1][3:])
                predict_pose_seq[-1][3:] = location[:]

                # Use only the last predicted pose in the following prediction
                last_pose = predict_pose_seq[-1]
                for j in range(len(last_pose)):
                    last_pose[j] += answer[-1][j]
                # Normalize angle to -Pi...Pi over y axis
                last_pose[0] = (last_pose[0] + np.pi) % (2 * np.pi) - np.pi
                answer.append(last_pose.tolist())

        print('len(answer): ', len(answer))
        print('expect len: ', len(glob.glob(f'{par.image_dir}/{climate}/{test_traj_id}/image_rgb/*.JPEG')))
        print('Predict use {} sec'.format(time.time() - st_t))

        # Save answer
        climate_sanitized = climate.replace('/', '_')
        with open(f'{save_dir}/out_{climate_sanitized}_{test_traj_id}.txt', 'w') as f:
            for pose in answer:
                if type(pose) == list:
                    f.write(', '.join([str(p) for p in pose]))
                else:
                    f.write(str(pose))
                f.write('\n')

        # Calculate loss
        gt_pose = np.load(f'{par.pose_dir}/{climate}/poses/poses_{traj_num}.npy')  # (n_images, 6)
        loss = 0
        min_len = min(len(answer), len(gt_pose))
        answer = answer[:min_len]
        gt_pose = gt_pose[:min_len]
        for t in range(min_len):
            angle_loss = np.sum((answer[t][:3] - gt_pose[t, :3]) ** 2)
            translation_loss = np.sum((answer[t][3:] - gt_pose[t, 3:6]) ** 2)
            loss += (100 * angle_loss + translation_loss)
        loss /= min_len
        print('Loss = ', loss)
        print('=' * 50)

    fd.close()