import os
import glob
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import transforms
import time
import pickle
from params import par
from helper import normalize_angle_delta, euler_to_rotation_matrix, to_ned_pose
from augmentations import get_transformer

def get_data_info(climate_sets, seq_len, overlap, sample_times=1, pad_y=False, shuffle=False, sort=True, include_test=False):
    all_traj_info = []
    for climate_set in climate_sets:
        traj_list = [d for d in os.listdir(f"{par.data_dir}/{climate_set}/") if d.startswith("trajectory_")]
        traj_list.sort()
        print(f"Climate set {climate_set}: {len(traj_list)} trajectories available")
        all_traj_info.extend([(climate_set, traj) for traj in traj_list])
    
    train_traj_info = []
    valid_traj_info = []
    test_traj_info = []
    
    for climate_set in climate_sets:
        for traj in par.train_traj_ids.get(climate_set, []):
            if (climate_set, traj) in all_traj_info:
                train_traj_info.append((climate_set, traj))
            else:
                print(f"Warning: Trajectory {traj} not found in {climate_set}")
        for traj in par.valid_traj_ids.get(climate_set, []):
            if (climate_set, traj) in all_traj_info:
                valid_traj_info.append((climate_set, traj))
            else:
                print(f"Warning: Trajectory {traj} not found in {climate_set}")
        if include_test:
            for traj in par.test_traj_ids.get(climate_set, []):
                if (climate_set, traj) in all_traj_info:
                    test_traj_info.append((climate_set, traj))
                else:
                    print(f"Warning: Trajectory {traj} not found in {climate_set}")
    
    print(f"Trajectories: Train: {len(train_traj_info)}, Valid: {len(valid_traj_info)}, Test: {len(test_traj_info)}")
    if set(train_traj_info) & set(valid_traj_info):
        raise ValueError("Overlap between train and valid trajectories!")
    
    def process_trajectories(traj_info, data_dict):
        for climate_set, traj in traj_info:
            start_t = time.time()
            traj_id = traj.split('_')[1]
            poses_path = f"{par.data_dir}/{climate_set}/poses/poses_{traj_id}.npy"
            if not os.path.exists(poses_path):
                print(f"Poses file not found for {climate_set}/{traj}, skipping.")
                continue
            poses = np.load(poses_path)
            poses = torch.tensor(poses, dtype=torch.float32)
            # Debug: Print conversion for pose ground truth.
            poses = to_ned_pose(poses, is_absolute=True, debug=True)
            
            # Collect image file paths.
            fpaths_03 = glob.glob(f"{par.data_dir}/{climate_set}/{traj}/image_rgb/*.JPEG")
            fpaths_02 = glob.glob(f"{par.data_dir}/{climate_set}/{traj}/image_rgb_right/*.JPEG")
            fpaths_03.sort()
            fpaths_02.sort()
            fpaths_depth = glob.glob(f"{par.data_dir}/{climate_set}/{traj}/depth/*.PNG") if par.enable_depth else []
            if par.enable_depth:
                fpaths_depth.sort()
            fpaths_imu = f"{par.data_dir}/{climate_set}/{traj}/imu.npy" if par.enable_imu else None
            fpaths_gps = f"{par.data_dir}/{climate_set}/{traj}/gps.npy" if par.enable_gps else None
            
            print(f"Trajectory {traj}: Left RGB: {len(fpaths_03)}, Right RGB: {len(fpaths_02)}")
            if par.enable_depth:
                print(f"Depth: {len(fpaths_depth)}")
            if par.enable_imu:
                print(f"IMU file: {fpaths_imu}")
            if par.enable_gps:
                print(f"GPS file: {fpaths_gps}")
            
            # Determine common length among images and poses.
            min_len = min(len(fpaths_03), len(fpaths_02), len(poses))
            if par.enable_depth and len(fpaths_depth) > 0:
                min_len = min(min_len, len(fpaths_depth))
            if min_len < len(fpaths_03):
                print(f"Warning: Trajectory {traj} has excess image files; truncating to {min_len}.")
                fpaths_03 = fpaths_03[:min_len]
                fpaths_02 = fpaths_02[:min_len]
                if par.enable_depth:
                    fpaths_depth = fpaths_depth[:min_len]
                poses = poses[:min_len]
            
            # Skip trajectories that are too short to form even one full sequence
            if min_len < seq_len:
                print(f"Warning: Trajectory {traj} has only {min_len} frames, which is less than seq_len={seq_len}. Skipping.")
                continue
            
            # For IMU and GPS, calculate frequency ratios.
            ratio_imu = 1.0
            if par.enable_imu and fpaths_imu and os.path.exists(fpaths_imu):
                imu_data = np.load(fpaths_imu)
                ratio_imu = len(imu_data) / float(len(fpaths_03))
                print(f"IMU ratio: {ratio_imu:.3f}")
            ratio_gps = 1.0
            if par.enable_gps and fpaths_gps and os.path.exists(fpaths_gps):
                gps_data = np.load(fpaths_gps)
                ratio_gps = len(gps_data) / float(len(fpaths_03))
                print(f"GPS ratio: {ratio_gps:.3f}")
            
            start_frames = [0] if sample_times <= 1 else list(range(0, seq_len, int(np.ceil(seq_len / sample_times))))
            if sample_times > 1:
                print(f"Sampling start frames: {start_frames}")
            
            for st in start_frames:
                n_frames = len(fpaths_03) - st
                jump = seq_len - overlap
                num_sequences = (n_frames - seq_len) // jump + 1
                print(f"Trajectory {traj}: Generating {num_sequences} sequences")
                x_segs_03, x_segs_02, x_segs_depth, x_segs_imu, x_segs_gps, y_segs = [], [], [], [], [], []
                for i in range(num_sequences):
                    start_idx = st + i * jump
                    end_idx = start_idx + seq_len
                    if end_idx > len(fpaths_03):
                        end_idx = len(fpaths_03)
                        start_idx = max(0, end_idx - seq_len)
                    x_seg_03 = fpaths_03[start_idx:end_idx]
                    x_seg_02 = fpaths_02[start_idx:end_idx]
                    x_seg_depth = fpaths_depth[start_idx:end_idx] if par.enable_depth else []
                    if par.enable_imu:
                        x_seg_imu = (fpaths_imu, start_idx, end_idx, ratio_imu)
                    else:
                        x_seg_imu = None
                    if par.enable_gps:
                        x_seg_gps = (fpaths_gps, start_idx, end_idx, ratio_gps)
                    else:
                        x_seg_gps = None
                    y_seg = poses[start_idx:end_idx]
                    # Only include sequences that have the full seq_len
                    if len(x_seg_03) == seq_len:
                        x_segs_03.append(x_seg_03)
                        x_segs_02.append(x_seg_02)
                        x_segs_depth.append(x_seg_depth)
                        x_segs_imu.append(x_seg_imu)
                        x_segs_gps.append(x_seg_gps)
                        y_segs.append(y_seg)
                    else:
                        print(f"Warning: Sequence at start_idx={start_idx} in trajectory {traj} has length {len(x_seg_03)}, expected {seq_len}. Skipping.")
                data_dict['Y'] += y_segs
                data_dict['X_path_03'] += x_segs_03
                data_dict['X_path_02'] += x_segs_02
                data_dict['X_path_depth'] += x_segs_depth
                data_dict['X_path_imu'] += x_segs_imu
                data_dict['X_path_gps'] += x_segs_gps
                data_dict['X_len'] += [len(xs) for xs in x_segs_03]
                print(f"Trajectory {traj}: Added {len(x_segs_03)} sequences after filtering")
            print(f"Finished trajectory {traj} in {time.time() - start_t:.2f} sec")
    
    train_data = {'X_path_03': [], 'X_path_02': [], 'X_path_depth': [], 'X_path_imu': [], 'X_path_gps': [], 'Y': [], 'X_len': []}
    valid_data = {'X_path_03': [], 'X_path_02': [], 'X_path_depth': [], 'X_path_imu': [], 'X_path_gps': [], 'Y': [], 'X_len': []}
    test_data = {'X_path_03': [], 'X_path_02': [], 'X_path_depth': [], 'X_path_imu': [], 'X_path_gps': [], 'Y': [], 'X_len': []}
    
    process_trajectories(train_traj_info, train_data)
    process_trajectories(valid_traj_info, valid_data)
    if include_test:
        process_trajectories(test_traj_info, test_data)
    
    def create_df(data_dict, name):
        df = pd.DataFrame({
            'seq_len': data_dict['X_len'],
            'image_path_03': data_dict['X_path_03'],
            'image_path_02': data_dict['X_path_02'],
            'depth_path': data_dict['X_path_depth'],
            'imu_path': data_dict['X_path_imu'],
            'gps_path': data_dict['X_path_gps'],
            'pose': data_dict['Y']
        })
        if shuffle:
            df = df.sample(frac=1)
        if sort:
            df = df.sort_values(by=['seq_len'], ascending=False)
        print(f"Total {name} sequences generated: {len(data_dict['X_path_03'])}")
        return df

    train_df = create_df(train_data, "training")
    valid_df = create_df(valid_data, "validation")
    test_df = create_df(test_data, "test") if include_test else None
    
    if include_test:
        return train_df, valid_df, test_df
    return train_df, valid_df

def get_partition_data_info(partition, climate_sets, seq_len, overlap, sample_times=1, pad_y=False, shuffle=False, sort=True):
    return get_data_info(climate_sets, seq_len, overlap, sample_times, pad_y, shuffle, sort)

class SortedRandomBatchSampler(Sampler):
    def __init__(self, info_dataframe, batch_size, drop_last=False):
        self.df = info_dataframe
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.unique_seq_lens = sorted(self.df.iloc[:].seq_len.unique(), reverse=True)
        print(f"Unique sequence lengths in dataset: {self.unique_seq_lens}")
        self.len = 0
        for v in self.unique_seq_lens:
            n_sample = len(self.df.loc[self.df.seq_len == v])
            n_batch = int(n_sample / self.batch_size)
            if not self.drop_last and n_sample % self.batch_size != 0:
                n_batch += 1
            self.len += n_batch

    def __iter__(self):
        list_batch_indexes = []
        start_idx = 0
        for v in self.unique_seq_lens:
            n_sample = len(self.df.loc[self.df.seq_len == v])
            n_batch = int(n_sample / self.batch_size)
            if not self.drop_last and n_sample % self.batch_size != 0:
                n_batch += 1
            rand_idxs = (start_idx + torch.randperm(n_sample)).tolist()
            tmp = [rand_idxs[s * self.batch_size: s * self.batch_size + self.batch_size] for s in range(n_batch)]
            list_batch_indexes += tmp
            start_idx += n_sample
        return iter(list_batch_indexes)

    def __len__(self):
        return self.len

class ImageSequenceDataset(Dataset):
    def __init__(self, info_dataframe, resize_mode='crop', new_size=None, img_means_03=None, img_stds_03=(1,1,1),
                 img_means_02=None, img_stds_02=(1,1,1), minus_point_5=False, is_training=False):
        stats_pickle_path = "datainfo/dataset_stats.pickle"
        self.depth_max = par.depth_max
        if os.path.exists(stats_pickle_path):
            print(f"Loading dataset statistics from {stats_pickle_path}")
            with open(stats_pickle_path, 'rb') as f:
                stats = pickle.load(f)
            self.depth_mean = stats['depth_mean']
            self.depth_std = stats['depth_std']
            self.imu_acc_mean = stats['imu_acc_mean']
            self.imu_acc_std = stats['imu_acc_std']
            self.imu_gyro_mean = stats['imu_gyro_mean']
            self.imu_gyro_std = stats['imu_gyro_std']
            self.gps_pos_mean = stats['gps_pos_mean']
            self.gps_pos_std = stats['gps_pos_std']
            self.gps_vel_mean = stats['gps_vel_mean']
            self.gps_vel_std = stats['gps_vel_std']
            self.depth_max = stats['depth_max']
        else:
            print("Computing dataset statistics on a limited subset (first 100 sequences per climate set)...")
            transform_ops = []
            if resize_mode == 'crop':
                transform_ops.append(transforms.CenterCrop((new_size[0], new_size[1])))
            elif resize_mode == 'rescale':
                transform_ops.append(transforms.Resize((new_size[0], new_size[1])))
            transform_ops.append(transforms.ToTensor())
            self.transformer = transforms.Compose(transform_ops)
            self.minus_point_5 = minus_point_5
            self.normalizer_03 = transforms.Normalize(mean=img_means_03, std=img_stds_03)
            self.normalizer_02 = transforms.Normalize(mean=img_means_02, std=img_stds_02)
            
            # Group sequences by climate set
            info_dataframe['climate_set'] = info_dataframe['image_path_03'].apply(lambda x: x[0].split('/')[4])  # Extract climate set from path
            grouped = info_dataframe.groupby('climate_set')
            
            # Limit to first 1000 sequences per climate set
            limited_df = pd.DataFrame()
            max_sequences_per_climate = 100
            for climate_set, group in grouped:
                limited_group = group.head(max_sequences_per_climate)
                limited_df = pd.concat([limited_df, limited_group])
            
            if par.enable_depth:
                depth_values = []
                for index, depth_path_seq in limited_df.iterrows():
                    depth_path_seq = depth_path_seq['depth_path']
                    for depth_path in depth_path_seq:
                        if os.path.exists(depth_path):
                            depth_img = Image.open(depth_path)
                            depth_array = np.array(depth_img, dtype=np.uint16)
                            depth_float16 = depth_array.view(np.float16)
                            depth_map = depth_float16.astype(np.float32)
                            depth_map = depth_map * (self.depth_max / 65535.0)
                            depth_values.extend(depth_map.flatten())
                depth_values = np.array(depth_values)
                depth_values = depth_values[depth_values > 0]
                self.depth_mean = depth_values.mean() if len(depth_values) > 0 else 0.0
                self.depth_std = depth_values.std() if len(depth_values) > 0 else 1.0
                print(f"Computed depth mean: {self.depth_mean:.4f}, std: {self.depth_std:.4f}")
                self.normalizer_depth = transforms.Normalize(mean=(self.depth_mean / self.depth_max,), std=(self.depth_std / self.depth_max,))
            else:
                self.depth_mean = 0.0
                self.depth_std = 1.0
                self.normalizer_depth = transforms.Normalize(mean=(0.0,), std=(1.0,))
            if par.enable_imu:
                imu_values_acc = []
                imu_values_gyro = []
                for imu_path_info in limited_df['imu_path']:
                    imu_path, start_idx, end_idx, _ = imu_path_info
                    if imu_path and os.path.exists(imu_path):
                        imu_data = np.load(imu_path)[start_idx:end_idx]
                        imu_values_acc.extend(imu_data[:, :3].flatten())
                        imu_values_gyro.extend(imu_data[:, 3:].flatten())
                imu_values_acc = np.array(imu_values_acc)
                imu_values_gyro = np.array(imu_values_gyro)
                self.imu_acc_mean = imu_values_acc.mean() if len(imu_values_acc) > 0 else 0.0
                self.imu_acc_std = imu_values_acc.std() if len(imu_values_acc) > 0 else 1.0
                self.imu_gyro_mean = imu_values_gyro.mean() if len(imu_values_gyro) > 0 else 0.0
                self.imu_gyro_std = imu_values_gyro.std() if len(imu_values_gyro) > 0 else 1.0
                print(f"Computed IMU acc mean: {self.imu_acc_mean:.4f}, std: {self.imu_acc_std:.4f}")
                print(f"Computed IMU gyro mean: {self.imu_gyro_mean:.4f}, std: {self.imu_gyro_std:.4f}")
            else:
                self.imu_acc_mean = self.imu_acc_std = self.imu_gyro_mean = self.imu_gyro_std = 0.0
            if par.enable_gps:
                gps_values_pos = []
                gps_values_vel = []
                for gps_path_info in limited_df['gps_path']:
                    gps_path, start_idx, end_idx, _ = gps_path_info
                    if gps_path and os.path.exists(gps_path):
                        gps_data = np.load(gps_path)[start_idx:end_idx]
                        gps_values_pos.extend(gps_data[:, :3].flatten())
                        gps_values_vel.extend(gps_data[:, 3:].flatten())
                gps_values_pos = np.array(gps_values_pos)
                gps_values_vel = np.array(gps_values_vel)
                self.gps_pos_mean = gps_values_pos.mean() if len(gps_values_pos) > 0 else 0.0
                self.gps_pos_std = gps_values_pos.std() if len(gps_values_pos) > 0 else 1.0
                self.gps_vel_mean = gps_values_vel.mean() if len(gps_values_vel) > 0 else 0.0
                self.gps_vel_std = gps_values_vel.std() if len(gps_values_vel) > 0 else 1.0
                print(f"Computed GPS pos mean: {self.gps_pos_mean:.4f}, std: {self.gps_pos_std:.4f}")
                print(f"Computed GPS vel mean: {self.gps_vel_mean:.4f}, std: {self.gps_vel_std:.4f}")
            else:
                self.gps_pos_mean = self.gps_pos_std = self.gps_vel_mean = self.gps_vel_std = 0.0

            stats = {
                'depth_mean': self.depth_mean,
                'depth_std': self.depth_std,
                'imu_acc_mean': self.imu_acc_mean,
                'imu_acc_std': self.imu_acc_std,
                'imu_gyro_mean': self.imu_gyro_mean,
                'imu_gyro_std': self.imu_gyro_std,
                'gps_pos_mean': self.gps_pos_mean,
                'gps_pos_std': self.gps_pos_std,
                'gps_vel_mean': self.gps_vel_mean,
                'gps_vel_std': self.gps_vel_std,
                'depth_max': self.depth_max
            }
            with open(stats_pickle_path, 'wb') as f:
                pickle.dump(stats, f)
            print(f"Saved dataset statistics to {stats_pickle_path}")

        # Use external augmentation for the transformation pipeline
        self.transformer = get_transformer(resize_mode, new_size, is_training)
        self.minus_point_5 = minus_point_5
        self.normalizer_03 = transforms.Normalize(mean=img_means_03, std=img_stds_03)
        self.normalizer_02 = transforms.Normalize(mean=img_means_02, std=img_stds_02)
        self.normalizer_depth = transforms.Normalize(mean=(self.depth_mean / self.depth_max,), std=(self.depth_std / self.depth_max,))

        self.data_info = info_dataframe
        self.seq_len_list = list(self.data_info.seq_len)
        self.image_arr_03 = np.asarray(self.data_info.image_path_03)
        self.image_arr_02 = np.asarray(self.data_info.image_path_02)
        self.depth_arr = np.asarray(self.data_info.depth_path)
        self.imu_arr = np.asarray(self.data_info.imu_path)
        self.gps_arr = np.asarray(self.data_info.gps_path)
        self.groundtruth_arr = np.asarray(self.data_info.pose)

    def __getitem__(self, index):
        raw_groundtruth = self.groundtruth_arr[index]
        # Compute relative poses with debug prints for conversion consistency.
        relative_poses = []
        for i in range(len(raw_groundtruth)):
            if i == 0:
                relative_pose = torch.zeros(6, dtype=torch.float32)
            else:
                # Use clone().detach() instead of torch.tensor()
                delta_pose = raw_groundtruth[i].clone().detach() - raw_groundtruth[i-1].clone().detach()
                roll_rel = normalize_angle_delta(delta_pose[0])
                pitch_rel = normalize_angle_delta(delta_pose[1])
                yaw_rel = normalize_angle_delta(delta_pose[2])
                t_rel = delta_pose[3:]
                roll_prev, pitch_prev, yaw_prev = raw_groundtruth[i-1][:3]
                R_prev = torch.tensor(euler_to_rotation_matrix(roll_prev, pitch_prev, yaw_prev), dtype=torch.float32)
                t_rel_transformed = torch.matmul(R_prev.T, t_rel.unsqueeze(-1)).squeeze(-1)
                relative_pose = torch.tensor([roll_rel, pitch_rel, yaw_rel,
                                            t_rel_transformed[0], t_rel_transformed[1], t_rel_transformed[2]], dtype=torch.float32)
                # Convert relative pose to NED
                relative_pose = to_ned_pose(relative_pose.unsqueeze(0), is_absolute=False).squeeze(0)
            relative_poses.append(relative_pose)
        groundtruth_sequence = torch.stack(relative_poses)
        
        image_path_sequence_03 = self.image_arr_03[index]
        image_path_sequence_02 = self.image_arr_02[index]
        depth_path_sequence = self.depth_arr[index]
        imu_path_info = self.imu_arr[index]
        gps_path_info = self.gps_arr[index]
        sequence_len = torch.tensor(self.seq_len_list[index])
        expected_len = len(image_path_sequence_03)
        
        image_sequence_03 = [self.normalizer_03(self.transformer(Image.open(img_path)) - (0.5 if self.minus_point_5 else 0)).unsqueeze(0)
                             for img_path in image_path_sequence_03]
        image_sequence_03 = torch.cat(image_sequence_03, 0)
        image_sequence_02 = [self.normalizer_02(self.transformer(Image.open(img_path)) - (0.5 if self.minus_point_5 else 0)).unsqueeze(0)
                             for img_path in image_path_sequence_02]
        image_sequence_02 = torch.cat(image_sequence_02, 0)
        
        if par.enable_depth:
            depth_sequence = []
            if len(depth_path_sequence) != expected_len:
                print(f"Warning: Depth sequence length ({len(depth_path_sequence)}) does not match RGB ({expected_len}) at index {index}. Adjusting.")
                depth_path_sequence = list(depth_path_sequence)
                if len(depth_path_sequence) > expected_len:
                    depth_path_sequence = depth_path_sequence[:expected_len]
                else:
                    depth_path_sequence.extend([None] * (expected_len - len(depth_path_sequence)))
            for depth_path in depth_path_sequence:
                if depth_path is None or not os.path.exists(depth_path):
                    depth_as_tensor = torch.zeros((1, par.img_h, par.img_w))
                else:
                    depth_img = Image.open(depth_path)
                    depth_array = np.array(depth_img, dtype=np.uint16)
                    depth_float16 = depth_array.view(np.float16)
                    depth_map = depth_float16.astype(np.float32)
                    depth_map = depth_map * (self.depth_max / 65535.0)
                    depth_as_tensor = torch.from_numpy(depth_map).float() / self.depth_max
                    depth_as_tensor = transforms.Resize((par.img_h, par.img_w))(depth_as_tensor.unsqueeze(0))
                    depth_as_tensor = self.normalizer_depth(depth_as_tensor)
                depth_sequence.append(depth_as_tensor.unsqueeze(0))
            depth_sequence = torch.cat(depth_sequence, 0)
        else:
            depth_sequence = torch.zeros((expected_len, 1, par.img_h, par.img_w))
        
        if par.enable_imu:
            imu_path, start_idx, end_idx, ratio_imu = imu_path_info if imu_path_info else (None, 0, 0, 1.0)
            if not imu_path or not os.path.exists(imu_path):
                imu_sequence = torch.zeros((expected_len, 6))
            else:
                imu_data = np.load(imu_path)
                adjusted_start = int(start_idx * ratio_imu)
                adjusted_end = int(end_idx * ratio_imu)
                imu_data = imu_data[adjusted_start:adjusted_end]
                imu_data = torch.tensor(imu_data, dtype=torch.float32)
                # Convert IMU data to NED convention.
                imu_data_ned = imu_data.clone()
                imu_data_ned[:, 0] = imu_data[:, 2]
                imu_data_ned[:, 1] = imu_data[:, 1]
                imu_data_ned[:, 2] = -imu_data[:, 0]
                imu_data_ned[:, 3] = imu_data[:, 5]
                imu_data_ned[:, 4] = imu_data[:, 4]
                imu_data_ned[:, 5] = -imu_data[:, 3]
                imu_acc = imu_data_ned[:, :3]
                imu_gyro = imu_data_ned[:, 3:]
                imu_acc_tensor = (imu_acc - self.imu_acc_mean) / self.imu_acc_std
                imu_gyro_tensor = (imu_gyro - self.imu_gyro_mean) / self.imu_gyro_std
                imu_sequence = torch.cat((imu_acc_tensor, imu_gyro_tensor), dim=1)
            if imu_sequence.size(0) != expected_len:
                print(f"Warning: IMU sequence length ({imu_sequence.size(0)}) does not match RGB ({expected_len}) at index {index}. Adjusting.")
                if imu_sequence.size(0) > expected_len:
                    imu_sequence = imu_sequence[:expected_len]
                else:
                    padding = torch.zeros((expected_len - imu_sequence.size(0), 6))
                    imu_sequence = torch.cat((imu_sequence, padding), dim=0)
        else:
            imu_sequence = torch.zeros((expected_len, 6))
        
        if par.enable_gps:
            gps_path, start_idx, end_idx, ratio_gps = gps_path_info if gps_path_info else (None, 0, 0, 1.0)
            if not gps_path or not os.path.exists(gps_path):
                gps_sequence = torch.zeros((expected_len, 6))
            else:
                gps_data = np.load(gps_path)
                adjusted_start = int(start_idx * ratio_gps)
                adjusted_end = int(end_idx * ratio_gps)
                gps_data = gps_data[adjusted_start:adjusted_end]
                gps_data = torch.tensor(gps_data, dtype=torch.float32)
                gps_data_ned = to_ned_pose(gps_data, is_absolute=True)
                gps_pos = gps_data_ned[:, :3]
                gps_vel = gps_data_ned[:, 3:]
                gps_pos_tensor = (gps_pos - self.gps_pos_mean) / self.gps_pos_std
                gps_vel_tensor = (gps_vel - self.gps_vel_mean) / self.gps_vel_std
                gps_sequence = torch.cat((gps_pos_tensor, gps_vel_tensor), dim=1)
            if gps_sequence.size(0) != expected_len:
                print(f"Warning: GPS sequence length ({gps_sequence.size(0)}) does not match RGB ({expected_len}) at index {index}. Adjusting.")
                if gps_sequence.size(0) > expected_len:
                    gps_sequence = gps_sequence[:expected_len]
                else:
                    padding = torch.zeros((expected_len - gps_sequence.size(0), 6))
                    gps_sequence = torch.cat((gps_sequence, padding), dim=0)
        else:
            gps_sequence = torch.zeros((expected_len, 6))
        
        assert image_sequence_03.size(0) == image_sequence_02.size(0) == depth_sequence.size(0) == imu_sequence.size(0) == gps_sequence.size(0)
        return (sequence_len, (image_sequence_03, image_sequence_02, depth_sequence, imu_sequence, gps_sequence), groundtruth_sequence)
    
    def __len__(self):
        return len(self.data_info.index)

if __name__ == '__main__':
    start_t = time.time()
    overlap = 1
    sample_times = 1
    climate_sets = ['PLE_training/spring']
    seq_len = 5
    train_df, valid_df = get_data_info(climate_sets, seq_len, overlap, sample_times)
    print(f'Elapsed Time (get_data_info): {time.time() - start_t:.2f} sec')