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
from helper import normalize_angle_delta
from scipy.spatial.transform import Rotation

def euler_to_rotation_matrix(roll, pitch, yaw):
    """Convert Euler angles (roll, pitch, yaw) to a rotation matrix."""
    return Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=False).as_matrix()

def rotation_matrix_to_euler(R):
    """Convert a rotation matrix to Euler angles (roll, pitch, yaw)."""
    return Rotation.from_matrix(R).as_euler('xyz', degrees=False)

def get_data_info(climate_sets, seq_len_range, overlap, sample_times=1, pad_y=False, shuffle=False, sort=True, include_test=False):
    # Collect all trajectories
    all_traj_info = []
    for climate_set in climate_sets:
        traj_list = [d for d in os.listdir(f"/media/krkavinda/New Volume/Mid-Air-Dataset/MidAir_processed/{climate_set}/") if d.startswith("trajectory_")]
        traj_list.sort()
        print(f"Climate set {climate_set}: {len(traj_list)} total trajectories available")
        all_traj_info.extend([(climate_set, traj) for traj in traj_list])
    
    # Filter trajectories based on specified IDs
    train_traj_info = []
    valid_traj_info = []
    test_traj_info = []
    
    for climate_set in climate_sets:
        train_ids = set(par.train_traj_ids.get(climate_set, []))
        valid_ids = set(par.valid_traj_ids.get(climate_set, []))
        test_ids = set(par.test_traj_ids.get(climate_set, [])) if include_test else set()
        
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
    
    print(f"Total training trajectories: {len(train_traj_info)}, Total validation trajectories: {len(valid_traj_info)}, Total test trajectories: {len(test_traj_info)}")
    for climate_set in climate_sets:
        train_count = len([t for cs, t in train_traj_info if cs == climate_set])
        valid_count = len([t for cs, t in valid_traj_info if cs == climate_set])
        test_count = len([t for cs, t in test_traj_info if cs == climate_set])
        print(f"Climate set {climate_set}: Train={train_count}, Valid={valid_count}, Test={test_count}")

    # Verify no overlap
    train_set = set((cs, t) for cs, t in train_traj_info)
    valid_set = set((cs, t) for cs, t in valid_traj_info)
    test_set = set((cs, t) for cs, t in test_traj_info)
    if train_set & valid_set or (include_test and (train_set & test_set or valid_set & test_set)):
        print("Error: Overlap detected between sets!")
        print(f"Train-Valid overlap: {train_set & valid_set}")
        if include_test:
            print(f"Train-Test overlap: {train_set & test_set}")
            print(f"Valid-Test overlap: {valid_set & test_set}")
        raise ValueError("Trajectory overlap detected!")
    else:
        print("No overlap between training, validation, and test trajectories.")

    # Process trajectories
    def process_trajectories(traj_info, data_dict):
        for climate_set, traj in traj_info:
            start_t = time.time()
            traj_id = traj.split('_')[1]
            poses_path = f"/media/krkavinda/New Volume/Mid-Air-Dataset/MidAir_processed/{climate_set}/poses/poses_{traj_id}.npy"
            if not os.path.exists(poses_path):
                print(f"Poses file not found for {climate_set}/{traj}, skipping.")
                continue
            poses = np.load(poses_path)
            
            # Always load RGB images
            fpaths_03 = glob.glob(f'/media/krkavinda/New Volume/Mid-Air-Dataset/MidAir_processed/{climate_set}/{traj}/image_rgb/*.JPEG')
            fpaths_02 = glob.glob(f'/media/krkavinda/New Volume/Mid-Air-Dataset/MidAir_processed/{climate_set}/{traj}/image_rgb_right/*.JPEG')
            fpaths_03.sort()
            fpaths_02.sort()

            # Conditionally load depth, IMU, and GPS based on enable flags
            fpaths_depth = glob.glob(f'/media/krkavinda/New Volume/Mid-Air-Dataset/MidAir_processed/{climate_set}/{traj}/depth/*.PNG') if par.enable_depth else []
            fpaths_imu = f'/media/krkavinda/New Volume/Mid-Air-Dataset/MidAir_processed/{climate_set}/{traj}/imu.npy' if par.enable_imu else None
            fpaths_gps = f'/media/krkavinda/New Volume/Mid-Air-Dataset/MidAir_processed/{climate_set}/{traj}/gps.npy' if par.enable_gps else None
            if par.enable_depth:
                fpaths_depth.sort()

            print(f"Climate set {climate_set}, Trajectory {traj}:")
            print(f"  image_rgb: {len(fpaths_03)} files")
            print(f"  image_rgb_right: {len(fpaths_02)} files")
            if par.enable_depth:
                print(f"  depth: {len(fpaths_depth)} files")
            if par.enable_imu:
                print(f"  imu: {fpaths_imu}")
            if par.enable_gps:
                print(f"  gps: {fpaths_gps}")

            if len(fpaths_03) == 0 or len(fpaths_02) == 0:
                print(f"Skipping trajectory {traj} due to missing RGB files.")
                continue

            # Adjust length based on available modalities
            min_len = min(len(fpaths_03), len(fpaths_02))
            if par.enable_depth and len(fpaths_depth) > 0:
                min_len = min(min_len, len(fpaths_depth))
            min_len = min(min_len, len(poses))
            if min_len < len(fpaths_03):
                print(f"Warning: Trajectory {traj} has fewer frames. Truncating.")
                fpaths_03 = fpaths_03[:min_len]
                fpaths_02 = fpaths_02[:min_len]
                if par.enable_depth:
                    fpaths_depth = fpaths_depth[:min_len]
                poses = poses[:min_len]

            if seq_len_range[0] == seq_len_range[1]:
                start_frames = [0] if sample_times <= 1 else list(range(0, seq_len_range[0], int(np.ceil(seq_len_range[0] / sample_times))))
                if sample_times > 1:
                    print(f'Sample start from frame {start_frames}')
                
                for st in start_frames:
                    seq_len = seq_len_range[0]
                    n_frames = len(fpaths_03) - st
                    jump = seq_len - int(overlap)
                    num_sequences = (n_frames - seq_len) // jump + 1
                    print(f"Climate set {climate_set}, Trajectory {traj}: Generating {num_sequences} sequences")
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
                        x_seg_imu = (fpaths_imu, start_idx, end_idx) if par.enable_imu else None
                        x_seg_gps = (fpaths_gps, start_idx, end_idx) if par.enable_gps else None
                        y_seg = poses[start_idx:end_idx]
                        x_segs_03.append(x_seg_03)
                        x_segs_02.append(x_seg_02)
                        x_segs_depth.append(x_seg_depth)
                        x_segs_imu.append(x_seg_imu)
                        x_segs_gps.append(x_seg_gps)
                        y_segs.append(y_seg)
                    data_dict['Y'] += y_segs
                    data_dict['X_path_03'] += x_segs_03
                    data_dict['X_path_02'] += x_segs_02
                    data_dict['X_path_depth'] += x_segs_depth
                    data_dict['X_path_imu'] += x_segs_imu
                    data_dict['X_path_gps'] += x_segs_gps
                    data_dict['X_len'] += [len(xs) for xs in x_segs_03]
            else:
                assert int(overlap) < min(seq_len_range)
                n_frames = len(fpaths_03)
                min_len, max_len = seq_len_range[0], seq_len_range[1]
                for i in range(sample_times):
                    start = 0
                    while start + min_len <= n_frames:
                        n = np.random.randint(min_len, max_len + 1)
                        if start + n <= n_frames:
                            x_seg_03 = fpaths_03[start:start + n]
                            x_seg_02 = fpaths_02[start:start + n]
                            x_seg_depth = fpaths_depth[start:start + n] if par.enable_depth else []
                            x_seg_imu = (fpaths_imu, start, start + n) if par.enable_imu else None
                            x_seg_gps = (fpaths_gps, start, start + n) if par.enable_gps else None
                            data_dict['Y'].append(poses[start:start + n])
                            data_dict['X_path_03'].append(x_seg_03)
                            data_dict['X_path_02'].append(x_seg_02)
                            data_dict['X_path_depth'].append(x_seg_depth)
                            data_dict['X_path_imu'].append(x_seg_imu)
                            data_dict['X_path_gps'].append(x_seg_gps)
                            data_dict['X_len'].append(len(x_seg_03))
                        else:
                            print(f'Last {start + n - n_frames} frames not used')
                            break
                        start += n - int(overlap)
            print(f'Climate set {climate_set}, Trajectory {traj} finished in {time.time() - start_t} sec')

    # Initialize data dictionaries
    train_data = {'X_path_03': [], 'X_path_02': [], 'X_path_depth': [], 'X_path_imu': [], 'X_path_gps': [], 'Y': [], 'X_len': []}
    valid_data = {'X_path_03': [], 'X_path_02': [], 'X_path_depth': [], 'X_path_imu': [], 'X_path_gps': [], 'Y': [], 'X_len': []}
    test_data = {'X_path_03': [], 'X_path_02': [], 'X_path_depth': [], 'X_path_imu': [], 'X_path_gps': [], 'Y': [], 'X_len': []}

    # Process each split
    process_trajectories(train_traj_info, train_data)
    process_trajectories(valid_traj_info, valid_data)
    if include_test:
        process_trajectories(test_traj_info, test_data)

    # Create dataframes
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

def get_partition_data_info(partition, climate_sets, seq_len_range, overlap, sample_times=1, pad_y=False, shuffle=False, sort=True):
    # This function is no longer needed since get_data_info returns both train and valid dataframes
    return get_data_info(climate_sets, seq_len_range, overlap, sample_times, pad_y, shuffle, sort)

class SortedRandomBatchSampler(Sampler):
    def __init__(self, info_dataframe, batch_size, drop_last=False):
        self.df = info_dataframe
        self.batch_size = batch_size
        self.drop_last = False  # Changed to False to avoid dropping sequences
        self.unique_seq_lens = sorted(self.df.iloc[:].seq_len.unique(), reverse=True)
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
            tmp = [rand_idxs[s * self.batch_size: s * self.batch_size + self.batch_size] for s in range(0, n_batch)]
            list_batch_indexes += tmp
            start_idx += n_sample
        return iter(list_batch_indexes)

    def __len__(self):
        return self.len

class ImageSequenceDataset(Dataset):
    def __init__(self, info_dataframe, resize_mode='crop', new_size=None, img_means_03=None, img_stds_03=(1, 1, 1), 
                 img_means_02=None, img_stds_02=(1, 1, 1), minus_point_5=False):
        # Define path for saving/loading statistics
        stats_pickle_path = "datainfo/dataset_stats.pickle"
        
        # Define depth_max (needed for normalization)
        self.depth_max = 100.0
        
        # Check if statistics file exists
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
            self.depth_max = stats['depth_max']  # Load depth_max from pickle
        else:
            print("Computing dataset statistics...")
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
            
            # Depth normalization (only if enabled)
            if par.enable_depth:
                depth_values = []
                for depth_path_seq in info_dataframe.depth_path[:100]:
                    for depth_path in depth_path_seq:
                        if os.path.exists(depth_path):
                            depth_map = np.array(Image.open(depth_path)).astype(np.float32) / 1000.0  # Mid-Air depth in meters
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

            # IMU and GPS normalization (store mean and std for manual normalization)
            if par.enable_imu:
                imu_values_acc = []
                imu_values_gyro = []
                for imu_path_info in info_dataframe.imu_path[:100]:
                    imu_path, start_idx, end_idx = imu_path_info
                    if imu_path and os.path.exists(imu_path):
                        imu_data = np.load(imu_path)[start_idx:end_idx]  # [ax, ay, az, wx, wy, wz]
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
                for gps_path_info in info_dataframe.gps_path[:100]:
                    gps_path, start_idx, end_idx = gps_path_info
                    if gps_path and os.path.exists(gps_path):
                        gps_data = np.load(gps_path)[start_idx:end_idx]  # [x, y, z, vx, vy, vz]
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

            # Save the computed statistics to a pickle file
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

        # Initialize the rest of the dataset
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
        raw_groundtruth = self.groundtruth_arr[index]  # Shape: (seq_len, 6) [roll, pitch, yaw, x, y, z]
        groundtruth_sequence = torch.FloatTensor(raw_groundtruth)

        # Debug: Log absolute pose stats for first 5 samples
        if index < 5:
            angles = groundtruth_sequence[:, :3]
            translations = groundtruth_sequence[:, 3:]
            print(f"Sample {index} - Absolute Angles mean: {angles.mean().item():.4f}, std: {angles.std().item():.4f}")
            print(f"Sample {index} - Absolute Translations mean: {translations.mean().item():.4f}, std: {translations.std().item():.4f}")

        # Convert absolute poses to relative poses
        relative_poses = []
        for i in range(groundtruth_sequence.shape[0]):
            if i == 0:
                # First frame: relative pose is zero
                relative_pose = torch.zeros(6, dtype=torch.float32)
            else:
                # Compute frame-to-frame relative pose
                delta_pose = groundtruth_sequence[i] - groundtruth_sequence[i-1]
                roll_rel, pitch_rel, yaw_rel = delta_pose[:3]
                t_rel = delta_pose[3:]

                # Normalize angles
                roll_rel = normalize_angle_delta(roll_rel)
                pitch_rel = normalize_angle_delta(pitch_rel)
                yaw_rel = normalize_angle_delta(yaw_rel)

                # Compute rotation matrix of previous frame to transform translation
                roll_prev, pitch_prev, yaw_prev = groundtruth_sequence[i-1, :3]
                R_prev = euler_to_rotation_matrix(roll_prev, pitch_prev, yaw_prev)
                # Transform translation to the previous frame's coordinate system
                t_rel_transformed = torch.FloatTensor(R_prev.T @ t_rel.numpy())

                relative_pose = torch.tensor([roll_rel, pitch_rel, yaw_rel, t_rel_transformed[0], t_rel_transformed[1], t_rel_transformed[2]], dtype=torch.float32)
            relative_poses.append(relative_pose)

        groundtruth_sequence = torch.stack(relative_poses)  # Shape: (seq_len, 6)

        # Debug: Log relative pose stats for first 5 samples
        if index < 5:
            angles = groundtruth_sequence[:, :3]
            translations = groundtruth_sequence[:, 3:]
            print(f"Sample {index} - Relative Angles mean: {angles.mean().item():.4f}, std: {angles.std().item():.4f}")
            print(f"Sample {index} - Relative Translations mean: {translations.mean().item():.4f}, std: {translations.std().item():.4f}")

        # Cross-check with GPS data (if enabled)
        if par.enable_gps:
            gps_path, start_idx, end_idx = self.gps_arr[index]
            if gps_path and os.path.exists(gps_path):
                gps_data = np.load(gps_path)[start_idx:end_idx]  # [x, y, z, vx, vy, vz]
                gps_pos = torch.FloatTensor(gps_data[:, :3])  # [x, y, z]
                # Compute frame-to-frame relative positions from GPS
                gps_relative_pos = gps_pos[1:] - gps_pos[:-1]
                gps_relative_pos = torch.cat([torch.zeros(1, 3), gps_relative_pos], dim=0)  # Pad first frame with zeros
                # Compare with ground truth translations
                gt_translations = groundtruth_sequence[:, 3:]
                if index < 5:
                    print(f"Sample {index} - GPS Relative Positions mean: {gps_relative_pos.mean().item():.4f}, std: {gps_relative_pos.std().item():.4f}")
                    print(f"Sample {index} - GT vs GPS Translation Diff mean: {(gt_translations - gps_relative_pos).abs().mean().item():.4f}")

        image_path_sequence_03 = self.image_arr_03[index]
        image_path_sequence_02 = self.image_arr_02[index]
        depth_path_sequence = self.depth_arr[index]
        imu_path_info = self.imu_arr[index]
        gps_path_info = self.gps_arr[index]
        sequence_len = torch.tensor(self.seq_len_list[index])
        expected_len = len(image_path_sequence_03)
        
        # Load and process image_03
        image_sequence_03 = [self.normalizer_03(self.transformer(Image.open(img_path)) - (0.5 if self.minus_point_5 else 0)).unsqueeze(0) 
                             for img_path in image_path_sequence_03]
        image_sequence_03 = torch.cat(image_sequence_03, 0)
        
        # Load and process image_02
        image_sequence_02 = [self.normalizer_02(self.transformer(Image.open(img_path)) - (0.5 if self.minus_point_5 else 0)).unsqueeze(0) 
                             for img_path in image_path_sequence_02]
        image_sequence_02 = torch.cat(image_sequence_02, 0)
        
        # Load and process depth (if enabled)
        if par.enable_depth:
            depth_sequence = []
            if len(depth_path_sequence) != expected_len:
                print(f"Warning: Depth sequence length ({len(depth_path_sequence)}) does not match RGB ({expected_len}) at index {index}. Adjusting.")
                depth_path_sequence = list(depth_path_sequence)
                if len(depth_path_sequence) > expected_len:
                    depth_path_sequence = depth_path_sequence[:expected_len]
                else:
                    depth_path_sequence.extend([depth_path_sequence[-1] if depth_path_sequence else 'dummy'] * (expected_len - len(depth_path_sequence)))
            
            for depth_path in depth_path_sequence:
                if not os.path.exists(depth_path) or depth_path == 'dummy':
                    depth_as_tensor = torch.zeros((1, par.img_h, par.img_w))
                else:
                    depth_map = np.array(Image.open(depth_path)).astype(np.float32) / 1000.0  # Mid-Air depth in meters
                    depth_as_tensor = torch.from_numpy(depth_map).float() / self.depth_max
                    depth_as_tensor = transforms.Resize((par.img_h, par.img_w))(depth_as_tensor.unsqueeze(0))
                    depth_as_tensor = self.normalizer_depth(depth_as_tensor)
                depth_sequence.append(depth_as_tensor.unsqueeze(0))
            depth_sequence = torch.cat(depth_sequence, 0)
        else:
            depth_sequence = torch.zeros((expected_len, 1, par.img_h, par.img_w))

        # Load and process IMU (if enabled)
        if par.enable_imu:
            imu_path, start_idx, end_idx = imu_path_info if imu_path_info else (None, 0, 0)
            if not imu_path or not os.path.exists(imu_path):
                imu_sequence = torch.zeros((expected_len, 6))
            else:
                imu_data = np.load(imu_path)[start_idx:end_idx]  # [ax, ay, az, wx, wy, wz]
                imu_acc = imu_data[:, :3]
                imu_gyro = imu_data[:, 3:]
                imu_acc_tensor = torch.from_numpy(imu_acc).float()
                imu_gyro_tensor = torch.from_numpy(imu_gyro).float()
                imu_acc_tensor = (imu_acc_tensor - self.imu_acc_mean) / self.imu_acc_std
                imu_gyro_tensor = (imu_gyro_tensor - self.imu_gyro_mean) / self.imu_gyro_std
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

        # Load and process GPS (if enabled)
        if par.enable_gps:
            gps_path, start_idx, end_idx = gps_path_info if gps_path_info else (None, 0, 0)
            if not gps_path or not os.path.exists(gps_path):
                gps_sequence = torch.zeros((expected_len, 6))
            else:
                gps_data = np.load(gps_path)[start_idx:end_idx]  # [x, y, z, vx, vy, vz]
                gps_pos = gps_data[:, :3]
                gps_vel = gps_data[:, 3:]
                gps_pos_tensor = torch.from_numpy(gps_pos).float()
                gps_vel_tensor = torch.from_numpy(gps_vel).float()
                gps_pos_tensor = (gps_pos_tensor - self.gps_pos_mean) / self.gps_pos_std
                gps_vel_tensor = (gps_vel_tensor - self.gps_vel_mean) / self.gps_vel_std
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
    climate_sets = ['Kite_training/cloudy']
    seq_len_range = (5, 7)
    train_df, valid_df = get_data_info(climate_sets, seq_len_range, overlap, sample_times)
    print(f'Elapsed Time (get_data_info): {time.time() - start_t} sec')
    n_workers = 4
    resize_mode = 'crop'
    new_size = (150, 600)
    img_means_03 = (-0.14968217427134656, -0.12941663107068363, -0.1320610301921484)
    img_stds_03 = (1, 1, 1)
    img_means_02 = (-0.14968217427134656, -0.12941663107068363, -0.1320610301921484)
    img_stds_02 = (1, 1, 1)
    train_dataset = ImageSequenceDataset(train_df, resize_mode, new_size, img_means_03, img_stds_03, img_means_02, img_stds_02)
    valid_dataset = ImageSequenceDataset(valid_df, resize_mode, new_size, img_means_03, img_stds_03, img_means_02, img_stds_02)
    train_sampler = SortedRandomBatchSampler(train_df, batch_size=4, drop_last=False)
    valid_sampler = SortedRandomBatchSampler(valid_df, batch_size=4, drop_last=False)
    train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=n_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_sampler=valid_sampler, num_workers=n_workers)
    print(f'Elapsed Time (dataloader): {time.time() - start_t} sec')

    for batch in train_dataloader:
        s, (x_03, x_02, x_depth, x_imu, x_gps), y = batch
    
    print(f'Elapsed Time: {time.time() - start_t} sec')
    print('Number of workers = ', n_workers)