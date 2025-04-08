import numpy as np

# Path to the ground truth file
pose_gt_file = '/media/krkavinda/New Volume/Mid-Air-Dataset/MidAir_processed/Kite_training/sunny/poses/poses_0008.npy'

# Load the ground truth poses
gt_poses = np.load(pose_gt_file)

# Print basic information
print(f"Ground truth shape: {gt_poses.shape}")
print(f"First 5 ground truth poses:\n{gt_poses[:5]}")
print(f"Last 5 ground truth poses:\n{gt_poses[-5:]}")

# Compute the range of x, y, z coordinates
x = gt_poses[:, 3]
y = gt_poses[:, 4]
z = gt_poses[:, 5]
print(f"X range: ({x.min()}, {x.max()})")
print(f"Y range: ({y.min()}, {y.max()})")
print(f"Z range: ({z.min()}, {z.max()})")