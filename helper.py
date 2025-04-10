import torch
import numpy as np
from scipy.spatial.transform import Rotation

def euler_to_rotation_matrix(roll, pitch, yaw):
    """Convert Euler angles (roll, pitch, yaw) to a rotation matrix."""
    return Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=False).as_matrix()

def rotation_matrix_to_euler(R):
    """Convert a rotation matrix to Euler angles (roll, pitch, yaw)."""
    return Rotation.from_matrix(R).as_euler('xyz', degrees=False)

def normalize_angle_delta(angle):
    """Normalize an angle to the range [-pi, pi]."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))

def to_ned_pose(pose, is_absolute=True):
    """
    Convert a pose to NED (North, East, Down) convention.
    pose: Tensor of shape [..., 6] (roll, pitch, yaw, x, y, z)
    is_absolute: If True, treat as absolute pose; if False, treat as relative pose.
    Returns: Tensor in NED convention.
    """
    pose_ned = pose.clone()
    # NED: X (North), Y (East), Z (Down)
    # Input: X (forward), Y (right), Z (up)
    # Translation: X -> Z, Y -> Y, Z -> -X
    pose_ned[..., 3] = pose[..., 5]  # Z -> X (North)
    pose_ned[..., 4] = pose[..., 4]  # Y -> Y (East)
    pose_ned[..., 5] = -pose[..., 3]  # -X -> Z (Down)
    return pose_ned

def from_ned_pose(pose, is_absolute=True):
    """
    Convert a pose from NED convention to the model's internal convention.
    pose: Tensor of shape [..., 6] (roll, pitch, yaw, x, y, z) in NED
    is_absolute: If True, treat as absolute pose; if False, treat as relative pose.
    Returns: Tensor in internal convention.
    """
    pose_internal = pose.clone()
    # NED: X (North), Y (East), Z (Down)
    # Internal: X (forward), Y (right), Z (up)
    # Translation: X -> -Z, Y -> Y, Z -> X
    pose_internal[..., 3] = -pose[..., 5]  # -Z -> X
    pose_internal[..., 4] = pose[..., 4]  # Y -> Y
    pose_internal[..., 5] = pose[..., 3]  # X -> Z
    return pose_internal

def integrate_relative_poses(relative_poses):
    """
    Integrate relative poses to absolute poses using SciPy for numerical stability.
    relative_poses: Tensor of shape [batch_size, seq_len, 6] (roll, pitch, yaw, x, y, z)
    Returns: Absolute poses in NED convention.
    """
    batch_size, seq_len, _ = relative_poses.size()
    absolute_poses = torch.zeros(batch_size, seq_len + 1, 6, device=relative_poses.device)
    
    for t in range(seq_len):
        rel_pose = relative_poses[:, t, :]  # [batch_size, 6]
        rel_angles = rel_pose[:, :3]  # [roll, pitch, yaw]
        rel_trans = rel_pose[:, 3:]  # [x, y, z]
        
        prev_pose = absolute_poses[:, t, :]  # [batch_size, 6]
        prev_angles = prev_pose[:, :3]
        prev_trans = prev_pose[:, 3:]
        
        # Convert previous angles to rotation matrix using SciPy
        R = torch.zeros(batch_size, 3, 3, device=relative_poses.device)
        for b in range(batch_size):
            R_np = Rotation.from_euler('xyz', prev_angles[b].cpu().numpy(), degrees=False).as_matrix()
            R[b] = torch.tensor(R_np, device=relative_poses.device, dtype=torch.float32)
        
        # Transform relative translation to world frame
        abs_trans = prev_trans + torch.matmul(R, rel_trans.unsqueeze(-1)).squeeze(-1)
        
        # Update angles
        abs_angles = prev_angles + rel_angles
        
        absolute_poses[:, t + 1, :3] = abs_angles
        absolute_poses[:, t + 1, 3:] = abs_trans
    
    return absolute_poses