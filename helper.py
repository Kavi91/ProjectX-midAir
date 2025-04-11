import torch
import numpy as np
from scipy.spatial.transform import Rotation

def euler_to_rotation_matrix(roll, pitch, yaw):
    """Convert Euler angles to a rotation matrix."""
    return Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=False).as_matrix()

def rotation_matrix_to_euler(R):
    """Convert a rotation matrix to Euler angles."""
    return Rotation.from_matrix(R).as_euler('xyz', degrees=False)

def normalize_angle_delta(angle):
    """Normalize an angle to the range [-pi, pi]."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))

def to_ned_pose(pose, is_absolute=True, debug=False):
    """
    Convert a pose (roll, pitch, yaw, x, y, z) from the internal coordinate system
    (X: forward, Y: right, Z: up) to NED (North, East, Down) convention.
    
    The conversion applied is as follows:
      - New X (North) = Original Z
      - New Y (East) = Original Y
      - New Z (Down)  = - Original X
      
    If debug=True, prints the input and output tensors.
    """
    pose_ned = pose.clone()
    pose_ned[..., 3] = pose[..., 5]   # Z -> North
    pose_ned[..., 4] = pose[..., 4]   # Y -> East
    pose_ned[..., 5] = -pose[..., 3]  # -X -> Down

    # if debug:
    #     print("DEBUG to_ned_pose:")
    #     print("Input pose:")
    #     print(pose)
    #     print("Converted pose (NED):")
    #     print(pose_ned)
    return pose_ned

def from_ned_pose(pose, is_absolute=True):
    """Convert a pose from NED back to the internal coordinate system."""
    pose_internal = pose.clone()
    pose_internal[..., 3] = -pose[..., 5]  # -Z -> X
    pose_internal[..., 4] = pose[..., 4]     # Y remains the same
    pose_internal[..., 5] = pose[..., 3]     # X -> Z
    return pose_internal

def integrate_relative_poses(relative_poses):
    """
    Integrate relative poses to absolute poses (in NED convention).

    Args:
        relative_poses: Tensor of shape [batch_size, seq_len, 6] (roll, pitch, yaw, x, y, z)
    
    Returns:
        absolute_poses: Tensor of shape [batch_size, seq_len+1, 6]
    """
    batch_size, seq_len, _ = relative_poses.size()
    absolute_poses = torch.zeros(batch_size, seq_len + 1, 6, device=relative_poses.device)
    
    for t in range(seq_len):
        rel_pose = relative_poses[:, t, :]
        rel_angles = rel_pose[:, :3]
        rel_trans = rel_pose[:, 3:]
        
        prev_pose = absolute_poses[:, t, :]
        prev_angles = prev_pose[:, :3]
        prev_trans = prev_pose[:, 3:]
        
        R = torch.zeros(batch_size, 3, 3, device=relative_poses.device)
        for b in range(batch_size):
            R_np = Rotation.from_euler('xyz', prev_angles[b].cpu().numpy(), degrees=False).as_matrix()
            R[b] = torch.tensor(R_np, device=relative_poses.device, dtype=torch.float32)
        
        abs_trans = prev_trans + torch.matmul(R, rel_trans.unsqueeze(-1)).squeeze(-1)
        abs_angles = prev_angles + rel_angles
        
        absolute_poses[:, t + 1, :3] = abs_angles
        absolute_poses[:, t + 1, 3:] = abs_trans
    
    return absolute_poses