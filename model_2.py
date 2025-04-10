import torch
import torch.nn as nn
import torch.nn.functional as F
from params import par

# Define the Adaptive Gated Attention Fusion Module
class AdaptiveGatedAttentionFusion(nn.Module):
    def __init__(self, rgb_channels, depth_channels, lidar_channels, depth_gate_scaling=20.0, imu_gate_scaling=15.0):
        super(AdaptiveGatedAttentionFusion, self).__init__()
        self.rgb_channels = rgb_channels
        self.depth_channels = depth_channels
        self.lidar_channels = lidar_channels
        self.total_channels = rgb_channels + depth_channels + lidar_channels

        # Attention gates for each modality
        self.rgb_gate = nn.Linear(self.total_channels, rgb_channels)
        self.depth_gate = nn.Linear(self.total_channels, depth_channels)
        self.lidar_gate = nn.Linear(self.total_channels, lidar_channels)

        # Scaling factors for depth and IMU (lidar) gates
        self.depth_gate_scaling = depth_gate_scaling  # Increased to prioritize depth features
        self.imu_gate_scaling = imu_gate_scaling  # Increased to prioritize IMU features

        # Final fusion layer
        self.fusion_layer = nn.Linear(self.total_channels, self.total_channels)

    def forward(self, rgb_features, depth_features, lidar_features):
        # Concatenate features
        combined = torch.cat((rgb_features, depth_features, lidar_features), dim=-1)

        # Compute attention gates
        rgb_attention = torch.sigmoid(self.rgb_gate(combined))
        depth_attention = torch.sigmoid(self.depth_gate(combined)) * self.depth_gate_scaling
        lidar_attention = torch.sigmoid(self.lidar_gate(combined)) * self.imu_gate_scaling

        # Apply attention to each modality
        rgb_weighted = rgb_features * rgb_attention
        depth_weighted = depth_features * depth_attention
        lidar_weighted = lidar_features * lidar_attention

        # Concatenate weighted features
        fused = torch.cat((rgb_weighted, depth_weighted, lidar_weighted), dim=-1)

        # Final fusion
        fused = self.fusion_layer(fused)
        return fused

# Define the Stereo Adaptive Visual Odometry Model
class StereoAdaptiveVO(nn.Module):
    def __init__(self, img_h, img_w, batch_norm, input_channels=3, hidden_size=512, num_layers=2):
        super(StereoAdaptiveVO, self).__init__()
        self.img_h = img_h
        self.img_w = img_w
        self.batch_norm = batch_norm
        self.hidden_size = hidden_size
        self.num_layers = num_layers  # Ensure num_layers is an integer

        # RGB feature extraction (for both left and right images)
        self.rgb_conv = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),  # Input channels for a single image
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Adjust the size based on img_h and img_w (assuming input size is img_h x img_w)
        # After three max pooling layers (stride 2), the size is reduced by a factor of 8
        self.rgb_feature_size = (img_h // 8) * (img_w // 8) * 256
        self.rgb_fc = nn.Linear(self.rgb_feature_size, 256)  # Output 256 features per image

        # Depth feature extraction
        self.depth_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.depth_feature_size = (img_h // 8) * (img_w // 8) * 128
        self.depth_fc = nn.Linear(self.depth_feature_size, 256)

        # IMU feature extraction
        self.imu_fc = nn.Sequential(
            nn.Linear(6, 64),  # IMU: [ax, ay, az, wx, wy, wz]
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

        # GPS feature extraction
        self.gps_fc = nn.Sequential(
            nn.Linear(6, 64),  # GPS: [x, y, z, vx, vy, vz]
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

        # Fusion module with increased depth and IMU scaling
        # RGB features are now 512 (256 from left + 256 from right)
        self.fusion_module = AdaptiveGatedAttentionFusion(
            rgb_channels=512,  # 256 (left) + 256 (right)
            depth_channels=256,
            lidar_channels=128 + 128,  # IMU + GPS
            depth_gate_scaling=20.0,  # Increased to prioritize depth
            imu_gate_scaling=15.0  # Increased to prioritize IMU
        )

        # RNN for temporal modeling
        self.rnn = nn.LSTM(
            input_size=512 + 256 + 128 + 128,  # RGB (left + right) + Depth + IMU + GPS
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Dropout for regularization
        self.rnn_drop_out = nn.Dropout(0.5)

        # Output layer for 6-DoF pose (roll, pitch, yaw, x, y, z)
        self.linear = nn.Linear(hidden_size, 6)

        # Loss weights
        self.k_factor = 0.5  # Reduced to balance angle and translation loss
        self.translation_loss_weight = 2.0  # Increased to prioritize translation
        self.depth_consistency_loss_weight = 10.0  # Increased to prioritize depth consistency
        self.gps_loss_weight = 5.0  # Increased to prioritize GPS data

    def forward(self, x):
        x_03, x_02, x_depth, x_imu, x_gps = x

        # Process left RGB image (x_03)
        batch_size, seq_len, C, H, W = x_03.shape  # Shape: [batch_size, seq_len, C, H, W]
        x_03 = x_03.view(batch_size * seq_len, C, H, W)  # Flatten batch and sequence dimensions
        left_features = self.rgb_conv(x_03)
        left_features = left_features.view(batch_size * seq_len, -1)
        left_features = self.rgb_fc(left_features)
        left_features = left_features.view(batch_size, seq_len, -1)  # Shape: [batch_size, seq_len, 256]

        # Process right RGB image (x_02)
        batch_size, seq_len, C, H, W = x_02.shape  # Shape: [batch_size, seq_len, C, H, W]
        x_02 = x_02.view(batch_size * seq_len, C, H, W)  # Flatten batch and sequence dimensions
        right_features = self.rgb_conv(x_02)
        right_features = right_features.view(batch_size * seq_len, -1)
        right_features = self.rgb_fc(right_features)
        right_features = right_features.view(batch_size, seq_len, -1)  # Shape: [batch_size, seq_len, 256]

        # Concatenate left and right features
        rgb_features = torch.cat((left_features, right_features), dim=-1)  # Shape: [batch_size, seq_len, 512]

        # Process Depth
        batch_size, seq_len, C_depth, H_depth, W_depth = x_depth.shape  # Shape: [batch_size, seq_len, 1, H, W]
        x_depth = x_depth.view(batch_size * seq_len, C_depth, H_depth, W_depth)
        depth_features = self.depth_conv(x_depth)
        depth_features = depth_features.view(batch_size * seq_len, -1)
        depth_features = self.depth_fc(depth_features)
        depth_features = depth_features.view(batch_size, seq_len, -1)  # Shape: [batch_size, seq_len, 256]

        # Process IMU
        batch_size, seq_len, _ = x_imu.shape  # Shape: [batch_size, seq_len, 6]
        x_imu = x_imu.view(batch_size * seq_len, -1)
        imu_features = self.imu_fc(x_imu)
        imu_features = imu_features.view(batch_size, seq_len, -1)  # Shape: [batch_size, seq_len, 128]

        # Process GPS
        batch_size, seq_len, _ = x_gps.shape  # Shape: [batch_size, seq_len, 6]
        x_gps = x_gps.view(batch_size * seq_len, -1)
        gps_features = self.gps_fc(x_gps)
        gps_features = gps_features.view(batch_size, seq_len, -1)  # Shape: [batch_size, seq_len, 128]

        # Concatenate IMU and GPS features
        lidar_features = torch.cat((imu_features, gps_features), dim=-1)  # Shape: [batch_size, seq_len, 256]

        # Fuse features using adaptive gated attention
        combined_features = torch.zeros(batch_size, seq_len, self.fusion_module.total_channels).to(rgb_features.device)
        for t in range(seq_len):
            fused = self.fusion_module(
                rgb_features[:, t, :],
                depth_features[:, t, :],
                lidar_features[:, t, :]
            )
            combined_features[:, t, :] = fused

        # Pass through RNN
        out, hc = self.rnn(combined_features)
        out = self.rnn_drop_out(out)
        out = self.linear(out)  # Shape: [batch_size, seq_len, 6]

        # Remap predicted poses to correct NED convention
        out_remapped = out.clone()
        out_remapped[:, :, 3] = out[:, :, 5]  # Z -> X (North)
        out_remapped[:, :, 4] = out[:, :, 4]  # Y -> Y (East)
        out_remapped[:, :, 5] = out[:, :, 3]  # X -> Z (Down)
        out_remapped[:, :, 3] *= -1  # Flip X (North) to correct mirroring
        out_remapped[:, :, 4] *= -1  # Keep sign flip for Y (East)
        
        return out_remapped

    def compute_absolute_poses(self, relative_poses):
        batch_size, seq_len, _ = relative_poses.size()
        absolute_poses = torch.zeros(batch_size, seq_len + 1, 6, device=relative_poses.device)
        
        for t in range(seq_len):
            rel_pose = relative_poses[:, t, :]  # [batch_size, 6]
            rel_angles = rel_pose[:, :3]  # [roll, pitch, yaw]
            rel_trans = rel_pose[:, 3:]  # [x, y, z]
            
            prev_pose = absolute_poses[:, t, :]  # [batch_size, 6]
            prev_angles = prev_pose[:, :3]
            prev_trans = prev_pose[:, 3:]
            
            cos_roll = torch.cos(prev_angles[:, 0])
            sin_roll = torch.sin(prev_angles[:, 0])
            cos_pitch = torch.cos(prev_angles[:, 1])
            sin_pitch = torch.sin(prev_angles[:, 1])
            cos_yaw = torch.cos(prev_angles[:, 2])
            sin_yaw = torch.sin(prev_angles[:, 2])
            
            R = torch.zeros(batch_size, 3, 3, device=relative_poses.device)
            R[:, 0, 0] = cos_yaw * cos_pitch
            R[:, 0, 1] = cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll
            R[:, 0, 2] = cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll
            R[:, 1, 0] = sin_yaw * cos_pitch
            R[:, 1, 1] = sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll
            R[:, 1, 2] = sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll
            R[:, 2, 0] = -sin_pitch
            R[:, 2, 1] = cos_pitch * sin_roll
            R[:, 2, 2] = cos_pitch * cos_roll
            
            abs_trans = prev_trans + torch.matmul(R, rel_trans.unsqueeze(-1)).squeeze(-1)
            
            abs_angles = prev_angles + rel_angles
            
            absolute_poses[:, t + 1, :3] = abs_angles
            absolute_poses[:, t + 1, 3:] = abs_trans
        
        return absolute_poses

    def get_loss(self, x, y):
        predicted = self.forward(x)  # Shape: [batch_size, seq_len, 6]
        y = y[:, 1:, :]  # Shape: [batch_size, seq_len-1, 6] (relative poses between frames)
        # Align predicted tensor with target by removing the first prediction
        predicted = predicted[:, 1:, :]  # Shape: [batch_size, seq_len-1, 6]
        angle_loss = torch.nn.functional.mse_loss(predicted[:, :, :3], y[:, :, :3])
        translation_loss = torch.nn.functional.mse_loss(predicted[:, :, 3:], y[:, :, 3:])
        l2_lambda = par.l2_lambda
        l2_loss = l2_lambda * sum(torch.norm(param) for param in self.parameters() if param.requires_grad)
        base_loss = self.k_factor * angle_loss + self.translation_loss_weight * translation_loss + l2_loss

        # Placeholder for GPS loss (assuming it's computed elsewhere)
        gps_loss = torch.tensor(0.0, device=predicted.device)  # Replace with actual GPS loss computation
        total_loss = base_loss + self.gps_loss_weight * gps_loss

        return total_loss

    def step(self, x, y, optimizer, scaler=None):
        """
        Perform a single training step: forward pass, compute loss, backpropagate, and update weights.
        
        Args:
            x (tuple): Input tuple (x_03, x_02, x_depth, x_imu, x_gps).
            y (torch.Tensor): Ground truth poses [batch_size, seq_len, 6].
            optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
            scaler (torch.amp.GradScaler, optional): Gradient scaler for mixed precision training.
        
        Returns:
            float: Loss value for the step.
        """
        optimizer.zero_grad()  # Clear accumulated gradients

        # Forward pass
        if scaler is not None:
            with torch.amp.autocast(device_type='cuda', enabled=True):  # Updated autocast syntax
                loss = self.get_loss(x, y)
        else:
            loss = self.get_loss(x, y)

        # Backward pass and optimization
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        return loss.item()

# Example instantiation (for reference)
if __name__ == "__main__":
    model = StereoAdaptiveVO(img_h=300, img_w=400, batch_norm=True, input_channels=3, hidden_size=512, num_layers=2)
    print(model)