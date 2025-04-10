import torch
import torch.nn as nn
import torch.nn.functional as F
from params import par
from helper import to_ned_pose, integrate_relative_poses

# Adaptive Gated Attention Fusion Module for multi-modal features.
class AdaptiveGatedAttentionFusion(nn.Module):
    def __init__(self, rgb_channels, depth_channels, lidar_channels, depth_gate_scaling=par.depth_gate_scaling, imu_gate_scaling=par.imu_gate_scaling):
        super(AdaptiveGatedAttentionFusion, self).__init__()
        self.rgb_channels = rgb_channels
        self.depth_channels = depth_channels
        self.lidar_channels = lidar_channels
        self.total_channels = rgb_channels + depth_channels + lidar_channels

        # Define attention gates.
        self.rgb_gate = nn.Linear(self.total_channels, rgb_channels)
        self.depth_gate = nn.Linear(self.total_channels, depth_channels)
        self.lidar_gate = nn.Linear(self.total_channels, lidar_channels)

        self.depth_gate_scaling = depth_gate_scaling
        self.imu_gate_scaling = imu_gate_scaling

        # Fusion layer combining weighted modalities.
        self.fusion_layer = nn.Linear(self.total_channels, self.total_channels)

    def forward(self, rgb_features, depth_features, lidar_features):
        combined = torch.cat((rgb_features, depth_features, lidar_features), dim=-1)
        rgb_attention = torch.sigmoid(self.rgb_gate(combined))
        depth_attention = torch.sigmoid(self.depth_gate(combined)) * self.depth_gate_scaling
        lidar_attention = torch.sigmoid(self.lidar_gate(combined)) * self.imu_gate_scaling
        rgb_weighted = rgb_features * rgb_attention
        depth_weighted = depth_features * depth_attention
        lidar_weighted = lidar_features * lidar_attention
        fused = torch.cat((rgb_weighted, depth_weighted, lidar_weighted), dim=-1)
        fused = self.fusion_layer(fused)
        return fused

# Stereo Adaptive Visual Odometry Model
class StereoAdaptiveVO(nn.Module):
    def __init__(self, img_h, img_w, batch_norm, input_channels=3, hidden_size=512, num_layers=2):
        super(StereoAdaptiveVO, self).__init__()
        self.img_h = img_h
        self.img_w = img_w
        self.batch_norm = batch_norm
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # RGB feature extraction for each view.
        self.rgb_conv = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),  # Reduced dropout from 0.3 to 0.2
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.rgb_feature_size = (img_h // 8) * (img_w // 8) * 256
        self.rgb_fc = nn.Linear(self.rgb_feature_size, 256)

        # Depth feature extraction (if enabled).
        if par.enable_depth:
            self.depth_conv = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.depth_feature_size = (img_h // 8) * (img_w // 8) * 128
            self.depth_fc = nn.Linear(self.depth_feature_size, 256)
        else:
            self.depth_fc = nn.Identity()

        # IMU feature extraction (if enabled).
        if par.enable_imu:
            self.imu_fc = nn.Sequential(
                nn.Linear(6, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU()
            )
        else:
            self.imu_fc = nn.Identity()

        # GPS feature extraction (if enabled).
        if par.enable_gps:
            self.gps_fc = nn.Sequential(
                nn.Linear(6, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU()
            )
        else:
            self.gps_fc = nn.Identity()

        # Fusion module to combine modalities.
        self.fusion_module = AdaptiveGatedAttentionFusion(
            rgb_channels=512,
            depth_channels=256 if par.enable_depth else 0,
            lidar_channels=(128 if par.enable_imu else 0) + (128 if par.enable_gps else 0)
        )

        # RNN for temporal modeling.
        self.rnn = nn.LSTM(
            input_size=512 + (256 if par.enable_depth else 0) + (128 if par.enable_imu else 0) + (128 if par.enable_gps else 0),
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.rnn_drop_out = nn.Dropout(par.rnn_dropout_out)

        # Output linear layer for 6-DoF pose.
        self.linear = nn.Linear(hidden_size, 6)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for layers (including those not updated by pretrained weights)."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def forward(self, x):
        x_03, x_02, x_depth, x_imu, x_gps = x

        # Process left RGB image.
        B, seq_len, C, H, W = x_03.shape
        x_03 = x_03.view(B * seq_len, C, H, W)
        left_features = self.rgb_conv(x_03)
        left_features = left_features.view(B * seq_len, -1)
        left_features = self.rgb_fc(left_features)
        left_features = left_features.view(B, seq_len, -1)

        # Process right RGB image.
        B, seq_len, C, H, W = x_02.shape
        x_02 = x_02.view(B * seq_len, C, H, W)
        right_features = self.rgb_conv(x_02)
        right_features = right_features.view(B * seq_len, -1)
        right_features = self.rgb_fc(right_features)
        right_features = right_features.view(B, seq_len, -1)

        # Concatenate left and right features.
        rgb_features = torch.cat((left_features, right_features), dim=-1)

        # Process Depth if enabled.
        if par.enable_depth:
            B, seq_len, C_depth, H_depth, W_depth = x_depth.shape
            x_depth = x_depth.view(B * seq_len, C_depth, H_depth, W_depth)
            depth_features = self.depth_conv(x_depth)
            depth_features = depth_features.view(B * seq_len, -1)
            depth_features = self.depth_fc(depth_features)
            depth_features = depth_features.view(B, seq_len, -1)
        else:
            depth_features = torch.zeros(B, seq_len, 0, device=x_03.device)

        # Process IMU if enabled.
        if par.enable_imu:
            B, seq_len, _ = x_imu.shape
            x_imu = x_imu.view(B * seq_len, -1)
            imu_features = self.imu_fc(x_imu)
            imu_features = imu_features.view(B, seq_len, -1)
        else:
            imu_features = torch.zeros(B, seq_len, 0, device=x_03.device)

        # Process GPS if enabled.
        if par.enable_gps:
            B, seq_len, _ = x_gps.shape
            x_gps = x_gps.view(B * seq_len, -1)
            gps_features = self.gps_fc(x_gps)
            gps_features = gps_features.view(B, seq_len, -1)
        else:
            gps_features = torch.zeros(B, seq_len, 0, device=x_03.device)

        # Combine IMU and GPS features.
        lidar_features = torch.cat((imu_features, gps_features), dim=-1)

        # Fuse features using adaptive gated attention.
        combined_features = torch.zeros(B, seq_len, self.fusion_module.total_channels).to(rgb_features.device)
        for t in range(seq_len):
            fused = self.fusion_module(
                rgb_features[:, t, :],
                depth_features[:, t, :],
                lidar_features[:, t, :]
            )
            combined_features[:, t, :] = fused

        # Feed through RNN.
        out, _ = self.rnn(combined_features)
        out = self.rnn_drop_out(out)
        out = self.linear(out)

        # Convert the relative output to NED convention.
        out = to_ned_pose(out, is_absolute=False)
        return out

    def compute_absolute_poses(self, relative_poses):
        # Integrate relative poses to compute absolute poses.
        absolute_poses = integrate_relative_poses(relative_poses)
        return absolute_poses

    def get_loss(self, x, y):
        predicted = self.forward(x)
        # Use sequences from second frame onward (accumulated relative errors)
        y = y[:, 1:, :]
        predicted = predicted[:, 1:, :]
        angle_loss = F.mse_loss(predicted[:, :, :3], y[:, :, :3])
        translation_loss = F.mse_loss(predicted[:, :, 3:], y[:, :, 3:])
        l2_lambda = par.l2_lambda
        l2_loss = l2_lambda * sum(torch.norm(param) for param in self.parameters() if param.requires_grad)

        # Depth consistency loss (only if depth is enabled)
        if par.enable_depth:
            depth_data = x[2]
            # Ensure matching dimensions using unsqueeze if needed
            depth_diff = depth_data[:, 1:, :, :, :] - depth_data[:, :-1, :, :, :]
            pred_trans_z = predicted[:, :, 5]
            depth_loss = F.mse_loss(pred_trans_z.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), depth_diff)
        else:
            depth_loss = torch.tensor(0.0, device=predicted.device)

        base_loss = par.k_factor * angle_loss + par.translation_loss_weight * translation_loss + l2_loss
        total_loss = base_loss + par.depth_consistency_loss_weight * depth_loss
        return total_loss

    def step(self, x, y, optimizer, scaler=None):
        optimizer.zero_grad()
        if scaler is not None:
            with torch.amp.autocast(device_type='cuda', enabled=True):
                loss = self.get_loss(x, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = self.get_loss(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()
        return loss.item()

if __name__ == "__main__":
    model = StereoAdaptiveVO(img_h=300, img_w=400, batch_norm=True, input_channels=3, hidden_size=512, num_layers=2)
    print(model)
