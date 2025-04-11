import torch
import torch.nn as nn
import torch.nn.functional as F
from params import par
from helper import to_ned_pose, integrate_relative_poses
from torch.autograd import Variable
import numpy as np

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, dropout=0):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)
        )

class AdaptiveGatedAttentionFusion(nn.Module):
    def __init__(self, rgb_channels, depth_channels, lidar_channels, depth_gate_scaling=par.depth_gate_scaling, imu_gate_scaling=par.imu_gate_scaling):
        super(AdaptiveGatedAttentionFusion, self).__init__()
        self.rgb_channels = rgb_channels
        self.depth_channels = depth_channels
        self.lidar_channels = lidar_channels
        self.total_channels = rgb_channels + depth_channels + lidar_channels

        self.rgb_gate = nn.Linear(self.total_channels, rgb_channels)
        self.depth_gate = nn.Linear(self.total_channels, depth_channels)
        self.lidar_gate = nn.Linear(self.total_channels, lidar_channels)

        self.depth_gate_scaling = depth_gate_scaling
        self.imu_gate_scaling = imu_gate_scaling

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

class StereoAdaptiveVO(nn.Module):
    def __init__(self, img_h, img_w, batch_norm, input_channels=3, hidden_size=512, num_layers=2):
        super(StereoAdaptiveVO, self).__init__()
        self.img_h = img_h
        self.img_w = img_w
        self.batch_norm = batch_norm
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # DeepVO-style feature extractor
        self.conv1   = conv(self.batch_norm,   6,   64, kernel_size=7, stride=2, dropout=par.conv_dropout[0])
        self.conv2   = conv(self.batch_norm,  64,  128, kernel_size=5, stride=2, dropout=par.conv_dropout[1])
        self.conv3   = conv(self.batch_norm, 128,  256, kernel_size=5, stride=2, dropout=par.conv_dropout[2])
        self.conv3_1 = conv(self.batch_norm, 256,  256, kernel_size=3, stride=1, dropout=par.conv_dropout[3])
        self.conv4   = conv(self.batch_norm, 256,  512, kernel_size=3, stride=2, dropout=par.conv_dropout[4])
        self.conv4_1 = conv(self.batch_norm, 512,  512, kernel_size=3, stride=1, dropout=par.conv_dropout[5])
        self.conv5   = conv(self.batch_norm, 512,  512, kernel_size=3, stride=2, dropout=par.conv_dropout[6])
        self.conv5_1 = conv(self.batch_norm, 512,  512, kernel_size=3, stride=1, dropout=par.conv_dropout[7])
        self.conv6   = conv(self.batch_norm, 512, 1024, kernel_size=3, stride=2, dropout=par.conv_dropout[8])

        # Load pretrained FlowNet weights
        if par.pretrained_flownet:
            try:
                pretrained_dict = torch.load(par.pretrained_flownet)
                model_dict = self.state_dict()
                update_dict = {k: v for k, v in pretrained_dict['state_dict'].items() if k in model_dict}
                missing_keys = set(model_dict.keys()) - set(update_dict.keys())
                if missing_keys:
                    print("Warning: Missing keys from pretrained checkpoint:", missing_keys)
                model_dict.update(update_dict)
                self.load_state_dict(model_dict)
                print("Loaded pretrained FlowNetS weights into DeepVO feature extractor.")
            except Exception as e:
                print(f"Error loading FlowNetS weights: {e}")

        # Compute the shape of the CNN output
        __tmp = Variable(torch.zeros(1, 6, img_h, img_w))
        __tmp = self.encode_image(__tmp)
        self.cnn_output_size = int(np.prod(__tmp.size()))
        self.flow_fc = nn.Linear(self.cnn_output_size, 256)

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

        if par.enable_imu:
            self.imu_fc = nn.Sequential(
                nn.Linear(6, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU()
            )
        else:
            self.imu_fc = nn.Identity()

        if par.enable_gps:
            self.gps_fc = nn.Sequential(
                nn.Linear(6, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU()
            )
        else:
            self.gps_fc = nn.Identity()

        # Fusion module
        self.fusion_module = AdaptiveGatedAttentionFusion(
            rgb_channels=512,  # Flow features from left + right (256 + 256)
            depth_channels=256 if par.enable_depth else 0,
            lidar_channels=(128 if par.enable_imu else 0) + (128 if par.enable_gps else 0)
        )

        self.rnn = nn.LSTM(
            input_size=512 + (256 if par.enable_depth else 0) + (128 if par.enable_imu else 0) + (128 if par.enable_gps else 0),
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=par.rnn_dropout_between
        )
        self.rnn_drop_out = nn.Dropout(par.rnn_dropout_out)
        self.linear = nn.Linear(hidden_size, 6)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name and 'bias' not in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
                        # Set forget gate bias to 1 (remember)
                        n = param.size(0)
                        start, end = n//4, n//2
                        param.data[start:end].fill_(1.)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def encode_image(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6(out_conv5)
        return out_conv6

    def forward(self, x):
        x_03, x_02, x_depth, x_imu, x_gps = x

        B, seq_len, C, H, W = x_03.shape
        flow_features = []
        for t in range(seq_len - 1):
            img_pair_left = torch.cat((x_03[:, t], x_03[:, t+1]), dim=1)  # [B, 6, H, W]
            img_pair_right = torch.cat((x_02[:, t], x_02[:, t+1]), dim=1)  # [B, 6, H, W]
            flow_left = self.encode_image(img_pair_left)  # [B, 1024, H/64, W/64]
            flow_right = self.encode_image(img_pair_right)  # [B, 1024, H/64, W/64]
            flow_left = flow_left.view(B, -1)  # Flatten: [B, 1024 * (H/64) * (W/64)]
            flow_right = flow_right.view(B, -1)  # Flatten: [B, 1024 * (H/64) * (W/64)]
            flow_left = self.flow_fc(flow_left)  # [B, 256]
            flow_right = self.flow_fc(flow_right)  # [B, 256]
            flow_combined = torch.cat((flow_left, flow_right), dim=-1)  # [B, 512]
            flow_features.append(flow_combined)

        flow_features = torch.stack(flow_features, dim=1)  # [B, seq_len-1, 512]

        if par.enable_depth:
            B, seq_len, C_depth, H_depth, W_depth = x_depth.shape
            x_depth = x_depth.view(B * seq_len, C_depth, H_depth, W_depth)
            depth_features = self.depth_conv(x_depth)
            depth_features = depth_features.view(B * seq_len, -1)
            depth_features = self.depth_fc(depth_features)
            depth_features = depth_features.view(B, seq_len, -1)
            depth_features = depth_features[:, :-1, :]  # Truncate to match flow features length
        else:
            depth_features = torch.zeros(B, seq_len-1, 0, device=x_03.device)

        if par.enable_imu:
            B, seq_len, _ = x_imu.shape
            x_imu = x_imu.view(B * seq_len, -1)
            imu_features = self.imu_fc(x_imu)
            imu_features = imu_features.view(B, seq_len, -1)
            imu_features = imu_features[:, :-1, :]  # Truncate to match flow features length
        else:
            imu_features = torch.zeros(B, seq_len-1, 0, device=x_03.device)

        if par.enable_gps:
            B, seq_len, _ = x_gps.shape
            x_gps = x_gps.view(B * seq_len, -1)
            gps_features = self.gps_fc(x_gps)
            gps_features = gps_features.view(B, seq_len, -1)
            gps_features = gps_features[:, :-1, :]  # Truncate to match flow features length
        else:
            gps_features = torch.zeros(B, seq_len-1, 0, device=x_03.device)

        lidar_features = torch.cat((imu_features, gps_features), dim=-1)

        combined_features = torch.zeros(B, seq_len-1, self.fusion_module.total_channels).to(flow_features.device)
        for t in range(seq_len-1):
            fused = self.fusion_module(
                flow_features[:, t, :],
                depth_features[:, t, :],
                lidar_features[:, t, :]
            )
            combined_features[:, t, :] = fused

        out, _ = self.rnn(combined_features)
        out = self.rnn_drop_out(out)
        out = self.linear(out)
        out = to_ned_pose(out, is_absolute=False)
        return out

    def compute_absolute_poses(self, relative_poses):
        return integrate_relative_poses(relative_poses)

    def get_loss(self, x, y):
        predicted = self.forward(x)
        y = y[:, 1:, :]  # Ground truth relative poses (already truncated in Dataset)
        predicted = predicted[:, :, :]  # [B, seq_len-1, 6]
        angle_loss = F.mse_loss(predicted[:, :, :3], y[:, :, :3])
        translation_loss = F.mse_loss(predicted[:, :, 3:], y[:, :, 3:])
        l2_lambda = par.l2_lambda
        l2_loss = l2_lambda * sum(torch.norm(param) for param in self.parameters() if param.requires_grad)
        
        if par.enable_depth:
            depth_data = x[2]  # x_depth
            depth_diff = depth_data[:, 1:, :, :, :] - depth_data[:, :-1, :, :, :]  # [B, seq_len-1, 1, H, W]
            pred_trans_z = predicted[:, :, 5]  # [B, seq_len-1]
            
            # Aggregate depth difference spatially to match pred_trans_z
            depth_diff_mean = depth_diff.mean(dim=[2, 3, 4])  # [B, seq_len-1]
            
            # Compute depth consistency loss
            depth_loss = F.mse_loss(pred_trans_z, depth_diff_mean)
        else:
            depth_loss = torch.tensor(0.0, device=predicted.device)
        
        # GPS supervised loss
        if par.enable_gps:
            x_03, x_02, x_depth, x_imu, x_gps = x
            # x_gps: [B, seq_len, 6] (position: [:3], velocity: [3:])
            gps_ground_truth = x_gps  # [B, seq_len, 6]
            
            # Integrate predicted relative poses to absolute poses
            predicted_absolute = self.compute_absolute_poses(predicted.unsqueeze(0))[0]  # [B, seq_len, 6]
            
            # Align predicted absolute poses with GPS ground truth at the first timestep
            offset = gps_ground_truth[:, 0, :] - predicted_absolute[:, 0, :]
            predicted_absolute += offset
            
            # Compute GPS supervised loss (compare positions and velocities)
            gps_pos_loss = F.mse_loss(predicted_absolute[:, :, 3:], gps_ground_truth[:, 1:, :3])  # Position loss
            gps_vel_loss = F.mse_loss(predicted_absolute[:, :, 3:], gps_ground_truth[:, 1:, 3:])  # Velocity loss (approximate)
            gps_loss = (gps_pos_loss + gps_vel_loss) / 2
        else:
            gps_loss = torch.tensor(0.0, device=predicted.device)
        
        base_loss = par.k_factor * angle_loss + par.translation_loss_weight * translation_loss + l2_loss
        total_loss = base_loss + par.depth_consistency_loss_weight * depth_loss + par.gps_loss_weight * gps_loss
        
        # Debug loss components
        print(f"Angle Loss: {angle_loss.item():.6f}")
        print(f"Translation Loss: {translation_loss.item():.6f}")
        print(f"Depth Loss: {depth_loss.item():.6f}")
        print(f"GPS Loss: {gps_loss.item():.6f}")
        print(f"Base Loss: {base_loss.item():.6f}")
        print(f"Total Loss: {total_loss.item():.6f}")
        
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