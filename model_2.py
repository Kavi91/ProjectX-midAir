import torch
import torch.nn as nn
import os
import pickle
from params import par
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_
import numpy as np
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, padding=None, dropout=0):
    if padding is None:
        padding = (kernel_size-1)//2
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)
        )

class CrossModalAttention(nn.Module):
    def __init__(self, query_channels, key_value_channels, num_heads=8, reduction=16):
        super(CrossModalAttention, self).__init__()
        self.num_heads = num_heads
        self.query_channels = query_channels
        self.key_value_channels = key_value_channels
        
        # Handle case where query_channels might be 0
        if query_channels == 0 or key_value_channels == 0:
            self.head_dim = 0  # No attention if either modality is disabled
            self.scale = 1.0   # Avoid computation when channels are 0
        else:
            self.head_dim = query_channels // num_heads
            assert self.head_dim * num_heads == query_channels, "query_channels must be divisible by num_heads"
            self.scale = (self.head_dim) ** -0.5

        # Query projection (based on query modality)
        self.query = nn.Conv2d(query_channels, query_channels, kernel_size=1) if query_channels > 0 else None
        # Key/Value projections (based on key/value modality)
        self.key = nn.Conv2d(key_value_channels, query_channels, kernel_size=1) if key_value_channels > 0 and query_channels > 0 else None
        self.value = nn.Conv2d(key_value_channels, query_channels, kernel_size=1) if key_value_channels > 0 and query_channels > 0 else None

        # Output projection
        self.out_proj = nn.Conv2d(query_channels, query_channels, kernel_size=1) if query_channels > 0 else None

    def forward(self, query_features, key_features, value_features):
        B, C_q, H, W = query_features.size()
        B, C_kv, H_kv, W_kv = key_features.size()
        
        if C_q == 0 or C_kv == 0:
            # Return query_features unchanged if either modality has zero channels
            return query_features

        assert C_q == self.query_channels, f"Query channels mismatch: expected {self.query_channels}, got {C_q}"
        assert C_kv == self.key_value_channels, f"Key/Value channels mismatch: expected {self.key_value_channels}, got {C_kv}"

        # Project features
        Q = self.query(query_features).view(B, self.num_heads, self.head_dim, H * W)  # [B, num_heads, head_dim, H*W]
        K = self.key(key_features).view(B, self.num_heads, self.head_dim, H_kv * W_kv)  # [B, num_heads, head_dim, H*W]
        V = self.value(value_features).view(B, self.num_heads, self.head_dim, H_kv * W_kv)  # [B, num_heads, head_dim, H*W]

        # Attention scores
        scores = torch.einsum('bnhd,bnkd->bnhk', Q, K) * self.scale  # [B, num_heads, H*W, H*W]
        attn = F.softmax(scores, dim=-1)

        # Apply attention
        out = torch.einsum('bnhk,bnkd->bnhd', attn, V)  # [B, num_heads, H*W, head_dim]
        out = out.view(B, self.query_channels, H, W)  # [B, C_q, H, W]

        # Output projection
        out = self.out_proj(out)
        return out

class AdaptiveGatedAttentionFusion(nn.Module):
    def __init__(self, rgb_channels, depth_channels, lidar_channels=0, reduction=16):
        super(AdaptiveGatedAttentionFusion, self).__init__()
        self.total_channels = rgb_channels + depth_channels + lidar_channels
        self.rgb_channels = rgb_channels
        self.depth_channels = depth_channels
        self.lidar_channels = lidar_channels

        # Cross-modal attention for each modality pair (only if both modalities have channels)
        self.rgb_to_depth = CrossModalAttention(query_channels=rgb_channels, key_value_channels=depth_channels, num_heads=8, reduction=reduction) if rgb_channels > 0 and depth_channels > 0 else None
        self.rgb_to_lidar = CrossModalAttention(query_channels=rgb_channels, key_value_channels=lidar_channels, num_heads=8, reduction=reduction) if rgb_channels > 0 and lidar_channels > 0 else None
        self.depth_to_rgb = CrossModalAttention(query_channels=depth_channels, key_value_channels=rgb_channels, num_heads=8, reduction=reduction) if depth_channels > 0 and rgb_channels > 0 else None
        self.depth_to_lidar = CrossModalAttention(query_channels=depth_channels, key_value_channels=lidar_channels, num_heads=8, reduction=reduction) if depth_channels > 0 and lidar_channels > 0 else None
        self.lidar_to_rgb = CrossModalAttention(query_channels=lidar_channels, key_value_channels=rgb_channels, num_heads=8, reduction=reduction) if rgb_channels > 0 and lidar_channels > 0 else None
        self.lidar_to_depth = CrossModalAttention(query_channels=lidar_channels, key_value_channels=depth_channels, num_heads=8, reduction=reduction) if depth_channels > 0 and lidar_channels > 0 else None

        # SE blocks for channel-wise attention (only if channels > 0)
        self.se_rgb = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(rgb_channels, rgb_channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(rgb_channels // reduction, rgb_channels, kernel_size=1),
            nn.Sigmoid()
        ) if rgb_channels > 0 else None
        self.se_depth = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(depth_channels, depth_channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(depth_channels // reduction, depth_channels, kernel_size=1),
            nn.Sigmoid()
        ) if depth_channels > 0 else None
        self.se_lidar = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(lidar_channels, lidar_channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(lidar_channels // reduction, lidar_channels, kernel_size=1),
            nn.Sigmoid()
        ) if lidar_channels > 0 else None

        # Gating mechanism (adjust input channels based on active modalities)
        gate_input_channels = sum([c for c in [rgb_channels, depth_channels, lidar_channels] if c > 0]) + 3
        self.gate = nn.Sequential(
            nn.Conv2d(gate_input_channels, 3, kernel_size=1),
            nn.Sigmoid()
        ) if gate_input_channels > 3 else None

        # Final fusion (only active channels)
        self.fusion = nn.Conv2d(self.total_channels, self.total_channels, kernel_size=1) if self.total_channels > 0 else None

    def forward(self, rgb_features, depth_features, lidar_features=None):
        # Handle case where all modalities might be disabled
        if self.total_channels == 0:
            raise ValueError("No active modalities for fusion.")

        # Ensure spatial dimensions match for active modalities
        if self.rgb_channels > 0:
            ref_h, ref_w = rgb_features.size(2), rgb_features.size(3)
            batch_size = rgb_features.size(0)
        elif self.depth_channels > 0:
            ref_h, ref_w = depth_features.size(2), depth_features.size(3)
            batch_size = depth_features.size(0)
        elif self.lidar_channels > 0 and lidar_features is not None:
            ref_h, ref_w = lidar_features.size(2), lidar_features.size(3)
            batch_size = lidar_features.size(0)
        else:
            raise ValueError("At least one modality must be active.")

        active_features = []
        if self.rgb_channels > 0:
            active_features.append(rgb_features)
        if self.depth_channels > 0:
            active_features.append(depth_features)
            if depth_features.size()[2:] != (ref_h, ref_w):
                depth_features = F.interpolate(depth_features, size=(ref_h, ref_w), mode='bilinear', align_corners=False)
        if self.lidar_channels > 0 and lidar_features is not None:
            active_features.append(lidar_features)
            if lidar_features.size()[2:] != (ref_h, ref_w):
                lidar_features = F.interpolate(lidar_features, size=(ref_h, ref_w), mode='bilinear', align_corners=False)

        # Cross-modal attention
        rgb_cross = rgb_features if self.rgb_channels > 0 else None
        if self.rgb_to_depth and rgb_cross is not None and self.depth_channels > 0:
            rgb_cross = rgb_cross + self.rgb_to_depth(rgb_features, depth_features, depth_features)
        if self.rgb_to_lidar and rgb_cross is not None and self.lidar_channels > 0 and lidar_features is not None:
            rgb_cross = rgb_cross + self.rgb_to_lidar(rgb_features, lidar_features, lidar_features)

        depth_cross = depth_features if self.depth_channels > 0 else None
        if self.depth_to_rgb and depth_cross is not None and self.rgb_channels > 0:
            depth_cross = depth_cross + self.depth_to_rgb(depth_features, rgb_features, rgb_features)
        if self.depth_to_lidar and depth_cross is not None and self.lidar_channels > 0 and lidar_features is not None:
            depth_cross = depth_cross + self.depth_to_lidar(depth_features, lidar_features, lidar_features)

        lidar_cross = lidar_features if self.lidar_channels > 0 and lidar_features is not None else None
        if self.lidar_to_rgb and lidar_cross is not None and self.rgb_channels > 0:
            lidar_cross = lidar_cross + self.lidar_to_rgb(lidar_features, rgb_features, rgb_features)
        if self.lidar_to_depth and lidar_cross is not None and self.depth_channels > 0:
            lidar_cross = lidar_cross + self.lidar_to_depth(lidar_features, depth_features, depth_features)

        # SE blocks (channel-wise attention)
        rgb_refined = rgb_cross * self.se_rgb(rgb_cross) if self.se_rgb and rgb_cross is not None else rgb_cross
        depth_refined = depth_cross * self.se_depth(depth_cross) if self.se_depth and depth_cross is not None else depth_cross
        lidar_refined = lidar_cross * self.se_lidar(lidar_cross) if self.se_lidar and lidar_cross is not None else lidar_cross

        # Quality metrics for gating (always provide 3 channels: RGB, depth, LiDAR)
        quality_metrics = []
        # Use dynamic batch size for dummy metric
        device = rgb_features.device if rgb_features is not None else depth_features.device if depth_features is not None else lidar_features.device
        dummy_metric = torch.zeros(batch_size, 1, ref_h, ref_w, device=device)

        if self.rgb_channels > 0:
            rgb_variance = rgb_features.var(dim=1, keepdim=True)  # Shape: [batch_size, 1, 3, 10]
            quality_metrics.append(rgb_variance)
        else:
            quality_metrics.append(dummy_metric)

        if self.depth_channels > 0:
            depth_sparsity = (depth_features == 0).float().mean(dim=1, keepdim=True)  # Shape: [batch_size, 1, 3, 10]
            quality_metrics.append(depth_sparsity)
        else:
            quality_metrics.append(dummy_metric)

        if self.lidar_channels > 0 and lidar_features is not None:
            lidar_sparsity = (lidar_features == 0).float().mean(dim=1, keepdim=True)  # Shape: [batch_size, 1, 3, 10]
            quality_metrics.append(lidar_sparsity)
        else:
            quality_metrics.append(dummy_metric)
        
        # Debug print to check tensor sizes
        #for i, metric in enumerate(quality_metrics):
            #print(f"Quality metric {i} shape: {metric.shape}")
        
        quality_metrics = torch.cat(quality_metrics, dim=1)  # Always 3 channels

        # Concatenate features for gating
        features_to_concat = [f for f in [rgb_refined, depth_refined, lidar_refined] if f is not None]
        if not features_to_concat:
            raise ValueError("No features to concatenate for fusion.")
        combined = torch.cat(features_to_concat, dim=1)
        combined_with_metrics = torch.cat((combined, quality_metrics), dim=1)

        # Gating
        if self.gate:
            gate_weights = self.gate(combined_with_metrics)
            rgb_gate, depth_gate, lidar_gate = gate_weights[:, 0:1], gate_weights[:, 1:2], gate_weights[:, 2:3]
        else:
            rgb_gate = torch.ones_like(quality_metrics[:, 0:1]) if self.rgb_channels > 0 else torch.zeros_like(quality_metrics[:, 0:1])
            depth_gate = torch.ones_like(quality_metrics[:, 1:2]) if self.depth_channels > 0 else torch.zeros_like(quality_metrics[:, 1:2])
            lidar_gate = torch.ones_like(quality_metrics[:, 2:3]) if self.lidar_channels > 0 else torch.zeros_like(quality_metrics[:, 2:3])

        # Apply gates
        rgb_weighted = rgb_refined * rgb_gate if rgb_refined is not None else None
        depth_weighted = depth_refined * depth_gate if depth_refined is not None else None
        lidar_weighted = lidar_refined * lidar_gate if lidar_refined is not None else None

        # Final fusion
        weighted_features = [f for f in [rgb_weighted, depth_weighted, lidar_weighted] if f is not None]
        if not weighted_features:
            raise ValueError("No weighted features for fusion.")
        fused_features = self.fusion(torch.cat(weighted_features, dim=1)) if self.fusion else weighted_features[0]
        return fused_features

class DeepVO(nn.Module):
    def __init__(self, imsize1, imsize2, batchNorm=True):
        super(DeepVO, self).__init__()
        self.batchNorm = batchNorm
        self.clip = par.clip
        self.conv1   = conv(self.batchNorm, 6, 64, kernel_size=7, stride=2, dropout=par.conv_dropout[0])
        self.conv2   = conv(self.batchNorm, 64, 128, kernel_size=5, stride=2, dropout=par.conv_dropout[1])
        self.conv3   = conv(self.batchNorm, 128, 256, kernel_size=5, stride=2, dropout=par.conv_dropout[2])
        self.conv3_1 = conv(self.batchNorm, 256, 256, kernel_size=3, stride=1, dropout=par.conv_dropout[3])
        self.conv4   = conv(self.batchNorm, 256, 512, kernel_size=3, stride=2, dropout=par.conv_dropout[4])
        self.conv4_1 = conv(self.batchNorm, 512, 512, kernel_size=3, stride=1, dropout=par.conv_dropout[5])
        self.conv5   = conv(self.batchNorm, 512, 512, kernel_size=3, stride=2, dropout=par.conv_dropout[6])
        self.conv5_1 = conv(self.batchNorm, 512, 512, kernel_size=3, stride=1, dropout=par.conv_dropout[7])
        self.conv6   = conv(self.batchNorm, 512, 1024, kernel_size=3, stride=2, dropout=par.conv_dropout[8])
        
        __tmp = Variable(torch.zeros(1, 6, imsize1, imsize2))
        __tmp = self.encode_image(__tmp)
        fused_feature_size = int(np.prod(__tmp.size())) * 2
        
        self.rnn = nn.LSTM(
            input_size=fused_feature_size, 
            hidden_size=par.rnn_hidden_size, 
            num_layers=2, 
            dropout=par.rnn_dropout_between, 
            batch_first=True)
        self.rnn_drop_out = nn.Dropout(par.rnn_dropout_out)
        self.linear = nn.Linear(in_features=par.rnn_hidden_size, out_features=6)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LSTM):
                kaiming_normal_(m.weight_ih_l0)
                kaiming_normal_(m.weight_hh_l0)
                m.bias_ih_l0.data.zero_()
                m.bias_hh_l0.data.zero_()
                n = m.bias_hh_l0.size(0)
                start, end = n//4, n//2
                m.bias_hh_l0.data[start:end].fill_(1.)
                kaiming_normal_(m.weight_ih_l1)
                kaiming_normal_(m.weight_hh_l1)
                m.bias_ih_l1.data.zero_()
                m.bias_hh_l1.data.zero_()
                n = m.bias_hh_l1.size(0)
                start, end = n//4, n//2
                m.bias_hh_l1.data[start:end].fill_(1.)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x_03, x_02 = x
        
        x_03 = torch.cat((x_03[:, :-1], x_03[:, 1:]), dim=2)
        x_02 = torch.cat((x_02[:, :-1], x_02[:, 1:]), dim=2)
        
        batch_size = x_03.size(0)
        seq_len = x_03.size(1)
        
        x_03 = x_03.view(batch_size * seq_len, x_03.size(2), x_03.size(3), x_03.size(4))
        x_02 = x_02.view(batch_size * seq_len, x_02.size(2), x_03.size(3), x_03.size(4))
        
        features_03 = self.encode_image(x_03)
        features_02 = self.encode_image(x_02)
        
        fused_features = torch.cat((features_03, features_02), dim=1)
        fused_features = fused_features.view(batch_size, seq_len, -1)
        
        out, hc = self.rnn(fused_features)
        out = self.rnn_drop_out(out)
        out = self.linear(out)
        
        return out

    def encode_image(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6(out_conv5)
        return out_conv6

    def get_loss(self, x, y):
        predicted = self.forward(x)
        y = y[:, 1:, :]
        angle_loss = torch.nn.functional.mse_loss(predicted[:, :, :3], y[:, :, :3])
        translation_loss = torch.nn.functional.mse_loss(predicted[:, :, 3:], y[:, :, 3:])
        loss = (100 * angle_loss + translation_loss)
        return loss

    def step(self, x, y, optimizer):
        optimizer.zero_grad()
        loss = self.get_loss(x, y)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.rnn.parameters(), self.clip)
        optimizer.step()
        return loss

class StereoAdaptiveVO(DeepVO):
    def __init__(self, imsize1, imsize2, batchNorm=True):
        super(StereoAdaptiveVO, self).__init__(imsize1, imsize2, batchNorm)
        
        # Depth processing (only if enabled)
        if par.enable_depth:
            self.depth_conv1 = conv(self.batchNorm, 2, 32, kernel_size=7, stride=2, dropout=par.conv_dropout[0])
            self.depth_conv2 = conv(self.batchNorm, 32, 64, kernel_size=5, stride=2, dropout=par.conv_dropout[1])
            self.depth_conv3 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=2, dropout=par.conv_dropout[2])
            self.depth_conv4 = conv(self.batchNorm, 128, 256, kernel_size=3, stride=2, dropout=par.conv_dropout[4])
        else:
            self.depth_conv1 = self.depth_conv2 = self.depth_conv3 = self.depth_conv4 = None
        
        # IMU and GPS processing (input size is 12 due to pairing of consecutive frames)
        self.imu_mlp = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        ) if par.enable_imu else None
        self.gps_mlp = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        ) if par.enable_gps else None
        
        # Compute feature sizes and save to pickle
        stats_pickle_path = "datainfo/dataset_stats.pickle"
        if os.path.exists(stats_pickle_path):
            print(f"Loading model feature sizes from {stats_pickle_path}")
            with open(stats_pickle_path, 'rb') as f:
                stats = pickle.load(f)
            if 'rgb_feature_size' in stats:
                rgb_feature_size = stats['rgb_feature_size']
                depth_feature_size = stats['depth_feature_size']
                vision_feature_size = stats['vision_feature_size']
                rnn_input_size = stats['rnn_input_size']
                # Compute channels and dimensions from loaded feature sizes
                __tmp_rgb = Variable(torch.zeros(1, 6, imsize1, imsize2))
                __tmp_rgb = self.encode_image(__tmp_rgb)
                rgb_channels = __tmp_rgb.size(1) * 2 if par.enable_rgb else 0  # 2048
                rgb_h, rgb_w = __tmp_rgb.size(2), __tmp_rgb.size(3)
                
                if par.enable_depth:
                    __tmp_depth = Variable(torch.zeros(1, 2, imsize1, imsize2))
                    __tmp_depth = self.encode_depth(__tmp_depth)
                    depth_channels = __tmp_depth.size(1)  # 256
                    depth_h, depth_w = __tmp_depth.size(2), __tmp_depth.size(3)
                else:
                    depth_channels = 0
                    depth_h, depth_w = rgb_h, rgb_w  # Match RGB dimensions
                
                lidar_channels = 0
                lidar_h, lidar_w = rgb_h, rgb_w  # No LiDAR in Mid-Air
            else:
                # Compute feature sizes if not present in pickle
                __tmp_rgb = Variable(torch.zeros(1, 6, imsize1, imsize2))
                __tmp_rgb = self.encode_image(__tmp_rgb)
                rgb_channels = __tmp_rgb.size(1) * 2 if par.enable_rgb else 0  # 2048
                rgb_h, rgb_w = __tmp_rgb.size(2), __tmp_rgb.size(3)
                rgb_feature_size = int(np.prod(__tmp_rgb.size())) * 2 if par.enable_rgb else 0
                print(f"Computed fused RGB feature size: {rgb_feature_size} (channels: {rgb_channels}, H: {rgb_h}, W: {rgb_w})")
                
                if par.enable_depth:
                    __tmp_depth = Variable(torch.zeros(1, 2, imsize1, imsize2))
                    __tmp_depth = self.encode_depth(__tmp_depth)
                    depth_channels = __tmp_depth.size(1)  # 256
                    depth_h, depth_w = __tmp_depth.size(2), __tmp_depth.size(3)
                    depth_feature_size = int(np.prod(__tmp_depth.size()))
                else:
                    depth_channels = 0
                    depth_h, depth_w = rgb_h, rgb_w
                    depth_feature_size = 0
                print(f"Computed depth feature size: {depth_feature_size} (channels: {depth_channels}, H: {depth_h}, W: {rgb_w})")
                
                lidar_channels = 0
                lidar_h, lidar_w = rgb_h, rgb_w  # No LiDAR in Mid-Air
                
                self.fusion_module = AdaptiveGatedAttentionFusion(rgb_channels, depth_channels, lidar_channels)
                
                # Define variables before conditional assignment
                __tmp_rgb_fused = torch.zeros(1, rgb_channels, rgb_h, rgb_w) if rgb_channels > 0 else None
                __tmp_depth_resized = torch.zeros(1, depth_channels, rgb_h, rgb_w) if depth_channels > 0 else None
                __tmp_lidar_resized = None if not par.enable_lidar else torch.zeros(1, lidar_channels, rgb_h, rgb_w)
                
                # Use explicit None check instead of 'or' operator
                rgb_input = __tmp_rgb_fused if __tmp_rgb_fused is not None else torch.zeros(1, 1, rgb_h, rgb_w)
                depth_input = __tmp_depth_resized if __tmp_depth_resized is not None else torch.zeros(1, 1, rgb_h, rgb_w)
                lidar_input = __tmp_lidar_resized if __tmp_lidar_resized is not None else torch.zeros(1, 1, rgb_h, rgb_w)
                __tmp_fused = self.fusion_module(rgb_input, depth_input, lidar_input)
                vision_feature_size = int(np.prod(__tmp_fused.size()))
                print(f"Adjusted vision feature size (RGB + depth after fusion): {vision_feature_size}")
                
                imu_feature_size = 256 if par.enable_imu else 0
                gps_feature_size = 256 if par.enable_gps else 0
                rnn_input_size = vision_feature_size + imu_feature_size + gps_feature_size
                print(f"Total LSTM input size (vision + IMU + GPS): {rnn_input_size}")
                
                stats.update({
                    'rgb_feature_size': rgb_feature_size,
                    'depth_feature_size': depth_feature_size,
                    'vision_feature_size': vision_feature_size,
                    'rnn_input_size': rnn_input_size
                })
                with open(stats_pickle_path, 'wb') as f:
                    pickle.dump(stats, f)
                print(f"Updated dataset statistics with feature sizes in {stats_pickle_path}")
        else:
            __tmp_rgb = Variable(torch.zeros(1, 6, imsize1, imsize2))
            __tmp_rgb = self.encode_image(__tmp_rgb)
            rgb_channels = __tmp_rgb.size(1) * 2 if par.enable_rgb else 0
            rgb_h, rgb_w = __tmp_rgb.size(2), __tmp_rgb.size(3)
            rgb_feature_size = int(np.prod(__tmp_rgb.size())) * 2 if par.enable_rgb else 0
            print(f"Computed fused RGB feature size: {rgb_feature_size} (channels: {rgb_channels}, H: {rgb_h}, W: {rgb_w})")
            
            if par.enable_depth:
                __tmp_depth = Variable(torch.zeros(1, 2, imsize1, imsize2))
                __tmp_depth = self.encode_depth(__tmp_depth)
                depth_channels = __tmp_depth.size(1)
                depth_h, depth_w = __tmp_depth.size(2), __tmp_depth.size(3)
                depth_feature_size = int(np.prod(__tmp_depth.size()))
            else:
                depth_channels = 0
                depth_h, depth_w = rgb_h, rgb_w
                depth_feature_size = 0
            print(f"Computed depth feature size: {depth_feature_size} (channels: {depth_channels}, H: {depth_h}, W: {rgb_w})")
            
            lidar_channels = 0
            lidar_h, lidar_w = rgb_h, rgb_w
            
            self.fusion_module = AdaptiveGatedAttentionFusion(rgb_channels, depth_channels, lidar_channels)
            
            # Use explicit None check instead of 'or' operator
            __tmp_rgb_fused = __tmp_rgb_fused if rgb_channels > 0 else torch.zeros(1, 1, rgb_h, rgb_w)
            __tmp_depth_resized = __tmp_depth_resized if depth_channels > 0 else torch.zeros(1, 1, rgb_h, rgb_w)
            __tmp_lidar_resized = __tmp_lidar_resized if par.enable_lidar else torch.zeros(1, 1, rgb_h, rgb_w)
            __tmp_fused = self.fusion_module(__tmp_rgb_fused, __tmp_depth_resized, __tmp_lidar_resized)
            vision_feature_size = int(np.prod(__tmp_fused.size()))
            print(f"Adjusted vision feature size (RGB + depth after fusion): {vision_feature_size}")
            
            imu_feature_size = 256 if par.enable_imu else 0
            gps_feature_size = 256 if par.enable_gps else 0
            rnn_input_size = vision_feature_size + imu_feature_size + gps_feature_size
            print(f"Total LSTM input size (vision + IMU + GPS): {rnn_input_size}")
            
            stats = {
                'rgb_feature_size': rgb_feature_size,
                'depth_feature_size': depth_feature_size,
                'vision_feature_size': vision_feature_size,
                'rnn_input_size': rnn_input_size
            }
            with open(stats_pickle_path, 'wb') as f:
                pickle.dump(stats, f)
            print(f"Saved model feature sizes to {stats_pickle_path}")

        # Initialize fusion module with computed channels
        self.fusion_module = AdaptiveGatedAttentionFusion(rgb_channels, depth_channels, lidar_channels)
        
        self.rnn = nn.LSTM(
            input_size=rnn_input_size, 
            hidden_size=par.rnn_hidden_size, 
            num_layers=2, 
            dropout=par.rnn_dropout_between, 
            batch_first=True)
        
        for m in self.rnn.modules():
            if isinstance(m, nn.LSTM):
                kaiming_normal_(m.weight_ih_l0)
                kaiming_normal_(m.weight_hh_l0)
                m.bias_ih_l0.data.zero_()
                m.bias_hh_l0.data.zero_()
                n = m.bias_hh_l0.size(0)
                start, end = n//4, n//2
                m.bias_hh_l0.data[start:end].fill_(1.)
                kaiming_normal_(m.weight_ih_l1)
                kaiming_normal_(m.weight_hh_l1)
                m.bias_ih_l1.data.zero_()
                m.bias_hh_l1.data.zero_()
                n = m.bias_hh_l1.size(0)
                start, end = n//4, n//2
                m.bias_hh_l1.data[start:end].fill_(1.)

    def encode_depth(self, x):
        if not par.enable_depth:
            raise ValueError("Depth encoding called but depth is disabled.")
        out_conv2 = self.depth_conv2(self.depth_conv1(x))
        out_conv3 = self.depth_conv3(out_conv2)
        out_conv4 = self.depth_conv4(out_conv3)
        return out_conv4

    def forward(self, x):
        x_03, x_02, x_depth, x_imu, x_gps = x
        
        # Debug input shapes
        #print(f"x_03 shape: {x_03.shape}")
        #print(f"x_02 shape: {x_02.shape}")
        
        if x_03.size(1) < 2 or x_02.size(1) < 2:
            raise ValueError(f"Sequence length too short: RGB_03={x_03.size(1)}, RGB_02={x_02.size(1)}")
        if par.enable_depth and x_depth.size(1) < 2:
            raise ValueError(f"Depth sequence length too short: {x_depth.size(1)}")
        if par.enable_imu and x_imu.size(1) < 2:
            raise ValueError(f"IMU sequence length too short: {x_imu.size(1)}")
        if par.enable_gps and x_gps.size(1) < 2:
            raise ValueError(f"GPS sequence length too short: {x_gps.size(1)}")
        
        x_03 = torch.cat((x_03[:, :-1], x_03[:, 1:]), dim=2)
        x_02 = torch.cat((x_02[:, :-1], x_02[:, 1:]), dim=2)
        if par.enable_depth:
            x_depth = torch.cat((x_depth[:, :-1], x_depth[:, 1:]), dim=2)
        if par.enable_imu:
            x_imu = torch.cat((x_imu[:, :-1], x_imu[:, 1:]), dim=2)
        if par.enable_gps:
            x_gps = torch.cat((x_gps[:, :-1], x_gps[:, 1:]), dim=2)
        
        # Debug shapes after pairing
        #print(f"x_03 paired shape: {x_03.shape}")
        #print(f"x_02 paired shape: {x_02.shape}")
        
        new_seq_len = x_03.size(1)
        batch_size = x_03.size(0)
        
        # Reshape with explicit dimension checks
        x_03 = x_03.view(batch_size * new_seq_len, x_03.size(2), x_03.size(3), x_03.size(4))
        x_02 = x_02.view(batch_size * new_seq_len, x_02.size(2), x_02.size(3), x_02.size(4))
        
        if par.enable_depth:
            x_depth = x_depth.view(batch_size * new_seq_len, x_depth.size(2), x_depth.size(3), x_depth.size(4))
        if par.enable_imu:
            x_imu = x_imu.view(batch_size * new_seq_len, x_imu.size(2))
        if par.enable_gps:
            x_gps = x_gps.view(batch_size * new_seq_len, x_gps.size(2))
        
        # Debug shapes after reshaping
        #print(f"x_03 reshaped shape: {x_03.shape}")
        #print(f"x_02 reshaped shape: {x_02.shape}")
        
        features_03 = self.encode_image(x_03) if par.enable_rgb else None
        features_02 = self.encode_image(x_02) if par.enable_rgb else None
        rgb_features = torch.cat((features_03, features_02), dim=1) if par.enable_rgb else None
        
        depth_features = self.encode_depth(x_depth) if par.enable_depth else None
        
        fused_features = self.fusion_module(rgb_features, depth_features, None)  # No LiDAR
        fused_features = fused_features.view(batch_size, new_seq_len, -1)
        #bottleneck = nn.Linear(69120, 2048).to(fused_features.device)  # Compress to 2048
        #fused_features = bottleneck(fused_features)  # [24, 6, 2048]
        
        # Process IMU and GPS only if enabled
        features_to_concat = [fused_features]
        if par.enable_imu and self.imu_mlp:
            imu_features = self.imu_mlp(x_imu)
            imu_features = imu_features.view(batch_size, new_seq_len, -1)
            features_to_concat.append(imu_features)
        if par.enable_gps and self.gps_mlp:
            gps_features = self.gps_mlp(x_gps)
            gps_features = gps_features.view(batch_size, new_seq_len, -1)
            features_to_concat.append(gps_features)
        
        combined_features = torch.cat(features_to_concat, dim=2)
        
        out, hc = self.rnn(combined_features)
        out = self.rnn_drop_out(out)
        out = self.linear(out)
        return out

    def get_loss(self, x, y):
        predicted = self.forward(x)
        y = y[:, 1:, :]  # Align with sequence length after pairing
        angle_loss = torch.nn.functional.mse_loss(predicted[:, :, :3], y[:, :, :3])
        translation_loss = torch.nn.functional.mse_loss(predicted[:, :, 3:], y[:, :, 3:])
        loss = 100 * angle_loss + translation_loss
        return loss

    def step(self, x, y, optimizer):
        optimizer.zero_grad()
        loss = self.get_loss(x, y)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.rnn.parameters(), self.clip)
        optimizer.step()
        return loss