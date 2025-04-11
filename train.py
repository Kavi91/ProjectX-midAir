import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import pandas as pd
import pickle
import torch.nn.functional as F
from params import par
from model import StereoAdaptiveVO
from data_helper import get_data_info, SortedRandomBatchSampler, ImageSequenceDataset
import wandb
from tqdm import tqdm
from torch.amp import GradScaler, autocast
import argparse
from params import Parameters

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train StereoAdaptiveVO model')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
args = parser.parse_args()

# Initialize Parameters with the specified batch size
par = Parameters(batch_size=args.batch_size)

# Initialize WandB
wandb.init(project="deepvo_training_midair", config=vars(par))
wandb.config.update({"start_time": time.time()})

# Record hyperparameters
mode = 'a' if par.resume else 'w'
with open(par.record_path, mode) as f:
    f.write('\n' + '=' * 50 + '\n')
    f.write('\n'.join(f"{k}: {v}" for k, v in vars(par).items()))
    f.write('\n' + '=' * 50 + '\n')

# Data preparation
if os.path.isfile(par.train_data_info_path) and os.path.isfile(par.valid_data_info_path):
    print('Loading data info from:', par.train_data_info_path)
    train_df = pd.read_pickle(par.train_data_info_path)
    valid_df = pd.read_pickle(par.valid_data_info_path)
else:
    print('Creating new data info...')
    train_df, valid_df = get_data_info(
        climate_sets=par.climate_sets, 
        seq_len=par.seq_len, 
        overlap=par.overlap,
        sample_times=par.sample_times, 
        shuffle=True, 
        sort=True, 
        include_test=False
    )
    train_df.to_pickle(par.train_data_info_path)
    valid_df.to_pickle(par.valid_data_info_path)

# Samplers and DataLoaders
train_sampler = SortedRandomBatchSampler(train_df, par.batch_size, drop_last=True)
train_dataset = ImageSequenceDataset(
    train_df, par.resize_mode, (par.img_h, par.img_w), par.img_means_03, par.img_stds_03,
    par.img_means_02, par.img_stds_02, par.minus_point_5, is_training=True
)
train_dl = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=par.num_workers, pin_memory=par.pin_mem)

valid_sampler = SortedRandomBatchSampler(valid_df, par.batch_size, drop_last=True)
valid_dataset = ImageSequenceDataset(
    valid_df, par.resize_mode, (par.img_h, par.img_w), par.img_means_03, par.img_stds_03,
    par.img_means_02, par.img_stds_02, par.minus_point_5, is_training=False
)
valid_dl = DataLoader(valid_dataset, batch_sampler=valid_sampler, num_workers=par.num_workers, pin_memory=par.pin_mem)

# Load or compute dataset sizes
stats_pickle_path = "datainfo/dataset_stats.pickle"
if os.path.exists(stats_pickle_path):
    print(f"Loading dataset sizes from {stats_pickle_path}")
    with open(stats_pickle_path, 'rb') as f:
        stats = pickle.load(f)
    if 'num_train_samples' in stats:
        num_train_samples = stats['num_train_samples']
        num_valid_samples = stats['num_valid_samples']
        num_train_batches = stats['num_train_batches']
        num_valid_batches = stats['num_valid_batches']
    else:
        num_train_samples = len(train_df.index)
        num_valid_samples = len(valid_df.index)
        num_train_batches = len(train_dl)
        num_valid_batches = len(valid_dl)
        stats.update({
            'num_train_samples': num_train_samples,
            'num_valid_samples': num_valid_samples,
            'num_train_batches': num_train_batches,
            'num_valid_batches': num_valid_batches
        })
        with open(stats_pickle_path, 'wb') as f:
            pickle.dump(stats, f)
        print(f"Updated dataset statistics in {stats_pickle_path}")
else:
    num_train_samples = len(train_df.index)
    num_valid_samples = len(valid_df.index)
    num_train_batches = len(train_dl)
    num_valid_batches = len(valid_dl)
    stats = {
        'num_train_samples': num_train_samples,
        'num_valid_samples': num_valid_samples,
        'num_train_batches': num_train_batches,
        'num_valid_batches': num_valid_batches
    }
    with open(stats_pickle_path, 'wb') as f:
        pickle.dump(stats, f)
    print(f"Saved dataset sizes to {stats_pickle_path}")

print('Training samples:', num_train_samples)
print('Validation samples:', num_valid_samples)
print('Training batches:', num_train_batches)
print('Validation batches:', num_valid_batches)
print(f"Batch size: {par.batch_size}")

# Instantiate model
M_deepvo = StereoAdaptiveVO(
    img_h=par.img_h,
    img_w=par.img_w,
    batch_norm=par.batch_norm,
    input_channels=3,
    hidden_size=par.rnn_hidden_size,
    num_layers=2
)
use_cuda = torch.cuda.is_available()
if use_cuda:
    print('CUDA is available; using GPU.')
    M_deepvo = M_deepvo.cuda()
else:
    print('CUDA not available; using CPU.')

# Load pretrained weights if available
if par.pretrained_flownet and not par.resume:
    try:
        if use_cuda:
            pretrained_w = torch.load(par.pretrained_flownet, map_location='cuda')
        else:
            pretrained_w = torch.load(par.pretrained_flownet, map_location='cpu')
        print('Loading FlowNet pretrained model...')
        model_dict = M_deepvo.state_dict()
        update_dict = {k: v for k, v in pretrained_w['state_dict'].items() if k in model_dict}
        missing_keys = set(model_dict.keys()) - set(update_dict.keys())
        if missing_keys:
            print("Warning: Missing keys from pretrained checkpoint:", missing_keys)
        model_dict.update(update_dict)
        M_deepvo.load_state_dict(model_dict)
    except Exception as e:
        print("Error loading pretrained weights:", e)
else:
    print('Skipping FlowNet pretrained weights loading.')

optimizer = torch.optim.Adam(M_deepvo.parameters(), lr=par.optim['lr'], weight_decay=par.optim.get('weight_decay', 0))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=par.epochs, eta_min=1e-6)
scaler = GradScaler()

# Clear GPU cache
if use_cuda:
    torch.cuda.empty_cache()

print('Recording training metrics in:', par.record_path)
min_loss_t = 1e10
min_loss_v = 1e10
patience = 20
best_val_loss = min_loss_v
epochs_no_improve = 0

M_deepvo.train()
print(f"Training with climate sets: {par.climate_sets}")
print(f"Total epochs: {par.epochs}")

# Gradient accumulation settings
effective_batch_size = 24  # Desired effective batch size
accumulation_steps = effective_batch_size // par.batch_size  # Number of steps to accumulate gradients
print(f"Effective batch size: {effective_batch_size}, Accumulation steps: {accumulation_steps}")

epoch_times = []
total_start_time = time.time()

for ep in range(par.epochs):
    epoch_start_time = time.time()
    print('=' * 50)
    # Training Phase
    M_deepvo.train()
    t_loss_list = []
    accumulated_loss = 0
    optimizer.zero_grad()  # Clear gradients at the start of the epoch
    
    with tqdm(train_dl, desc=f"Epoch {ep+1}/{par.epochs} [Train]", unit="batch") as tbar:
        for batch_idx, (_, (t_x_03, t_x_02, t_x_depth, t_x_imu, t_x_gps), t_y) in enumerate(tbar):
            if use_cuda:
                t_x_03 = t_x_03.cuda(non_blocking=par.pin_mem)
                t_x_02 = t_x_02.cuda(non_blocking=par.pin_mem)
                t_x_depth = t_x_depth.cuda(non_blocking=par.pin_mem)
                t_x_imu = t_x_imu.cuda(non_blocking=par.pin_mem)
                t_x_gps = t_x_gps.cuda(non_blocking=par.pin_mem)
                t_y = t_y.cuda(non_blocking=par.pin_mem)
            
            # Debug shapes
            if batch_idx == 0:
                print(f"Epoch {ep+1}: t_x_03 shape: {t_x_03.shape}")
                print(f"Epoch {ep+1}: t_x_depth shape: {t_x_depth.shape}")
                print(f"Epoch {ep+1}: t_x_imu shape: {t_x_imu.shape}")
                print(f"Epoch {ep+1}: t_x_gps shape: {t_x_gps.shape}")
                print(f"Epoch {ep+1}: t_y shape: {t_y.shape}")
            
            # Forward pass with mixed precision
            with autocast(device_type='cuda', enabled=True):
                predicted = M_deepvo.forward((t_x_03, t_x_02, t_x_depth, t_x_imu, t_x_gps))
                total_loss = M_deepvo.get_loss((t_x_03, t_x_02, t_x_depth, t_x_imu, t_x_gps), t_y)
            
            # Debug predicted shape
            if batch_idx == 0:
                print(f"Epoch {ep+1}: predicted shape: {predicted.shape}")
            
            # Accumulate loss
            accumulated_loss += total_loss / accumulation_steps
            
            # Backpropagation with gradient accumulation
            scaler.scale(total_loss / accumulation_steps).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(M_deepvo.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                t_loss_list.append(float(accumulated_loss.item()))
                tbar.set_postfix({'loss': f"{accumulated_loss.item():.4f}"})
                accumulated_loss = 0
    
    train_time = time.time() - epoch_start_time
    print('Training phase completed in {:.1f} sec'.format(train_time))
    train_loss_mean = np.mean(t_loss_list)
    train_loss_std = np.std(t_loss_list)
    
    # Validation Phase
    M_deepvo.eval()
    v_loss_list = []
    valid_start_time = time.time()
    with torch.no_grad():
        with tqdm(valid_dl, desc=f"Epoch {ep+1}/{par.epochs} [Valid]", unit="batch") as vbar:
            for batch_idx, (_, (v_x_03, v_x_02, v_x_depth, v_x_imu, v_x_gps), v_y) in enumerate(vbar):
                if use_cuda:
                    v_x_03 = v_x_03.cuda(non_blocking=par.pin_mem)
                    v_x_02 = v_x_02.cuda(non_blocking=par.pin_mem)
                    v_x_depth = v_x_depth.cuda(non_blocking=par.pin_mem)
                    v_x_imu = v_x_imu.cuda(non_blocking=par.pin_mem)
                    v_x_gps = v_x_gps.cuda(non_blocking=par.pin_mem)
                    v_y = v_y.cuda(non_blocking=par.pin_mem)
                with autocast(device_type='cuda', enabled=True):
                    v_total_loss = M_deepvo.get_loss((v_x_03, v_x_02, v_x_depth, v_x_imu, v_x_gps), v_y)
                v_loss_val = float(v_total_loss.data.cpu().numpy())
                v_loss_list.append(v_loss_val)
                vbar.set_postfix({'loss': f"{v_loss_val:.4f}"})
    
    valid_time = time.time() - valid_start_time
    print('Validation phase completed in {:.1f} sec'.format(valid_time))
    valid_loss_mean = np.mean(v_loss_list)
    valid_loss_std = np.std(v_loss_list)
    
    # Print summary metrics in console
    print(f"Epoch {ep+1} Summary:")
    print(f"   Train Loss: Mean = {train_loss_mean:.4f}, Std = {train_loss_std:.4f}")
    print(f"   Validation Loss: Mean = {valid_loss_mean:.4f}, Std = {valid_loss_std:.4f}")
    
    scheduler.step()
    
    epoch_time = time.time() - epoch_start_time
    epoch_times.append(epoch_time)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {ep+1} completed in {epoch_time:.1f} sec, Learning Rate: {current_lr:.6f}")
    
    wandb.log({
        "epoch": ep + 1,
        "train_loss_mean": train_loss_mean,
        "train_loss_std": train_loss_std,
        "valid_loss_mean": valid_loss_mean,
        "valid_loss_std": valid_loss_std,
        "epoch_time": epoch_time,
        "learning_rate": current_lr,
    })
    
    with open(par.record_path, 'a') as f:
        f.write(f'\nEpoch {ep + 1}\nTrain Loss: Mean = {train_loss_mean:.4f}, Std = {train_loss_std:.4f}\n')
        f.write(f'Validation Loss: Mean = {valid_loss_mean:.4f}, Std = {valid_loss_std:.4f}\n')
        f.write(f'Learning Rate: {current_lr}\n')
    
    if valid_loss_mean < min_loss_v:
        min_loss_v = valid_loss_mean
        print(f"Validation loss improved at epoch {ep+1}; saving model...")
        torch.save(M_deepvo.state_dict(), par.save_model_path + '.valid')
        torch.save(optimizer.state_dict(), par.save_optimzer_path + '.valid')
    if train_loss_mean < min_loss_t:
        min_loss_t = train_loss_mean
        print(f"Training loss improved at epoch {ep+1}; saving model...")
        torch.save(M_deepvo.state_dict(), par.save_model_path + '.train')
        torch.save(optimizer.state_dict(), par.save_optimzer_path + '.train')
    
    if valid_loss_mean < best_val_loss:
        best_val_loss = valid_loss_mean
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {ep+1}: No improvement for {patience} epochs.")
            break

wandb.finish()
print("Training complete in {:.1f} sec".format(time.time() - total_start_time))