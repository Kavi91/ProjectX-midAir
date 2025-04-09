import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import pandas as pd
import pickle
from params import par
from model_2 import StereoAdaptiveVO
from data_helper import get_data_info, SortedRandomBatchSampler, ImageSequenceDataset, get_partition_data_info
import wandb
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast  # Import AMP modules

# Initialize WandB
wandb.init(project="deepvo_training_midair", config=vars(par))
wandb.config.update({"start_time": time.time()})

# Write all hyperparameters to record_path
mode = 'a' if par.resume else 'w'
with open(par.record_path, mode) as f:
    f.write('\n' + '=' * 50 + '\n')
    f.write('\n'.join("%s: %s" % item for item in vars(par).items()))
    f.write('\n' + '=' * 50 + '\n')

# Memory usage warning
if par.batch_size > 16:
    print(f"Warning: Batch size {par.batch_size} is large and may lead to high memory usage (both RAM and GPU VRAM).")
    print(f"Consider reducing batch size, num_workers, or enabling mixed precision training if memory issues occur.")

# Prepare Data
if os.path.isfile(par.train_data_info_path) and os.path.isfile(par.valid_data_info_path):
    print('Load data info from {}'.format(par.train_data_info_path))
    train_df = pd.read_pickle(par.train_data_info_path)
    valid_df = pd.read_pickle(par.valid_data_info_path)
else:
    print('Create new data info')
    if par.partition is not None:
        train_df, valid_df = get_partition_data_info(
            par.partition, par.climate_sets, par.seq_len, overlap=par.overlap, 
            sample_times=par.sample_times, shuffle=True, sort=True
        )
    else:
        train_df, valid_df = get_data_info(
            climate_sets=par.climate_sets, seq_len_range=par.seq_len, overlap=par.overlap,
            sample_times=par.sample_times, shuffle=True, sort=True, include_test=False
        )
    train_df.to_pickle(par.train_data_info_path)
    valid_df.to_pickle(par.valid_data_info_path)

train_sampler = SortedRandomBatchSampler(train_df, par.batch_size, drop_last=True)
train_dataset = ImageSequenceDataset(
    train_df, par.resize_mode, (par.img_h, par.img_w), par.img_means_03, par.img_stds_03,
    par.img_means_02, par.img_stds_02, par.minus_point_5
)
train_dl = DataLoader(
    train_dataset, batch_sampler=train_sampler, num_workers=24, pin_memory=par.pin_mem
)

valid_sampler = SortedRandomBatchSampler(valid_df, par.batch_size, drop_last=True)
valid_dataset = ImageSequenceDataset(
    valid_df, par.resize_mode, (par.img_h, par.img_w), par.img_means_03, par.img_stds_03,
    par.img_means_02, par.img_stds_02, par.minus_point_5
)
valid_dl = DataLoader(
    valid_dataset, batch_sampler=valid_sampler, num_workers=24, pin_memory=par.pin_mem
)

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
        print(f"Updated dataset statistics with sizes in {stats_pickle_path}")
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

print('Number of samples in training dataset: ', num_train_samples)
print('Number of samples in validation dataset: ', num_valid_samples)
print('Number of training batches: ', num_train_batches)
print('Number of validation batches: ', num_valid_batches)

# Model
M_deepvo = StereoAdaptiveVO(par.img_h, par.img_w, par.batch_norm)
use_cuda = torch.cuda.is_available()
if use_cuda:
    print('CUDA used.')
    M_deepvo = M_deepvo.cuda()
else:
    print('CUDA not used.')

# Load FlowNet weights
if par.pretrained_flownet and not par.resume:
    if use_cuda:
        pretrained_w = torch.load(par.pretrained_flownet, weights_only=True)
    else:
        pretrained_w = torch.load(par.pretrained_flownet, map_location='cpu', weights_only=True)
    print('Load FlowNet pretrained model')
    model_dict = M_deepvo.state_dict()
    update_dict = {k: v for k, v in pretrained_w['state_dict'].items() if k in model_dict}
    model_dict.update(update_dict)
    M_deepvo.load_state_dict(model_dict)
else:
    print('Skipping FlowNet model loading.')

# Create optimizer
if par.optim['opt'] == 'Adam':
    optimizer = torch.optim.Adam(M_deepvo.parameters(), lr=par.optim['lr'], weight_decay=par.optim.get('weight_decay', 0))
elif par.optim['opt'] == 'Adagrad':
    optimizer = torch.optim.Adagrad(M_deepvo.parameters(), lr=par.optim['lr'], weight_decay=par.optim.get('weight_decay', 0))
elif par.optim['opt'] == 'Cosine':
    optimizer = torch.optim.SGD(M_deepvo.parameters(), lr=par.optim['lr'], weight_decay=par.optim.get('weight_decay', 0))
    T_iter = par.optim['T'] * len(train_dl)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_iter, eta_min=0, last_epoch=-1)

# Learning rate schedulers
warmup_epochs = 5
lr_scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: min(1.0, (epoch + 1) / warmup_epochs))
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Load trained model and optimizer
if par.resume:
    M_deepvo.load_state_dict(torch.load(par.load_model_path, weights_only=True))
    optimizer.load_state_dict(torch.load(par.load_optimizer_path, weights_only=True))
    print('Load model from: ', par.load_model_path)
    print('Load optimizer from: ', par.load_optimizer_path)

# Initialize GradScaler for mixed precision training
scaler = GradScaler()

# Train
print('Record loss in: ', par.record_path)
min_loss_t = 1e10
min_loss_v = 1e10
M_deepvo.train()

total_start_time = time.time()
epoch_times = []

for ep in range(par.epochs):
    epoch_start_time = time.time()
    print('=' * 50)
    # Train
    M_deepvo.train()
    loss_mean = 0
    t_loss_list = []
    with tqdm(train_dl, desc=f"Epoch {ep+1}/{par.epochs} [Train]", unit="batch") as tbar:
        for batch_idx, (_, (t_x_03, t_x_02, t_x_depth, t_x_imu, t_x_gps), t_y) in enumerate(tbar):
            if use_cuda:
                t_x_03 = t_x_03.cuda(non_blocking=par.pin_mem)
                t_x_02 = t_x_02.cuda(non_blocking=par.pin_mem)
                t_x_depth = t_x_depth.cuda(non_blocking=par.pin_mem)
                t_x_imu = t_x_imu.cuda(non_blocking=par.pin_mem)
                t_x_gps = t_x_gps.cuda(non_blocking=par.pin_mem)
                t_y = t_y.cuda(non_blocking=par.pin_mem)
            
            if ep == 0 and batch_idx == 0:
                print(f"Input shapes - t_x_03: {t_x_03.shape}, t_x_02: {t_x_02.shape}, ...")
            
            # Clear gradients before each step
            optimizer.zero_grad()
            
            # Use autocast to enable mixed precision for the forward pass
            with autocast():
                ls = M_deepvo.step((t_x_03, t_x_02, t_x_depth, t_x_imu, t_x_gps), t_y, optimizer)
            
            # Scale the loss and perform backpropagation with GradScaler
            scaler.scale(ls).backward()
            
            # Apply gradient clipping if specified
            if M_deepvo.clip is not None:
                scaler.unscale_(optimizer)  # Unscale gradients before clipping
                torch.nn.utils.clip_grad_norm_(M_deepvo.rnn.parameters(), M_deepvo.clip)
            
            # Update optimizer and scaler
            scaler.step(optimizer)
            scaler.update()
            
            ls = ls.data.cpu().numpy()
            t_loss_list.append(float(ls))
            loss_mean += float(ls)
            tbar.set_postfix({'loss': f"{float(ls):.4f}"})
    train_time = time.time() - epoch_start_time
    print('Train take {:.1f} sec'.format(train_time))
    loss_mean /= len(train_dl)

    # Validation
    valid_start_time = time.time()
    M_deepvo.eval()
    loss_mean_valid = 0
    v_loss_list = []
    with tqdm(valid_dl, desc=f"Epoch {ep+1}/{par.epochs} [Valid]", unit="batch") as vbar:
        for batch_idx, (_, (v_x_03, v_x_02, v_x_depth, v_x_imu, v_x_gps), v_y) in enumerate(vbar):
            if use_cuda:
                v_x_03 = v_x_03.cuda(non_blocking=par.pin_mem)
                v_x_02 = v_x_02.cuda(non_blocking=par.pin_mem)
                v_x_depth = v_x_depth.cuda(non_blocking=par.pin_mem)
                v_x_imu = v_x_imu.cuda(non_blocking=par.pin_mem)
                v_x_gps = v_x_gps.cuda(non_blocking=par.pin_mem)
                v_y = v_y.cuda(non_blocking=par.pin_mem)
            # Use autocast for validation as well
            with autocast():
                v_ls = M_deepvo.get_loss((v_x_03, v_x_02, v_x_depth, v_x_imu, v_x_gps), v_y)
            v_ls = v_ls.data.cpu().numpy()
            v_loss_list.append(float(v_ls))
            loss_mean_valid += float(v_ls)
            vbar.set_postfix({'loss': f"{float(v_ls):.4f}"})
    valid_time = time.time() - valid_start_time
    print('Valid take {:.1f} sec'.format(valid_time))
    loss_mean_valid /= len(valid_dl)

    # ETA calculation
    epoch_time = time.time() - epoch_start_time
    epoch_times.append(epoch_time)
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    remaining_epochs = par.epochs - (ep + 1)
    eta_minutes = (avg_epoch_time * remaining_epochs) / 60
    print(f"Epoch {ep+1}/{par.epochs} completed in {epoch_time:.1f} sec, ETA: {eta_minutes:.1f} minutes")

    # Get current learning rate
    current_lr = optimizer.param_groups[0]['lr']

    # Log to WandB
    wandb.log({
        "epoch": ep + 1,
        "train_loss_mean": loss_mean,
        "train_loss_std": np.std(t_loss_list),
        "valid_loss_mean": loss_mean_valid,
        "valid_loss_std": np.std(v_loss_list),
        "epoch_time": epoch_time,
        "eta_minutes": eta_minutes,
        "learning_rate": current_lr,
    })

    with open(par.record_path, 'a') as f:
        f.write(f'Epoch {ep + 1}\ntrain loss mean: {loss_mean}, std: {np.std(t_loss_list):.2f}\nvalid loss mean: {loss_mean_valid}, std: {np.std(v_loss_list):.2f}\nlearning rate: {current_lr}\n')

    # Save model
    check_interval = 1
    if loss_mean_valid < min_loss_v and ep % check_interval == 0:
        min_loss_v = loss_mean_valid
        print(f'Save model at ep {ep + 1}, mean of valid loss: {loss_mean_valid}')
        torch.save(M_deepvo.state_dict(), par.save_model_path + '.valid')
        torch.save(optimizer.state_dict(), par.save_optimzer_path + '.valid')
    if loss_mean < min_loss_t and ep % check_interval == 0:
        min_loss_t = loss_mean
        print(f'Save model at ep {ep + 1}, mean of train loss: {loss_mean}')
        torch.save(M_deepvo.state_dict(), par.save_model_path + '.train')
        torch.save(optimizer.state_dict(), par.save_optimzer_path + '.train')

    # Apply schedulers
    if ep < warmup_epochs:
        lr_scheduler_warmup.step()
        print(f"Warmup Epoch {ep+1}: Learning rate adjusted to {current_lr}")
    else:
        lr_scheduler.step(loss_mean_valid)
        print(f"Epoch {ep+1}: Learning rate adjusted to {current_lr} based on validation loss")

wandb.finish()