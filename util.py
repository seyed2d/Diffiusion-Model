%%writefile util.py

import os
import math
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Calc time embedding
def calc_t_emb(ts, t_emb_dim):
    device = ts.device
    half_dim = t_emb_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
    emb = ts[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return emb

# Flatten a nested list of lists [[1, 2], [3, 4, 5], [6]] into a single list [1, 2, 3, 4, 5, 6]
def flatten(l):
    return [item for sublist in l for item in sublist]

# Print model size
def print_size(net):
    n_params = sum(p.numel() for p in net.parameters())
    print(f"Model has {n_params / 1e6:.2f}M parameters")

# Training loss
def training_loss(net, criterion, T, X, Alpha_bar):
    B = X.size(0)
    ts = torch.randint(0, T, (B,), device=X.device).long()
    noise = torch.randn_like(X)
    alpha_bar = Alpha_bar[ts].view(B, 1, 1, 1)
    noisy_X = torch.sqrt(alpha_bar) * X + torch.sqrt(1 - alpha_bar) * noise
    pred_noise = net((noisy_X, ts))  # <--- اینجا اصلاح شد: ورودی به صورت تاپل
    return criterion(pred_noise, noise)

# Rescale image from [-1, 1] to [0, 1]
def rescale(x):
    return (x + 1) * 0.5

# Sampling process
@torch.no_grad()
def sampling(net, shape, T, Alpha, Alpha_bar, Sigma, device):
    B, C, H, W = shape
    x = torch.randn(shape, device=device)
    for t in tqdm(reversed(range(T)), desc='Sampling'):
        ts = torch.full((B,), t, device=device, dtype=torch.long)
        z = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        alpha = Alpha[t]
        alpha_bar = Alpha_bar[t]
        sigma = Sigma[t]
        pred_noise = net((x, ts))  # <--- اینجا هم اصلاح شد
        x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * pred_noise) + sigma * z
    return x

# Find max epoch checkpoint
def find_max_epoch(ckpt_path, name):
    max_epoch = -1
    for file in os.listdir(ckpt_path):
        if file.startswith(name):
            try:
                epoch = int(file.split('_')[-1].split('.')[0])
                if epoch > max_epoch:
                    max_epoch = epoch
            except:
                continue
    return max_epoch
