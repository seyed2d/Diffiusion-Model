import os
import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image

from load_data import load_UTKFace
from util import training_loss, sampling, rescale, find_max_epoch, print_size
from UNet import UNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(output_directory, ckpt_epoch, n_epochs, learning_rate, batch_size,
          T, beta_0, beta_T, unet_config):

    Beta = torch.linspace(beta_0, beta_T, T).to(device)
    Alpha = 1 - Beta
    Alpha_bar = torch.ones(T).to(device)
    Beta_tilde = Beta.clone()
    for t in range(T):
        Alpha_bar[t] *= Alpha[t] * Alpha_bar[t-1] if t else Alpha[t]
        if t > 0:
            Beta_tilde[t] *= (1 - Alpha_bar[t-1]) / (1 - Alpha_bar[t])
    Sigma = torch.sqrt(Beta_tilde)

    trainloader = load_UTKFace(batch_size=batch_size)
    print('Data loaded')

    net = UNet(**unet_config).to(device)
    print_size(net)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    time0 = time.time()
    if ckpt_epoch == 'max':
        ckpt_epoch = find_max_epoch(output_directory, 'unet_ckpt')
    if ckpt_epoch >= 0:
        model_path = os.path.join(output_directory, f'unet_ckpt_{ckpt_epoch}.pkl')
        checkpoint = torch.load(model_path, map_location=device)
        print(f'Model at epoch {ckpt_epoch} has been trained for {checkpoint["training_time_seconds"]} seconds')
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        time0 -= checkpoint['training_time_seconds']
        print('Checkpoint model loaded successfully')
    else:
        ckpt_epoch = -1
        print('No valid checkpoint found. Training from scratch.')

    for epoch in range(ckpt_epoch + 1, n_epochs):
        for i, (X, _) in enumerate(trainloader):
            X = X.to(device)
            optimizer.zero_grad()
            loss = training_loss(net, nn.MSELoss(), T, X, Alpha_bar)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"epoch: {epoch}, iter: {i}, loss: {loss.item():.7f}", flush=True)

        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'training_time_seconds': int(time.time()-time0)
            }, os.path.join(output_directory, f'unet_ckpt_{epoch}.pkl'))
            print(f'model at epoch {epoch} is saved')
            # ---- Generate and save some images during training ----
            net.eval()
            with torch.no_grad():
                X_gen = sampling(net, (4, 3, 64, 64), T, Alpha, Alpha_bar, Sigma, device)
                for j in range(4):
                    save_path = os.path.join(output_directory, f'sample_epoch{epoch}_img{j}.jpg')
                    save_image(rescale(X_gen[j]), save_path)
                print(f'Saved 4 sample images at epoch {epoch}')
            net.train()



def generate(output_directory, ckpt_path, ckpt_epoch, n,
             T, beta_0, beta_T, unet_config):

    Beta = torch.linspace(beta_0, beta_T, T).to(device)
    Alpha = 1 - Beta
    Alpha_bar = torch.ones(T).to(device)
    Beta_tilde = Beta.clone()
    for t in range(T):
        Alpha_bar[t] *= Alpha[t] * Alpha_bar[t-1] if t else Alpha[t]
        if t > 0:
            Beta_tilde[t] *= (1 - Alpha_bar[t-1]) / (1 - Alpha_bar[t])
    Sigma = torch.sqrt(Beta_tilde)

    net = UNet(**unet_config).to(device)
    print_size(net)

    if ckpt_epoch == 'max':
        ckpt_epoch = find_max_epoch(ckpt_path, 'unet_ckpt')
    model_path = os.path.join(ckpt_path, f'unet_ckpt_{ckpt_epoch}.pkl')
    checkpoint = torch.load(model_path, map_location=device)
    print(f'Model at epoch {ckpt_epoch} has been trained for {checkpoint["training_time_seconds"]} seconds')
    net.load_state_dict(checkpoint['model_state_dict'])

    time0 = time.time()
    X_gen = sampling(net, (n, 3, 64, 64), T, Alpha, Alpha_bar, Sigma, device)
    print(f'Generated {n} samples at epoch {ckpt_epoch} in {int(time.time()-time0)} seconds')

    for i in range(n):
        save_image(rescale(X_gen[i]), os.path.join(output_directory, f'img_{i}.jpg'))
    print(f'Saved generated samples at epoch {ckpt_epoch}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.json',
                        help='JSON file for configuration')
    parser.add_argument('-t', '--task', type=str, choices=['train', 'generate'],
                        help='Run either training or generation')
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    unet_config      = config["unet_config"]
    diffusion_config = config["diffusion_config"]
    train_config     = config["train_config"]
    gen_config       = config["gen_config"]

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    if args.task == 'train':
        train(**train_config, **diffusion_config, unet_config=unet_config)
    elif args.task == 'generate':
        generate(**gen_config, **diffusion_config, unet_config=unet_config)
    else:
        raise Exception("Invalid task.")