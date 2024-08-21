import scipy.io as io
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import argparse
from einops import rearrange, repeat
from model import *
from datasets import *

import optuna

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0, help='device')
parser.add_argument('--view', type=int, default=0)
args = parser.parse_args()

# Setting up device and dataset configurations
device = f'cuda:{args.device}'
view = args.view

# Fixed parameters for the experiment
n_T = 1000
drop_prob = 0.1
n_feat = 128

# Load your dataset
train_dataloader, test_dataloader, configs = get_data(view=view)

# Variables to store the best hyperparameters and losses
best_params = None
best_train_loss = float('inf')
best_val_loss = float('inf')
best_epoch = 0

# Define the objective function for Optuna
def objective(trial):
    global best_params, best_train_loss, best_val_loss, best_epoch
    
    # Suggest hyperparameters for optimization
    lrate = trial.suggest_loguniform('lrate', 1e-4, 1e-1)
    beta1 = trial.suggest_uniform('beta1', 1e-6, 1e-4)
    beta2 = trial.suggest_uniform('beta2', 1e-2, 2e-1)
    n_epoch = trial.suggest_int('n_epoch', 100, 1000)
    
    betas = (beta1, beta2)
    
    # Initialize model, optimizer, and scheduler with suggested hyperparameters
    ddpm = DDPM(
        nn_model=Autoencoder(n_feat, feature_dim=configs['dim_c']),
        betas=betas, n_T=n_T, device=device, drop_prob=drop_prob
    ).to(device)
    
    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)
    milestones = [int(n_epoch * 0.25), int(n_epoch * 0.5), int(n_epoch * 0.75), int(n_epoch * 0.9)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=milestones, gamma=0.1)
    
    # Training loop
    train_loss = float('inf')
    for ep in range(n_epoch):
        ddpm.train()
        loss_ema = None
        for x, c in tqdm(train_dataloader, desc=f"Epoch {ep}/{n_epoch}"):
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device)
            loss = ddpm(x, c, ep)
            loss.backward()
            optim.step()
            if loss_ema is None:
                loss_ema = loss.sum().item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.sum().item()
        scheduler.step()
        
        if loss_ema < train_loss:
            train_loss = loss_ema
            best_epoch = ep
    
    # Validate the model and return validation loss
    ddpm.eval()
    val_loss = 0
    with torch.no_grad():
        for x, c in test_dataloader:
            x = x.to(device)
            c = c.to(device)
            loss = ddpm(x, c, ep)
            val_loss += loss.sum().item()
    
    # Check if this is the best trial so far
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_train_loss = train_loss
        best_params = trial.params

    # Print trial information
    print(f"Trial {trial.number} finished")
    print(f"  Hyperparameters: lrate={lrate}, betas=({beta1}, {beta2}), n_epoch={n_epoch}")
    print(f"  Validation loss: {val_loss:.4f}")
    print(f"  Training loss: {train_loss:.4f}")
    print(f"  Best epoch: {best_epoch}")
    
    return val_loss

# Run the Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# Print and save the best trial results
print("Best trial:")
trial = study.best_trial

print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Save the best hyperparameters and losses
with open('best_hyperparameters.txt', 'w') as f:
    f.write(f"Best Validation Loss: {best_val_loss:.4f}\n")
    f.write(f"Best Training Loss: {best_train_loss:.4f}\n")
    f.write(f"Best Epoch: {best_epoch}\n")
    f.write("Best Hyperparameters:\n")
    for key, value in best_params.items():
        f.write(f"  {key}: {value}\n")
