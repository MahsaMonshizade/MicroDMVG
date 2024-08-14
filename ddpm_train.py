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
parser.add_argument('--dataname', type=str, default='carl')
parser.add_argument('--view', type=int, default=0)
parser.add_argument('--pairedrate', type=float, default=0.1)
parser.add_argument('--fold', type=int, default=0)
args = parser.parse_args()

# device=f'cuda:{args.device}'
# dataname=args.dataname
# view=args.view
# pairedrate=args.pairedrate
# fold=args.fold

# n_epoch=1000
# n_T=1000
# lrate=1e-3
# betas=(1e-6, 2e-2) # betas=(1e-4, 2e-2)
# drop_prob=0.1
# n_feat=512

# train_dataloader, test_dataloader, configs=get_data(dataname=dataname, view=view)

# ddpm=DDPM(
#     nn_model=Autoencoder(n_feat, feature_dim=configs['dim_c']),
#     betas=betas, n_T=n_T, device=device, drop_prob=drop_prob
#     )
# ddpm=ddpm.to(device)
# # net=torch.compile(net)
# optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

# milestones = [int(n_epoch * 0.25), int(n_epoch * 0.5), int(n_epoch * 0.75), int(n_epoch * 0.9)]
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=milestones, gamma=0.1)


# for ep in range(n_epoch):
#     print(f'epoch {ep}')

#     ddpm.train()  # training mode
#     # linear lrate decay
#     # optim.param_groups[0]['lr'] = lrate * (1 - ep / n_epoch)
#     pbar = tqdm(train_dataloader)
#     loss_ema = None
#     for x,c in pbar:
#         optim.zero_grad()
#         x=x.to(device)
#         c = c.to(device)
#         # print(x.shape)
#         # print('hi')
#         # print(c.shape)
#         loss = ddpm(x,c, ep)
#         loss.backward()
#         if loss_ema is None:
#             loss_ema = loss.sum().item()
#         else:
#             loss_ema = 0.95 * loss_ema + 0.05 * loss.sum().item()
#         pbar.set_description(f"loss: {loss_ema:.4f}")
#         optim.step()
#         # Access gradients of model parameters
#         # for name, param in ddpm.named_parameters():
#         #     if param.grad is not None:
#         #         print(f"Gradient for {name}: {param.grad.mean().item()} (mean), {param.grad.std().item()} (std)")
#         #     else:
#         #         print(f"No gradient for {name}")
#     scheduler.step()
#     if (ep+1)%1000==0:
#         torch.save(ddpm.state_dict(), f"./models/ddpm_{dataname}_view{view}_pairedrate{pairedrate}_fold{fold}_ep{ep+1}.pth")



# Setting up device and dataset configurations
device = f'cuda:{args.device}'
dataname = args.dataname
view = args.view
pairedrate = args.pairedrate
fold = args.fold

# Fixed parameters for the experiment
n_T = 1000
drop_prob = 0.1
n_feat = 512

# Load your dataset
train_dataloader, test_dataloader, configs = get_data(dataname=dataname, view=view)

# Define the objective function for Optuna
def objective(trial):
    # Suggest hyperparameters for optimization
    lrate = trial.suggest_loguniform('lrate', 1e-4, 1e-2)
    beta1 = trial.suggest_uniform('beta1', 1e-6, 1e-6)
    beta2 = trial.suggest_uniform('beta2', 2e-2, 2e-2)
    n_epoch = trial.suggest_int('n_epoch', 250, 500)
    
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
    train_loss = 10000
    best_epoch = 0
    for ep in range(n_epoch):
        ddpm.train()
        # pbar = tqdm(train_dataloader)
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
        if loss_ema<train_loss:
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
    
    # Print trial information
    print(f"Trial {trial.number} finished")
    print(f"  Hyperparameters: lrate={lrate}, betas=({beta1}, {beta2}), n_epoch={n_epoch}")
    print(f"  Validation loss: {val_loss:.4f}")
    print(f"  train loss: {train_loss:.4f}")
    print(f"  train epoch: {best_epoch:.4f}")
    
    return val_loss

# Run the Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

# Print the best trial
print("Best trial:")
trial = study.best_trial

print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")