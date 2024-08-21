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

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0, help='device')
parser.add_argument('--view', type=int, default=0)
args = parser.parse_args()

device=f'cuda:{args.device}'
view=args.view

n_epoch=1000
n_T=1000
lrate=1e-3
betas=(1e-6, 2e-2) # betas=(1e-4, 2e-2)
n_feat=128

train_dataloader, val_dataloader, test_dataloader, configs=get_data(view=view)

ddpm=DDPM(
    nn_model=Autoencoder(n_feat, feature_dim=configs['dim_c']),
    betas=betas, n_T=n_T, device=device
    )
ddpm=ddpm.to(device)
optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

milestones = [int(n_epoch * 0.25), int(n_epoch * 0.5), int(n_epoch * 0.75), int(n_epoch * 0.9)]
scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=milestones, gamma=0.1)


for ep in range(n_epoch):
    print(f'epoch {ep}')

    ddpm.train()  # training mode
    pbar = tqdm(train_dataloader)
    loss_ema = None
    for x,c in pbar:
        optim.zero_grad()
        x=x.to(device)
        c = c.to(device)
        loss = ddpm(x,c, ep)
        loss.backward()
        if loss_ema is None:
            loss_ema = loss.sum().item()
        else:
            loss_ema = 0.95 * loss_ema + 0.05 * loss.sum().item()
        pbar.set_description(f"loss: {loss_ema:.4f}")
        optim.step()

    if (ep+1)%100 == 0:
        # Validate the model and return validation loss
        ddpm.eval()
        val_loss = 0
        with torch.no_grad():
            for x, c in val_dataloader:
                x = x.to(device)
                c = c.to(device)
                loss = ddpm(x, c, ep)
                val_loss += loss.sum().item()
            print(f"  Validation loss: {val_loss:.4f}")
        
        torch.save(ddpm.state_dict(), f"./models/ddpm_view{view}_ep{ep+1}.pth")
    
    scheduler.step()
    # if (ep+1)%1000==0:
    #     torch.save(ddpm.state_dict(), f"./models/ddpm_view{view}_ep{ep+1}.pth")
