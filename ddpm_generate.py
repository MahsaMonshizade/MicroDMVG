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
parser.add_argument('--dataname', type=str, default='carl')
parser.add_argument('--view', type=int, default=0)
parser.add_argument('--pairedrate', type=float, default=0.1)
parser.add_argument('--fold', type=int, default=0)
args = parser.parse_args()

device=f'cuda:{args.device}'
dataname=args.dataname
view=args.view
pairedrate=args.pairedrate
fold=args.fold

n_epoch=1000
n_T=1000
lrate=5e-3
betas=(1e-5, 1e-1) # betas=(1e-4, 2e-2)
drop_prob=0.1
n_feat=512

train_dataloader, test_dataloader, configs=get_data(dataname=dataname, view=view)

ddpm=DDPM(
    nn_model=Autoencoder(n_feat, feature_dim=configs['dim_c']),
    betas=betas, n_T=n_T, device=device
    )
ddpm=ddpm.to(device)
ddpm.load_state_dict(torch.load(f"./models/ddpm_{dataname}_view{view}_pairedrate{pairedrate}_fold{fold}_ep1000.pth",map_location=device))


ddpm.eval()
pbar=tqdm(test_dataloader)
with torch.no_grad():
    out=[]
    for x,c in pbar:
        x=x.to(device)
        c=c.to(device)
        x_rec=ddpm.ddpm_sample(c=c, n_sample=1, size=x.shape[1], device=device)
        out.append(x_rec.cpu())
    out=torch.cat(out,dim=0).numpy()
    np.save(f"./data/ddpm_{dataname}_view{view}_pairedrate{pairedrate}_fold{fold}.npy",out)