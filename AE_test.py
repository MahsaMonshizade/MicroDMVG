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
parser.add_argument('--view', type=int, default=2)

args = parser.parse_args()

device=f'cuda:{args.device}'
dataname=args.dataname
view=args.view

n_epoch=500
lrate=1e-4


def process_data(dataname='carl', view=2):
    if view == 0:
        data = io.loadmat(f"./metagenomics_data.mat")
    elif view == 1:
        data = io.loadmat(f"./metabolomics_data.mat")
    elif view == 2:
        data = io.loadmat(f"./metatranscriptomics_data.mat")
    
    x = data['metatranscriptomics_X']
    
    x_tensor = torch.tensor(x, dtype=torch.float32)
    x_dataset = TensorDataset(x_tensor)
    dataloader = DataLoader(x_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)
    return dataloader
    


dataloader=process_data(dataname=dataname, view=view)

for batch in dataloader:
    x_batch = batch[0]  # Extract the first item in the batch tuple, which is x_train_tensor
    input_dim = x_batch.shape[1]
    break  # Exit after the first batch

hidden_dims = [512, 256, 128]  # Example hidden layer dimensions

net = AE(input_dim=input_dim, hidden_dims=hidden_dims)
net = net.to(device)
net.load_state_dict(torch.load(f"./models/AE_{dataname}_view{view}_ep500.pth",map_location=device))
# net=torch.compile(net)
# optim = torch.optim.Adam(net.parameters(), lr=lrate)

net.eval()
pbar=tqdm(dataloader)
with torch.no_grad():
    out=[]
    loss_ema = None
    for x, in pbar:
        x=x.to(device)
        x_rec,z = net(x)
        loss=ae_mse_loss(x_rec,x)
        if loss_ema is None:
            loss_ema = loss.sum().item()
        else:
            loss_ema = 0.95 * loss_ema + 0.05 * loss.sum().item()
        pbar.set_description(f"loss: {loss_ema:.4f}")
        out.append(z.cpu())
    out=torch.cat(out,dim=0).numpy()
    np.save(f"./data/AE_{dataname}_view{view}.npy", out)
