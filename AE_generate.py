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

args = parser.parse_args()
Del=0.1
fold=0
device=f'cuda:{args.device}'
dataname=args.dataname
view=args.view

n_epoch=500
lrate=1e-4


def process_data(dataname='carl', view=0,Del=0.1,fold=0):
    
    z=np.load(f"./data/ddpm_carl_view{view}_pairedrate{Del}_fold{fold}.npy")
    x_tensor = torch.tensor(z, dtype=torch.float32)
    print(x_tensor)
    x_dataset = TensorDataset(x_tensor)
    dataloader = DataLoader(x_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)
    return dataloader
    


dataloader=process_data(dataname=dataname, view=view)

for batch in dataloader:
    x_batch = batch[0]  # Extract the first item in the batch tuple, which is x_train_tensor
    input_dim = x_batch.shape[1]
    break  # Exit after the first batch

hidden_dims = [512, 256, 128]  # Example hidden layer dimensions

net = AE(input_dim=578, hidden_dims=hidden_dims)
net = net.to(device)
net.load_state_dict(torch.load(f"./models/AE_{dataname}_view{view}_ep500.pth",map_location=device))
# net=torch.compile(net)
# optim = torch.optim.Adam(net.parameters(), lr=lrate)

net.eval()
pbar=tqdm(dataloader)
with torch.no_grad():
    out=[]
    for z, in pbar:
        z=z.to(device)
        x_rec = net.forward_x_rec(z)
        out.append(x_rec.cpu())
    out=torch.cat(out,dim=0).numpy()
    np.save(f"./data/generate_{dataname}_view{view}_del{Del}_fold{fold}.npy", out)