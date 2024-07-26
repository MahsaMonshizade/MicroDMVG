import scipy.io as io
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchvision.utils import save_image
from tqdm import tqdm
import argparse
from einops import rearrange
from model import *
from datasets import *

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0, help='device')
parser.add_argument('--dataname', type=str, default='carl')
parser.add_argument('--view', type=int, default=2)
args = parser.parse_args()

print(torch.backends.cudnn.enabled)
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"PyTorch version: {torch.__version__}")

device = f'cuda:{args.device}'
print(torch.cuda.is_available())
dataname = args.dataname
view = args.view

n_epoch = 500
lrate = 1e-4

def process_data(dataname='carl', view=2):
    if view == 0:
        data = io.loadmat(f"./metagenomics_data.mat")
    elif view == 1:
        data = io.loadmat(f"./metabolomics_data.mat")
    elif view == 2:
        data = io.loadmat(f"./metatranscriptomics_data.mat")

    available_omics = io.loadmat(f"./available_omics.mat")['available_omics']
    x = data['metatranscriptomics_X']
    mask = torch.tensor(available_omics)
    
    ind_train = mask[:, view] == 1
    ind_test = mask[:, view] == 0
    x_train = x[ind_train]
    x_test = x[ind_test]

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)

    # Split the training data into training and validation sets
    train_size = int(0.8 * len(x_train_tensor))
    val_size = len(x_train_tensor) - train_size
    train_dataset, val_dataset = random_split(TensorDataset(x_train_tensor), [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)
    test_dataset = TensorDataset(x_test_tensor)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)
    
    return train_dataloader, val_dataloader, test_dataloader

train_dataloader, val_dataloader, test_dataloader = process_data(dataname=dataname, view=view)

for batch in train_dataloader:
    x_batch = batch[0]  # Extract the first item in the batch tuple, which is x_train_tensor
    input_dim = x_batch.shape[1]
    break  # Exit after the first batch

hidden_dims = [512, 256, 128]  # Example hidden layer dimensions

net = AE(input_dim=input_dim, hidden_dims=hidden_dims)
net = net.to(device)
optim = torch.optim.Adam(net.parameters(), lr=lrate)

for ep in range(n_epoch):
    print(f'epoch {ep}')

    net.train()  # training mode
    # linear lrate decay
    optim.param_groups[0]['lr'] = lrate * (1 - ep / n_epoch)
    pbar = tqdm(train_dataloader)
    loss_ema = None
    for x, in pbar:
        optim.zero_grad()
        x = x.to(device)
        x_rec, z = net(x)
        loss = ae_mse_loss(x_rec, x)
        loss.backward()
        if loss_ema is None:
            loss_ema = loss.sum().item()
        else:
            loss_ema = 0.95 * loss_ema + 0.05 * loss.sum().item()
        pbar.set_description(f"loss: {loss_ema:.4f}")
        optim.step()
    if (ep + 1) % 100 == 0:
        torch.save(net.state_dict(), f"./models/AE_{dataname}_view{view}_ep{ep+1}.pth")

        # Validation step
        net.eval()  # evaluation mode
        val_loss = 0
        with torch.no_grad():
            for x, in val_dataloader:
                x = x.to(device)
                x_rec, z = net(x)
                loss = ae_mse_loss(x_rec, x)
                val_loss += loss.item()
        val_loss /= len(val_dataloader)
        print(f'Validation loss: {val_loss:.4f}')
