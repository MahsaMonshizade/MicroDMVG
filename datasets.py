from einops import rearrange, repeat
import scipy.io as io
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def get_data(dataname='carl', view=0):   
    if dataname=='carl':
       
        dims=[128, 128, 128]
        configs={}
        configs['dim_c']=sum(dims)-dims[view]

        # data=io.loadmat(f"./data/carl.mat")
        available_omics = io.loadmat(f"./available_omics.mat")['available_omics']
        x0=np.load(f"./data/AE_carl_view0.npy")
        x1=np.load(f"./data/AE_carl_view1.npy")
        x2=np.load(f"./data/AE_carl_view2.npy")


        x0 = torch.tensor(x0, dtype=torch.float32)
        x1 = torch.tensor(x1, dtype=torch.float32)
        x2 = torch.tensor(x2, dtype=torch.float32)
        x=[x0,x1,x2]

        mask = torch.tensor(available_omics)
        ind_train=mask[:,view]==1
        ind_test=mask[:,view]==0

        x_train=x[view][ind_train]
        c_train=[]
        for i in range(len(x)):
            if i!=view:
                c_train.append((x[i] * mask[:,i].view(-1,1))[ind_train])

        
        c_train=torch.cat(c_train, dim=-1)

        x_test=x[view][ind_test]
        c_test=[]
        for i in range(len(x)):
            if i!=view:
                c_test.append((x[i] * mask[:,i].view(-1,1))[ind_test])
        c_test=torch.cat(c_test, dim=-1)
        train_dataset=TensorDataset(x_train, c_train)
        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
        test_dataset=TensorDataset(x_test, c_test)
        test_dataloader=DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)
        return train_dataloader, test_dataloader, configs

if __name__=="__main__":
    get_data()  