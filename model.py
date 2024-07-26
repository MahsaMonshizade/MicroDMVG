# %% [1]import
import torch
import torch.nn as nn
import numpy as np
from torch.nn import init

# %% [2]AE
def Normalize(in_channels):
    return nn.GroupNorm(num_groups=8, num_channels=in_channels, eps=1e-5, affine=True)
    # return nn.BatchNorm1d(in_channels)
    # return nn.BatchNorm2d(in_channels)

def ae_mse_loss(recon_x, x):   
    MSE = nn.functional.mse_loss(recon_x, x, reduction='mean')
    # BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='mean')
    return MSE
def ae_bce_loss(recon_x, x):   
    # MSE = nn.functional.mse_loss(recon_x, x, reduction='mean')
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='mean')
    return BCE


class AE(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(AE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # Encoder layers
        encoder_layers = []
        for i in range(len(hidden_dims)):
            if i == 0:
                encoder_layers.append(nn.Linear(input_dim, hidden_dims[i]))
            else:
                encoder_layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            
            # Add Batch Normalization
            encoder_layers.append(nn.BatchNorm1d(hidden_dims[i]))
            
            encoder_layers.append(nn.Tanh())
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder layers
        decoder_layers = []
        for i in range(len(hidden_dims)-1, 0, -1):
            decoder_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i-1]))
            
            # Add Batch Normalization
            decoder_layers.append(nn.BatchNorm1d(hidden_dims[i-1]))
            
            decoder_layers.append(nn.ReLU())
        
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        decoder_layers.append(nn.Sigmoid())
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec, z


# %% [3]diffusion
class Embed(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super().__init__()
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )
    # def initialize(self):
    #     for module in self.modules():
    #         if isinstance(module, nn.Linear):
    #             init.xavier_uniform_(module.weight)
    #             init.zeros_(module.bias)

    def forward(self, x):
        return self.model(x.reshape(-1, self.input_dim))


class Autoencoder(nn.Module):
    def __init__(self, n_feat, feature_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(n_feat, n_feat),   # Input layer to middle dimension
            nn.BatchNorm1d(n_feat),  # Batch normalization
            nn.GELU(),           # Activation function
            nn.Linear(n_feat, 2*n_feat),   # Input layer to middle dimension
            nn.BatchNorm1d(2*n_feat),  # Batch normalization
            nn.GELU(),           # Activation function
            nn.Linear(2*n_feat, 2**2 * n_feat),   # Input layer to middle dimension
            nn.BatchNorm1d(2**2 * n_feat),  # Batch normalization
            nn.GELU(),           # Activation function
            nn.Linear(2**2 * n_feat, 2**3 * n_feat),  # Middle dimension to another feature dimension
            nn.BatchNorm1d(2**3 * n_feat), # Batch normalization
            nn.GELU(),           # Activation function
        )

        self.vec2vec = nn.Linear(2**3 * n_feat + feature_dim, 2**3 * n_feat)
        self.bn = nn.BatchNorm1d(2**3 * n_feat)  # Batch normalization
        self.linear1 = nn.Linear(2**4 * n_feat, 2**2 * n_feat)
        self.bn1 = nn.BatchNorm1d(2**2 * n_feat)  # Batch normalization
        self.linear2 = nn.Linear(2**3 * n_feat, 2 * n_feat)
        self.bn2 = nn.BatchNorm1d(2 * n_feat)  # Batch normalization
        self.linear3 = nn.Linear(2**2 * n_feat, n_feat)
        # self.bn3 = nn.BatchNorm1d(n_feat)   # Batch normalization

        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        # self.sigmoid = nn.Sigmoid()

        temb = [Embed(1, 2**(i+1)*n_feat) for i in range(3)]
        self.temb = nn.ModuleList(temb)
    
    def forward(self, x, c, t):
        temb = [self.temb[i](t)[:, :] for i in range(3)]
        c = c.to(torch.float32)
        x = self.encoder(x)

        x = self.vec2vec(torch.cat([x, c], dim=1))
        x = self.bn(x)  # Apply batch normalization
        x = self.relu(x)

        x = self.linear1(torch.cat([x, temb[2]], dim=1))
        x = self.bn1(x)  # Apply batch normalization
        x = self.gelu(x)
        
        x = self.linear2(torch.cat([x, temb[1]], dim=1))
        x = self.bn2(x)  # Apply batch normalization
        x = self.gelu(x)
        
        x = self.linear3(torch.cat([x, temb[0]], dim=1))
        # x = self.bn3(x)  # Apply batch normalization
        # x = self.relu(x)
        
        return x
    

# class Autoencoder(nn.Module):
#     def __init__(self, n_feat, feature_dim):
#         super(Autoencoder, self).__init__()
#         # Encoder
#         self.encoder = nn.Sequential(
#             nn.Linear(n_feat, n_feat),   # Input layer to middle dimension
#             nn.BatchNorm1d(n_feat),  # Batch normalization
#             nn.SiLU(),           # Activation function
#             nn.Linear(n_feat, 2*n_feat),   # Input layer to middle dimension
#             nn.BatchNorm1d(2*n_feat),  # Batch normalization
#             nn.SiLU(),           # Activation function
#             nn.Linear(2*n_feat, 2**2 * n_feat),   # Input layer to middle dimension
#             nn.BatchNorm1d(2**2 * n_feat),  # Batch normalization
#             nn.SiLU(),           # Activation function
#             nn.Linear(2**2 * n_feat, 2**3 * n_feat),  # Middle dimension to another feature dimension
#             nn.BatchNorm1d(2**3 * n_feat), # Batch normalization
#             nn.SiLU(),           # Activation function
#         )

#         self.vec2vec = nn.Linear(2**3 * n_feat, 2**3 * n_feat)
#         self.bn = nn.BatchNorm1d(2**3 * n_feat)  # Batch normalization
#         self.linear1 = nn.Linear(2**4 * n_feat, 2**2 * n_feat)
#         self.bn1 = nn.BatchNorm1d(2**2 * n_feat)  # Batch normalization
#         self.linear2 = nn.Linear(2**3 * n_feat, 2 * n_feat)
#         self.bn2 = nn.BatchNorm1d(2 * n_feat)  # Batch normalization
#         self.linear3 = nn.Linear(2**2 * n_feat, n_feat)
#         # self.bn3 = nn.BatchNorm1d(n_feat)   # Batch normalization

#         self.relu = nn.SiLU()
        

#         temb = [Embed(1, 2**(i+1)*n_feat) for i in range(3)]
#         self.temb = nn.ModuleList(temb)
    
#     def forward(self, x, c, t):
#         temb = [self.temb[i](t)[:, :] for i in range(3)]
#         c = c.to(torch.float32)
#         x = self.encoder(x)
        
#         x = self.vec2vec(x)
#         x = self.bn(x)  # Apply batch normalization
#         x = self.relu(x)

#         x = self.linear1(torch.cat([x, temb[2]], dim=1))
#         x = self.bn1(x)  # Apply batch normalization
#         x = self.relu(x)
        
#         x = self.linear2(torch.cat([x, temb[1]], dim=1))
#         x = self.bn2(x)  # Apply batch normalization
#         x = self.relu(x)
        
#         x = self.linear3(torch.cat([x, temb[0]], dim=1))
#         # x = self.bn3(x)  # Apply batch normalization
#         # x = self.relu(x)
        
#         return x
    

def ddpm_schedules(beta1, beta2, n_T):
    '''
    Returns pre-computed schedules for DDPM sampling, training process.
    '''
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, n_T + 1, dtype=torch.float32) / n_T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super().__init__()
        self.nn_model = nn_model.to(device)

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss(reduction='mean')

    def forward(self, x, c, epoch):
        """
        this method is used in training, so samples t and noise randomly
        """
        
        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(x.device)  # t ~ Uniform(0, n_T)
        noise=torch.randn_like(x)  # eps ~ N(0, 1)
            
        xt=self.sqrtab[_ts, None] * x + self.sqrtmab[_ts, None] * noise
       
        # This is the xt, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this xt. Loss is what we return.

        # dropout context with some probability
        # 0 represent mask

        # return MSE between added noise, and our predicted noise
        noise_pre = self.nn_model(xt, c, _ts.view(-1,1) / self.n_T)
        if epoch == 999:
            print("xxxxxxxxx")
            print(x)
            print(x.min())
            print(x.max())
            print('noise')
            print(noise)
            print(noise.min())
            print(noise.max())
            print("xt")
            print(xt)
            print(xt.min())
            print(xt.max())
            print("noise_pre")
            print(noise_pre)
            print(noise_pre.min())
            print(noise_pre.max())
        loss = self.loss_mse(noise, noise_pre)
        return loss


# %%
