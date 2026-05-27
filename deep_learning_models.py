import torch
from astropy.io import fits
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from astropy.io import fits
import numpy as np
import random
import os
from torch.utils.data import random_split
import torch
from torchviz import make_dot
from IPython.display import Image
################################################################# Load Dataset

class LensDataset(Dataset):
    def __init__(self, df, path_tensor_folder):
        self.df = df
        self.path = path_tensor_folder

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # filename stored in df, e.g. "sample_0001.pt"
        tensor_path = os.path.join(self.path, row.name.split('/')[1] + '.pt')

        x = torch.load(tensor_path)
        x = x[::2]
        # x = x[:-1]
 
        # Clamp and transform
        x = torch.clamp(x, min=0)
        x = torch.sqrt(x)
  
        # label
        y = torch.tensor(row.values, dtype=torch.float32)

        return x, y
    
    def split_data(self, training_pct, test_pct, batch_size):
 
        n = len(self)
        test_size = int(test_pct * n)
        remaining_size = n - test_size

        generator = torch.Generator().manual_seed(42)

        remaining_dataset, test_dataset = random_split(
            self,
            [remaining_size, test_size],
            generator=generator)

        train_size = int(training_pct * n)
        val_size = remaining_size - train_size

        train_dataset, val_dataset = random_split(
            remaining_dataset,
            [train_size, val_size])
     
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=3)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=3)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=3)
        
        return train_loader, val_loader, test_loader
    
    
class LensDataset_UNet(Dataset):
    def __init__(self, df, path_tensor_folder):
        self.df = df
        self.path = path_tensor_folder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # filename stored in df, e.g. "sample_0001.pt"
        tensor_path = os.path.join(self.path, row.name.split('/')[1] + '.pt')

        # load the tensor
        x = torch.load(tensor_path)
#         x = torch.asinh(x)
        x = torch.clamp(x, min=0)
        x = torch.sqrt(x)

        # label
        y = torch.tensor(row.values, dtype=torch.float32)

        return (x[::2], x[1::2]), (y, row.name) # x[::2] are the 4 bands g-r-i-z for the lens + source, x[1::2] are the 4 bands with the lens only.
    
    def split_data(self, training_pct, test_pct, batch_size):
 
        n = len(self)
        test_size = int(test_pct * n)
        remaining_size = n - test_size

        generator = torch.Generator().manual_seed(42)

        remaining_dataset, test_dataset = random_split(
            self,
            [remaining_size, test_size],
            generator=generator)

        train_size = int(training_pct * n)
        val_size = remaining_size - train_size

        train_dataset, val_dataset = random_split(
            remaining_dataset,
            [train_size, val_size])
     
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=3)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=3)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=3)
        
        return train_loader, val_loader, test_loader

######################################################################################### View architechture

class viewer():
    def __init__(self, model):
        self.model = model
    
    def show_model(self):
        x = torch.randn(1, 4, 224, 224) 

        y = self.model(x)

        dot = make_dot(y, params=dict(self.model.named_parameters()))
        dot.format = "png"
        dot.render("resnetmini", directory=".", cleanup=True)

########################################################################################################## ResNet Mini
import torch.nn as nn
import torch

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return torch.relu(out)

class ResNetMini(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(4, 16, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(16)

        self.block1 = ResidualBlock(16)
        self.pool1 = nn.MaxPool2d(2)

        self.block2 = ResidualBlock(16)
        self.pool2 = nn.MaxPool2d(2)

        self.block3 = ResidualBlock(16)
        self.pool3 = nn.MaxPool2d(2)

        self.fc = nn.Linear(16 * 17 * 17, 5)  # adjust if input size changes

    def forward(self, x):
        x = torch.relu(self.bn_in(self.conv_in(x)))
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)
    


##################################################################### ResNetHoliSmokes and Baysian neuron network

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # projection if channels or stride change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += identity
        return F.relu(out)


class ResNetHoliSmokes(nn.Module):
    def __init__(self, num_outputs=5):
        super().__init__()
        self.model_name = 'ResNetHoliSmokes'

        # Input: 4 channels (g,r,i,z)
        self.conv_in = nn.Conv2d(4, 32, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(32)

        # 4 residual stages
        self.layer1 = BasicBlock(32, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.layer2 = BasicBlock(32, 64, stride=1)
        self.pool2 = nn.MaxPool2d(2)

        self.layer3 = BasicBlock(64, 128, stride=1)
        self.pool3 = nn.MaxPool2d(2)

        self.layer4 = BasicBlock(128, 256, stride=1)
        self.pool4 = nn.MaxPool2d(2)
        
        self.final_pool = nn.AdaptiveAvgPool2d((4, 4))
        # self.fc = nn.Sequential(
        #     nn.Linear(50176, 4096),
        #     nn.ReLU(),
        #     nn.Linear(4096, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, num_outputs)
        # )

        self.fc = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, num_outputs)
        )

        
        # Regression head
        # self.fc = nn.Sequential(
        #     nn.Linear(256,128),
        #     nn.ReLU(),
        #     nn.Linear(128, num_outputs)
        # )

    def forward(self, x):
        x = F.relu(self.bn_in(self.conv_in(x)))

        x = self.pool1(self.layer1(x))
        x = self.pool2(self.layer2(x))
        x = self.pool3(self.layer3(x))
        x = self.pool4(self.layer4(x))
        x = self.final_pool(x)
        x = x.view(x.size(0), -1)
        # Global average pooling
        # x = x.mean(dim=[2, 3])  # shape: [B, 256]
        return self.fc(x)
   

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        std = 0.1
        self.weight_mu.data.normal_(0, std)
        self.weight_rho.data.normal_(-3, std)
        self.bias_mu.data.normal_(0, std)
        self.bias_rho.data.normal_(-3, std)

    def sample_weights(self):
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))

        eps_w = torch.randn_like(weight_sigma)
        eps_b = torch.randn_like(bias_sigma)
        
#         print('shape mu : ', self.weight_mu.mean(dim = 1))
#         print('shape sigma : ', weight_sigma.mean(dim = 1))
        
        weight = self.weight_mu + weight_sigma * eps_w
        bias = self.bias_mu + bias_sigma * eps_b

        return weight, bias

    def forward(self, x, sample=True):
        if self.training or sample:
            w, b = self.sample_weights()
        else:
            w, b = self.weight_mu, self.bias_mu
        return F.linear(x, w, b)


#     def forward(self, x):
#         # Bayesian layer
#         w, b = self.sample_weights()
#         x = F.linear(x, w, b)
#         return x
    
    def kl_divergence(self):
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))

        kl_weight = (torch.log(1.0 / weight_sigma) + (weight_sigma**2 + self.weight_mu**2) / 2.0 - 0.5).sum()

        kl_bias = (torch.log(1.0 / bias_sigma) + (bias_sigma**2 + self.bias_mu**2) / 2.0 - 0.5).sum()
        
        return kl_weight + kl_bias



class BayesianResNetMini(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_name = 'BayesianResNetMini'
        
        self.conv_in = nn.Conv2d(4, 32, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(32)

        # 4 residual stages
        self.layer1 = BasicBlock(32, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.layer2 = BasicBlock(32, 64, stride=1)
        self.pool2 = nn.MaxPool2d(2)

        self.layer3 = BasicBlock(64, 128, stride=1)
        self.pool3 = nn.MaxPool2d(2)

        self.layer4 = BasicBlock(128, 256, stride=1)
        self.pool4 = nn.MaxPool2d(2)
        
        self.dropout = nn.Dropout(p=0.2)
        
        # Bayesian head
        self.fc1 = nn.Linear(256 * 8 * 8, 4096) # 256 * 8 * 8/4
        self.fc2 = nn.Linear(4096, 512)
        self.fcb = BayesianLinear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)
        
    def forward_features(self, x):
        x = torch.relu(self.bn_in(self.conv_in(x)))
        x = self.pool1(self.layer1(x))
        x = self.pool2(self.layer2(x))
        x = self.pool3(self.layer3(x))
        x = self.pool4(self.layer4(x))
        x = x.view(x.size(0), -1)
        return x
    
    def forward_head(self, x):
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fcb(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        out = self.fc4(x)
        
        mu, log_var = out.chunk(2, dim=-1)
        sigma = torch.exp(0.5 * log_var)
#         sigma = torch.sqrt(F.softplus(log_var) + 1e-6)
        return mu, sigma, log_var
    
    def forward(self, x):
        features = self.forward_features(x)
        return self.forward_head(features)
    
    def kl_divergence(self):
        return self.fcb.kl_divergence()
    
    
############################################################################################################"""" U - NET

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2,
                                         kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

        self.bilinear = bilinear

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # pad if needed (in case of odd sizes)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self,
                 in_channels=4,
                 out_channels=4,
                 base_channels=64,
                 bilinear=True):
        super().__init__()
        self.model_name = "UNet"

        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        factor = 2 if bilinear else 1
        self.down3 = Down(base_channels * 4, base_channels * 8// factor)
#         self.down4 = Down(base_channels * 8, base_channels * 16 // factor)

#         self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)
        self.outc = OutConv(base_channels, out_channels)

    def forward(self, x, return_latent=False):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
#         x5 = self.down4(x4)
        
#         x = self.up1(x5, x4)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        if return_latent:
            return logits, x4
        else:
            return logits



################################################################## U net then Resnet
    
    
class UNetThenResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_name = "UNetThenResNet"
        self.unet = UNet()                     # first stage
        self.resnet = ResNetHoliSmokes()       # second stage

    def forward(self, x):
        features = self.forward_unet(x)
        return self.forward_resnet(features)
    
    def forward_unet(self, x):
        return self.unet(x) 
    
    def forward_resnet(self, x):
        return self.resnet(x)
    
    
    
######################################################################## UNet then NN
class LatentNN(nn.Module):
    def __init__(self, output_dim=5):
        super().__init__()
        self.model_name = "LatentNN"

        self.flatten = nn.Flatten()

        self.net = nn.Sequential(
#             nn.Linear(512 * 8 * 8, 2048),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)   # final output
        )

    def forward(self, x):
        x = x.mean(dim=[2, 3]) 
        x = self.flatten(x)
        return self.net(x)

    
class UNetThenNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_name = "UNetThenNN"
        self.unet = UNet()                     # first stage
        self.latentNN = LatentNN()       # second stage

    def forward(self, x):
        _, latent = self.unet(x, return_latent=True)  # (B, 512, 8, 8) (B, 256, 17, 17)
        out = self.latentNN(latent)
        return out
    
    def forward_unet(self, x, return_latent=False):
        return self.unet(x, return_latent) 
    
    def forward_latentNN(self, x):
        return self.latentNN(x)
    

# dimensions X = [2000, 4, 140, 140]
# then [2000, 8, 140, 140] because padding of 1, and 8 filters of size 3x3
# max pool [2000, 8, 70, 70]
## [2000, 16, 70, 70]
# max pool [2000, 16, 35, 35]