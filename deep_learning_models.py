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



#################################################################
# Dataset utilities
#################################################################

class LensDataset(Dataset):
    """
    PyTorch dataset for strong gravitational lensing images.

    Each sample consists of:
        - the input image tensor
        - the corresponding ground-truth lens parameters

    This dataset can be used to train ResNet, Vision Transformer,
    or Bayesian ResNet models.
    """

    def __init__(self, df, path_tensor_folder):
        # DataFrame:
        #   index   -> image path
        #   columns -> target lens parameters
        self.df = df

        # Directory containing the image tensors (.pt files)
        self.path = path_tensor_folder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Build the path to the corresponding tensor file
        tensor_path = os.path.join(
            self.path,
            row.name.split('/')[1] + '.pt'
        )

        # Load the image tensor
        x = torch.load(tensor_path)

        # Keep only the lensed source and lens galaxy channels
        x = x[::2]

        # Apply a square-root intensity transform
        x = torch.clamp(x, min=0)
        x = torch.sqrt(x)

        # Load the ground-truth lens parameters
        y = torch.tensor(row.values, dtype=torch.float32)

        return x, (y, row.name)
        
    def split_data(self, training_pct, test_pct, batch_size):
    
        # Total number of samples
        n = len(self)

        # Compute split sizes
        test_size = int(test_pct * n)
        remaining_size = n - test_size

        # Fix the random seed to obtain the same test set
        generator = torch.Generator().manual_seed(42)

        # Split into remaining data and test set
        remaining_dataset, test_dataset = random_split(
            self,
            [remaining_size, test_size],
            generator=generator,
        )
        # Split the remaining data into training and validation sets
        train_size = int(training_pct * n)
        val_size = remaining_size - train_size

        train_dataset, val_dataset = random_split(
            remaining_dataset,
            [train_size, val_size],
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=3,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=3,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=3,
        )

        return train_loader, val_loader, test_loader
    
    
class LensDataset_UNet(Dataset):
    """
    This class allows to load the images and the true parameters for training the model.
    This class LensDataset is made for training the U-Net, the U-Net + ResNet.
    """

    def __init__(self, df, path_tensor_folder):
        self.df = df
        self.path = path_tensor_folder

    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tensor_path = os.path.join(self.path, row.name.split('/')[1] + '.pt') # +'.pt' because the images are stored into tensors files

        # load the tensor
        x = torch.load(tensor_path)
        x = torch.clamp(x, min=0)

        # # Code for normalization , uncomment to make it active 
        # x_min = x.amin(dim=(1, 2), keepdim=True)
        # x_max = x.amax(dim=(1, 2), keepdim=True)
        # # print(x_min.shape, x_max.shape)
        # x = (x - x_min) / (x_max - x_min + 1e-8)
        x = torch.sqrt(x)

        # load the ground truth values
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
        
        self.final_pool = nn.AdaptiveAvgPool2d((7, 7))

        # # Regression head
        self.fc = nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128, num_outputs)
        )

    def forward(self, x):
        # print(x.shape)
        x = F.relu(self.bn_in(self.conv_in(x)))
        # print(x.shape)
        x = self.pool1(self.layer1(x))
        # print(x.shape)
        x = self.pool2(self.layer2(x))
        # print(x.shape)
        x = self.pool3(self.layer3(x))
        # print(x.shape)
        x = self.pool4(self.layer4(x))
        # print(x.shape)
        # Global average pooling
        x = x.mean(dim=[2, 3])  # shape: [B, 256]
        # print(x.shape)
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


    def forward(self, x, return_latent=False, p=0):
        # print(x.shape)
        x1 = F.dropout(self.inc(x), p=p, training=True)
        # print(x1.shape)
        x2 = F.dropout(self.down1(x1), p=p, training=True)
        # print(x2.shape)
        x3 = F.dropout(self.down2(x2), p=p, training=True)
        # print(x3.shape)
        x4 = F.dropout(self.down3(x3), p=p, training=True)
        # print(x4.shape)
    
        x = F.dropout(self.up2(x4, x3), p=p, training=True)
        # print(x.shape)
        x = F.dropout(self.up3(x, x2), p=p, training=True)
        # print(x.shape)
        x = F.dropout(self.up4(x, x1), p=p, training=True)
        # print(x.shape)
        logits = self.outc(x)
        
        if return_latent:
            return logits, x4
        else:
            return logits

#     def forward(self, x, return_latent=False):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         # print(x4.shape)
# #         x5 = self.down4(x4)
        
# #         x = self.up1(x5, x4)
#         x = self.up2(x4, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         logits = self.outc(x)
        
#         if return_latent:
#             return logits, x4
#         else:
#             return logits



import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetPlusPlus(nn.Module):
    """
    U-Net++ (Nested U-Net)

    x0_0 -------- x0_1 -------- x0_2 -------- x0_3
      |             |             |             |
      v             v             v             v
    x1_0 -------- x1_1 -------- x1_2 --------
      |             |             |
      v             v             v
    x2_0 -------- x2_1 --------
      |             |
      v             v
    x3_0 --------
    """

    def __init__(
        self,
        in_channels=4,
        out_channels=4,
        base_channels=64,
        deep_supervision=False,
    ):
        super().__init__()

        self.model_name = "UNet++"
        self.deep_supervision = deep_supervision

        nb = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        ]

        self.pool = nn.MaxPool2d(2)
        self.up = lambda x: F.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=True
        )

        # Encoder
        self.conv0_0 = DoubleConv(in_channels, nb[0])
        self.conv1_0 = DoubleConv(nb[0], nb[1])
        self.conv2_0 = DoubleConv(nb[1], nb[2])
        self.conv3_0 = DoubleConv(nb[2], nb[3])

        # Nested decoder
        self.conv0_1 = DoubleConv(nb[0] + nb[1], nb[0])

        self.conv1_1 = DoubleConv(nb[1] + nb[2], nb[1])
        self.conv0_2 = DoubleConv(nb[0] * 2 + nb[1], nb[0])

        self.conv2_1 = DoubleConv(nb[2] + nb[3], nb[2])
        self.conv1_2 = DoubleConv(nb[1] * 2 + nb[2], nb[1])
        self.conv0_3 = DoubleConv(nb[0] * 3 + nb[1], nb[0])

        # outputs
        self.final = nn.Conv2d(nb[0], out_channels, kernel_size=1)

        if deep_supervision:
            self.final1 = nn.Conv2d(nb[0], out_channels, 1)
            self.final2 = nn.Conv2d(nb[0], out_channels, 1)
            self.final3 = nn.Conv2d(nb[0], out_channels, 1)

    def forward(self, x, return_latent=False):

        # encoder
        x0_0 = self.conv0_0(x)

        x1_0 = self.conv1_0(self.pool(x0_0))

        x2_0 = self.conv2_0(self.pool(x1_0))

        x3_0 = self.conv3_0(self.pool(x2_0))

        # level 1
        x0_1 = self.conv0_1(
            torch.cat([x0_0, self.up(x1_0)], dim=1)
        )

        x1_1 = self.conv1_1(
            torch.cat([x1_0, self.up(x2_0)], dim=1)
        )

        x2_1 = self.conv2_1(
            torch.cat([x2_0, self.up(x3_0)], dim=1)
        )

        # level 2
        x0_2 = self.conv0_2(
            torch.cat([x0_0, x0_1, self.up(x1_1)], dim=1)
        )

        x1_2 = self.conv1_2(
            torch.cat([x1_0, x1_1, self.up(x2_1)], dim=1)
        )

        # level 3
        x0_3 = self.conv0_3(
            torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], dim=1)
        )

        if self.deep_supervision:
            y1 = self.final1(x0_1)
            y2 = self.final2(x0_2)
            y3 = self.final3(x0_3)

            if return_latent:
                return [y1, y2, y3], x3_0

            return [y1, y2, y3]

        logits = self.final(x0_3)

        if return_latent:
            return logits, x3_0

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








import torch
import torch.nn as nn


class ResNetHoliSmokesBayesian(nn.Module):
    """
    Bayesian ResNet architecture for regression.

    The network predicts:
        - mu: the mean of each target variable.
        - logvar: the logarithm of the predictive variance,
                  allowing uncertainty estimation.
    """

    def __init__(self, num_outputs=5):
        super().__init__()

        # Model name (useful for logging/checkpointing)
        self.model_name = 'ResNetHoliSmokesBayesian'
        self.num_outputs = num_outputs

        # ---------------------------------------------------------------------
        # Feature extraction
        # ---------------------------------------------------------------------

        # Initial convolution adapted for 4-channel input images
        self.conv_in = nn.Conv2d(4, 32, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(32)

        # Residual feature extraction blocks
        self.layer1 = BasicBlock(32, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.layer2 = BasicBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.layer3 = BasicBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.layer4 = BasicBlock(128, 256)
        self.pool4 = nn.MaxPool2d(2)

        # Global average pooling reduces the feature map to a 256-dimensional vector
        self.final_pool = nn.AdaptiveAvgPool2d((1, 1))

        # ---------------------------------------------------------------------
        # Shared fully connected representation
        # ---------------------------------------------------------------------

        # Shared latent representation used by both prediction heads
        self.fc_shared = nn.Sequential(
            nn.Dropout(p=0.2),      # Regularization
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )

        # ---------------------------------------------------------------------
        # Output heads
        # ---------------------------------------------------------------------

        # Mean prediction (μ)
        self.fc_mu = nn.Linear(128, num_outputs)

        # Log-variance prediction (log σ²)
        # Predicting the log-variance improves numerical stability and
        # guarantees a positive variance after exponentiation.
        self.fc_logvar = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_outputs)
        )

        # Initialize the uncertainty head with small values to encourage
        # stable training at the beginning of optimization.
        nn.init.zeros_(self.fc_logvar[-1].bias)
        nn.init.normal_(self.fc_logvar[-1].weight, std=0.01)

    def forward(self, x):
        # -----------------------------------------------------------------
        # Feature extraction
        # -----------------------------------------------------------------
    
        # Initial convolution followed by batch normalization and ReLU activation
        x = F.relu(self.bn_in(self.conv_in(x)))
    
        # Residual blocks with progressive spatial downsampling
        x = self.pool1(self.layer1(x))
        x = self.pool2(self.layer2(x))
        x = self.pool3(self.layer3(x))
        x = self.pool4(self.layer4(x))
    
        # Global average pooling to obtain a fixed-length feature vector
        x = self.final_pool(x)
    
        # Flatten feature maps into a 1D vector
        x = x.view(x.size(0), -1)

        # -----------------------------------------------------------------
        # Shared latent representation
        # -----------------------------------------------------------------
    
        # Compute the shared feature representation used by both output heads
        features = self.fc_shared(x)
    
        # -----------------------------------------------------------------
        # Bayesian output heads
        # -----------------------------------------------------------------
    
        # Predict the mean of each target variable
        mu = self.fc_mu(features)
    
        # Predict the logarithm of the predictive variance
        log_var = self.fc_logvar(features)
    
        # Clamp log-variance to avoid numerical instability during training
        log_var = torch.clamp(log_var, -6, 6)
    
        return mu, log_var

class UNetBayesian(nn.Module):
    """
    Two-stage Bayesian architecture.

    Stage 1:
        A U-Net generates a feature-enhanced representation of the input.

    Stage 2:
        A Bayesian ResNet predicts the target values together with their
        associated uncertainties.
    """

    def __init__(self):
        super().__init__()

        # Model name (useful for logging and checkpointing)
        self.model_name = "UNetBayesian"

        # -----------------------------------------------------------------
        # Model components
        # -----------------------------------------------------------------

        # First stage: U-Net for feature extraction
        self.unet = UNet()

        # Second stage: Bayesian ResNet for regression and uncertainty estimation
        self.resnet = ResNetHoliSmokesBayesian()

    def forward(self, x):
        """Run the complete two-stage pipeline."""
        features = self.forward_unet(x)
        return self.forward_resnet(features)

    def forward_unet(self, x):
        """Extract features using the U-Net."""
        return self.unet(x)

    def forward_resnet(self, x):
        """Predict the target means and uncertainties."""
        return self.resnet(x)


# dimensions X = [2000, 4, 140, 140]
# then [2000, 8, 140, 140] because padding of 1, and 8 filters of size 3x3
# max pool [2000, 8, 70, 70]
## [2000, 16, 70, 70]
# max pool [2000, 16, 35, 35]