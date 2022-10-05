import torch
import torch.nn as nn
import numpy as np
from sklearn.manifold import TSNE
import argparse
import json
import logging
import random
from datetime import datetime
from importlib import import_module
from itertools import chain
from os.path import join, exists
import matplotlib.pyplot as plt
import torch
from torch.autograd import grad
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader

_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cuda_setup(cuda=True, gpu_idx=0):
    if cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.set_device(gpu_idx)
    else:
        device = torch.device('cpu')
    return device

class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.z_size = config['z_size']
        self.use_bias = config['model']['G']['use_bias']
        self.relu_slope = config['model']['G']['relu_slope']
        self.model = nn.Sequential(
            nn.Linear(in_features=self.z_size, out_features=64, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=64, out_features=128, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=128, out_features=512, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=512, out_features=1024, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=1024, out_features=400 * 3, bias=self.use_bias),
        )

    def forward(self, input):
        output = self.model(input.squeeze())
        output = output.view(-1, 3, 400)
        return output


class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.z_size = config['z_size']
        self.use_bias = config['model']['D']['use_bias']
        self.relu_slope = config['model']['D']['relu_slope']
        self.dropout = config['model']['D']['dropout']

        self.model = nn.Sequential(

            nn.Linear(self.z_size, 512, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(512, 512, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(512, 128, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(128, 64, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(64, 1, bias=True)
        )

    def forward(self, x):
        logit = self.model(x)
        return logit


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.z_size = config['z_size']
        self.use_bias = config['model']['E']['use_bias']
        self.relu_slope = config['model']['E']['relu_slope']

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1,
                      bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1,
                      bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1,
                      bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1,
                      bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1,
                      bias=self.use_bias),
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 256, bias=True),
            nn.ReLU(inplace=True)
        )

        self.mu_layer = nn.Linear(256, self.z_size, bias=True)
        self.std_layer = nn.Linear(256, self.z_size, bias=True)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        output = self.conv(x)
        output2 = output.max(dim=2)[0]
        logit = self.fc(output2)
        mu = self.mu_layer(logit)
        logvar = self.std_layer(logit)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    
#Latent Space for 512
with open('models/AAE/hyperparams_parts.json') as f:
    config = json.load(f)

def weights_init(m):
    classname = m.__class__.__name__
    if classname in ('Conv1d', 'Linear'):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

random.seed(config['seed'])
torch.manual_seed(config['seed'])
torch.cuda.manual_seed_all(config['seed'])

device = cuda_setup(config['cuda'], config['gpu'])


G = Generator(config).to(device)
E = Encoder(config).to(device)
D = Discriminator(config).to(device)

G.apply(weights_init)
E.apply(weights_init)
D.apply(weights_init)



EG_optim = getattr(optim, config['optimizer']['EG']['type'])
EG_optim = EG_optim(chain(E.parameters(), G.parameters()),
                    **config['optimizer']['EG']['hyperparams'])

D_optim = getattr(optim, config['optimizer']['D']['type'])
D_optim = D_optim(D.parameters(),
                  **config['optimizer']['D']['hyperparams'])



