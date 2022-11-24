import torch
import torch.nn.functional as F

import torch.optim as optim

import torch.nn as nn


class QNetwork(nn.Module):

    def __init__(self, model, seed, lr):
        super(QNetwork, self).__init__()

        self.seed = torch.manual_seed(seed)
        self.layers = []
        self.last_out = 4
        self.setup_model(model)
        self.conv = nn.Sequential(*self.layers)
        # self.optimizer = optim.RMSprop(self.conv, lr)
        print(self.conv)

    def setup_model(self, model):
        for layer in model:
            vals = model[layer]

            if 'Conv2D' in layer:
                self.layers.append(nn.Conv2d(
                    in_channels=self.last_out,
                    out_channels=vals['filters'],
                    kernel_size=vals['kernel_size'],
                    padding=vals['padding'] if "padding" in vals else 0,
                    stride=vals['stride'] if "stride" in vals else 1)
                )
                self.layers.append(nn.ReLU())
                self.last_out = vals['filters']

            elif 'Flatten' in layer:
                self.layers.append(nn.Flatten())

            elif 'Dense' in layer:
                self.layers.append(nn.Linear(in_features=self.last_out, out_features=vals['units']))
                self.layers.append(nn.ReLU())
                self.last_out = vals['units']

    def forward(self, x):
        for layer in self.conv:
            x = layer(x)
        return x
