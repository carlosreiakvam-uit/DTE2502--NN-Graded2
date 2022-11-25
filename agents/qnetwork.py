import torch
import torch.nn.functional as F

import torch.optim as optim

import torch.nn as nn


class QNetwork(nn.Module):

    def __init__(self, model, seed, lr):
        super(QNetwork, self).__init__()

        self.seed = torch.manual_seed(seed)
        layers = self.setup_model(model)
        self.conv = nn.Sequential(*layers)
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        print(self.conv)

    def setup_model(self, model):
        layers = []
        last_out = 4

        for layer in model:
            vals = model[layer]

            if 'Conv2D' in layer:
                layers.append(nn.Conv2d(
                    in_channels=last_out,
                    out_channels=vals['filters'],
                    kernel_size=vals['kernel_size'],
                    padding=vals['padding'] if "padding" in vals else 0,
                    stride=vals['stride'] if "stride" in vals else 1)
                )
                layers.append(nn.ReLU())
                last_out = vals['filters']

            elif 'Flatten' in layer:
                layers.append(nn.Flatten())

            elif 'Dense' in layer:
                layers.append(nn.Linear(in_features=last_out, out_features=vals['units']))
                layers.append(nn.ReLU())
                last_out = vals['units']
        return layers

    def forward(self, x):
        print(x)
        for layer in self.conv:
            x = layer(x)
        return self.conv[-1](x)
