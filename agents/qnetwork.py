import torch
import torch.nn.functional as F
import torch.nn as nn


class QNetwork(nn.Module):

    def __init__(self, model, seed):
        super(QNetwork, self).__init__()

        self.seed = torch.manual_seed(seed)
        self.layers = []

        for layer in model:
            vals = model[layer]
            if 'Conv2D' in layer:
                conv2d = {
                    'layer': nn.Conv2d(in_channels=10, out_channels=vals['filters'], kernel_size=vals['kernel_size']),
                    'activation': vals['activation']}
                self.layers.append(conv2d)

            if 'Flatten' in layer and len(vals) > 0:
                flatten = {torch.flatten}
                self.layers.append(flatten)

            if 'Dense' in layer:
                dense = {
                    'layer': nn.Linear(in_channels=10, out_channels=vals['filters']),
                    'activation': vals['activation']}
                self.layers.append(dense)

            self.out = nn.Linear(2, 10)  # må vel ha en utgang men verdier er tatt fra luft og kjærlighet

        def forward(self):
            x = nn.Linear(10, 2)  # input verdier kun tatt fra tynn luft
            for layer in self.layers:
                x = layer(x)

            return self.fc3(x)
