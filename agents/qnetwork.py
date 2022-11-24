import torch
import torch.nn.functional as F
import torch.nn as nn


class QNetwork(nn.Module):

    def __init__(self, model, seed):
        super(QNetwork, self).__init__()

        self.seed = torch.manual_seed(seed)
        self.layers = []
        self.last_out = 4

        for layer in model:
            vals = model[layer]

            if 'Conv2D' in layer:
                conv2d = {
                    'layer': nn.Conv2d(
                        in_channels=self.last_out,
                        out_channels=vals['filters'],
                        kernel_size=vals['kernel_size'],
                        padding=vals['padding'] if "padding" in vals else 0,
                        stride=vals['stride'] if "stride" in vals else 1),
                    'activation': nn.ReLU if vals['activation'] == 'relu' else nn.ReLU}

                self.layers.append(conv2d)
                self.last_out = vals['filters']  # update last out to be used for inputs

            if 'Flatten' in layer:
                self.layers.append(torch.flatten)

            if 'Dense' in layer:
                dense = {
                    'layer': nn.Linear(in_features=self.last_out, out_features=vals['units']),
                    'activation': nn.ReLU if vals['activation'] == 'relu' else nn.ReLU}
                self.layers.append(dense)
                self.last_out = vals['units']

    def forward(self):
        for layer in self.layers:
            x = layer['model']
            x = layer['activation']
        return x
