import json
import torch.nn as nn
from torch import optim
from agents.Agent import mean_huber_loss

# Deep Q Agent Model Network
class DQM(nn.Module):

    def __init__(self, version, device, frames=4, n_actions=4, board_size=10, buffer_size=10000,
                 gamma=0.99, use_target_net=True):
        super(DQM, self).__init__()
        self.version = version
        self._n_frames = frames
        self._n_actions = n_actions
        self._board_size = board_size

        with open('model_config/{:s}.json'.format(self.version), 'r') as f:
            m = json.loads(f.read())

        out_channels_prev = m['frames']
        layers = []
        self.device = device

        for layer in m['model']:
            l = m['model'][layer]
            if ('Conv2D' in layer):
                if "padding" in l:
                    padding = l["padding"]
                    layers.append(
                        nn.Conv2d(in_channels=out_channels_prev, out_channels=l["filters"],
                                  kernel_size=l["kernel_size"],
                                  padding=padding))
                else:
                    layers.append(
                        nn.Conv2d(in_channels=out_channels_prev, out_channels=l["filters"],
                                  kernel_size=l["kernel_size"]))
                if "activation" in l:
                    layers.append(nn.ReLU())
                out_channels_prev = l["filters"]
            if 'Flatten' in layer:
                layers.append(nn.Flatten())
                out_channels_prev = 64 * 4 * 4
            if 'Dense' in layer:
                layers.append(nn.Linear(out_channels_prev, l['units']))
                if "activation" in l:
                    layers.append(nn.ReLU())

        self.conv = nn.Sequential(*layers).to(device)
        self.out = nn.Linear(64, self._n_actions).to(device)

        self.criterion = mean_huber_loss
        self.optimizer = optim.RMSprop(self.parameters(), lr=0.0005)

        self.criterion = mean_huber_loss
        self.to(device)

    def forward(self, t):
        t.to(self.device)
        t = self.conv(t)
        return self.out(t)
