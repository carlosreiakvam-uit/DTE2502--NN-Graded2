import torch.nn as nn
import torch
import torch.nn.functional as F

# Policy Gradient Actor Model Network

class PGAM(nn.Module):

    def __init__(self):
        super(PGAM, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding='same').to(self.device)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3).to(self.device)

        self.flatten = nn.Flatten().to(self.device)
        self.out = nn.Linear(64 * 4 * 4, 64).to(self.device)

    def forward(self, t):
        t = self.conv1(t)
        t = F.relu(t)
        t = self.conv2(t)
        t = F.relu(t)

        t = self.flatten(t)

        t = self.out(t)
        return t
