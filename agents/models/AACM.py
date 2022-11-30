import torch.nn as nn
import torch
import torch.nn.functional as F


class AACM(nn.Module):

    def __init__(self, model_type):
        super(AACM, self).__init__()
        self.model_type = model_type
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding='same').to(self.device)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3).to(self.device)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5).to(self.device)
        self.flatten = nn.Flatten().to(self.device)
        self.fc1 = nn.Linear(64 * 4 * 4, 64).to(self.device)
        self.fc2 = nn.Linear(64, 4).to(self.device)

    def forward(self, t):

        t = self.conv1(t)
        t = F.relu(t)
        t = self.conv2(t)
        t = F.relu(t)

        t = self.flatten(t)

        t = self.conv3(t)
        t = F.relu(t)

        if self.model_type == 'model_logits':
            action_logits = self.fc1(t)
            action_logits = F.relu(action_logits)
            return action_logits
        elif self.model_type == 'model_full':
            action_logits = self.fc1(t)
            action_logits = F.relu(action_logits)
            state_values = self.fc2(t)
            state_values = F.relu(state_values)
            return [action_logits, state_values]
        elif self.model_type == 'model_values':
            state_values = self.fc2(t)
            state_values = F.relu(state_values)
            return state_values
