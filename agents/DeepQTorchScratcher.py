from agents.agent import Agent
from agents.simon_model import DeepQLearningNet as Network
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn


# class Network(nn.Module):
#
#     def __init__(self):
#         super(Network, self).__init__()
#         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding='same').to(self.device)
#         self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3).to(self.device)
#         self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5).to(self.device)
#
#         self.flatten = nn.Flatten().to(self.device)
#         self.fc1 = nn.Linear(64 * 4 * 4, 64).to(self.device)
#         self.out = nn.Linear(64, 4).to(self.device)
#
#     def forward(self, t):
#         t = torch.Tensor(t).to(self.device)
#         t = self.conv1(t)
#         t = F.relu(t)
#         t = self.conv2(t)
#         t = F.relu(t)
#         t = self.conv3(t)
#         t = F.relu(t)
#         t = self.flatten(t)
#         t = self.fc1(t)
#         t = F.relu(t)
#         t = self.out(t)
#         # t = F.softmax(t, dim=-1)
#         return t


class DeepQTorchScratcher(Agent):
    def __init__(self, board_size, frames, buffer_size, n_actions, version, use_target_net=True, gamma=0.99):
        super().__init__(board_size, frames, buffer_size, gamma, n_actions, use_target_net, version)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model = Network().to(self.device)
        self._target_net = self._model
        self._target_net.to(self.device)
        self.update_target_net()
        self._input_shape = (self._board_size, self._board_size, self._n_frames)

    def getModel(self):
        return self._model

    def update_target_net(self):  # true
        self._target_net.load_state_dict(
            self._model.state_dict())  # from simon # sets weights for _target_net from _model

    def train_agent(self, batch_size=32, num_games=1, reward_clip=False):

        s, a, r, next_s, done, legal_moves = self._buffer.sample(batch_size)
        if reward_clip:  # true
            r = np.sign(r)
        current_model = self._target_net if self._use_target_net else self._model  # goes through

        next_model_outputs = self._get_model_outputs(next_s, current_model)

        discounted_reward = r + (self._gamma * np.max(
            np.where(legal_moves == 1, next_model_outputs.cpu().detach(), -np.inf),
            axis=1).reshape(-1, 1)) * (1 - done)

        discounted_reward = torch.from_numpy(discounted_reward).to(self.device)

        target = self._get_model_outputs(s)

        a = torch.tensor(a).to(self.device)

        target = (1 - a) * target + a * discounted_reward

        loss = self.train_model(self._normalize(s), target, self._model)  # var current_model sist
        # loss = round(loss, 5)
        return loss

    def train_model(self, board, target, model):
        optimizer = model.optimizer
        model.train()

        labels = target.type(torch.float32).to(self.device)
        predicts = model(board).to(self.device)

        # Zero the gradients
        optimizer.zero_grad()

        model.loss = model.criterion(predicts, labels)
        model.loss.backward()

        # Adjust learning weights
        optimizer.step()


        # self.optimizer = optim.RMSprop(model.parameters(), lr=0.0005)
        # model.train()
        # # targets = torch.from_numpy(targets)  # is 64, should be 32
        #
        # targets = targets.type(torch.float32).to(self.device)
        #
        # states = np.transpose(states, (0, 3, 1, 2))  # alternative to stack reshape
        # states = torch.from_numpy(states).to(self.device)
        # preds = self._model(states).to(self.device)
        # # loss = torch.sqrt(F.mse_loss(preds, labels))  # RMSE
        # self.optimizer.zero_grad()
        # model.loss = nn.functional.huber_loss(preds, targets, reduction='mean')
        # model.loss.backward()
        # self.optimizer.step()
        return model.loss.item()

    def _get_model_outputs(self, board, model=None):
        board = self._prepare_input(board)

        if model is None:  # false
            model = self._model.to(self.device)
        model_outputs = model((torch.tensor(np.transpose(board, (0, 3, 1, 2)))).to(self.device))

        return model_outputs

    def _prepare_input(self, board):
        if (board.ndim == 3):  # always false?
            board = board.reshape((1,) + self._input_shape)
        board = self._normalize(board.copy())
        return board.copy()

    def _normalize(self, board):
        return board.astype(np.float32) / 4  # normalization

    def move(self, board, legal_moves, value=None):
        # use the agent model to make the predictions
        model_outputs = self._get_model_outputs(board, self._model)
        model_outputs = model_outputs.cpu().detach().numpy()
        return np.argmax(np.where(legal_moves == 1, model_outputs, -np.inf), axis=1)  # argmax

    def save_model(self, file_path='', iteration=None):
        if iteration is not None:
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        torch.save(self._model.state_dict(), "{}/model_{:04d}.h5".format(file_path, iteration))
        # torch.save(self._model, f="{}/model_{:04d}.h5".format(file_path, iteration))
        if self._use_target_net:
            # torch.save(self._target_net, f="{}/model_{:04d}_target.h5".format(file_path, iteration))
            torch.save(self._target_net.state_dict(), "{}/model_{:04d}_target.h5".format(file_path, iteration))

    def load_model(self, file_path='', iteration=None):
        if iteration is not None:
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        # self._model.load("{}/model_{:04d}.h5".format(file_path, iteration))
        # torch.load("{}/model_{:04d}.h5".format(file_path, iteration))
        self._model.load_state_dict((torch.load("{}/model_{:04d}.h5".format(file_path, iteration))))

        if self._use_target_net:
            # torch.load("{}/model_{:04d}_target.h5".format(file_path, iteration))
            # self._target_net.load("{}/model_{:04d}_target.h5".format(file_path, iteration))
            self._target_net.load_state_dict(torch.load("{}/model_{:04d}_target.h5".format(file_path, iteration)))

        # print("Couldn't locate models at {}, check provided path".format(file_path))
