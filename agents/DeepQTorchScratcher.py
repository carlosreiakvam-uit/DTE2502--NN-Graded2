import torch
from agents.agent import Agent
from replay_buffer import ReplayBuffer, ReplayBufferNumpy
import numpy as np
import pickle
import torch
import torch.nn.functional as F
import torch.nn as nn


class Network(nn.Module):

    def __init__(self, seed):
        super(Network, self).__init__()
        self.seed = seed
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.out = nn.Linear(64, 4)

    def forward(self, t):
        t = self.conv1(t)
        t = F.relu(t)
        t = self.conv2(t)
        t = F.relu(t)
        t = self.conv3(t)
        t = F.relu(t)
        t = self.flatten(t)
        t = self.fc1(t)
        t = F.relu(t)
        t = self.out(t)
        # t = F.softmax(t, dim=-1)
        return t


class DeepQTorchScratcher(Agent):
    def __init__(self, board_size, frames, buffer_size, n_actions, version, use_target_net=True, gamma=0.99):
        super().__init__(board_size, frames, buffer_size, gamma, n_actions, use_target_net, version)
        self._model = Network(seed=0)
        self._target_net = self._model
        self.update_target_net()

    def getModel(self):
        return self._model

    # Manually implpementing the same model as in the v17.1 setup

    def update_target_net(self):
        # self._target_net.set_weights(self._model.get_weights())
        self._target_net.load_state_dict(self._model.state_dict())  # simon

    def train_agent(self, batch_size=32, num_games=1, reward_clip=False):
        s, a, r, next_s, done, legal_moves = self._buffer.sample(batch_size)
        if reward_clip:
            r = np.sign(r)
        # calculate the discounted reward, and then train accordingly
        current_model = self._target_net if self._use_target_net else self._model
        next_model_outputs = self._get_model_outputs(next_s, current_model)
        # our estimate of expexted future discounted reward
        discounted_reward = r + (self._gamma * np.max(
            np.where(legal_moves == 1, next_model_outputs, -np.inf),
            axis=1).reshape(-1, 1)) * (1 - done)
        # create the target variable, only the column with action has different value
        target = self._get_model_outputs(s)
        # we bother only with the difference in reward estimate at the selected action
        target = (1 - a) * target + a * discounted_reward
        # fit
        loss = self._model.train_on_batch(self._normalize_board(s), target)
        # loss = round(loss, 5)
        return loss

    def _get_model_outputs(self, board, model=None):
        # to correct dimensions and normalize
        board = self._prepare_input(board)
        # the default model to use
        if model is None:
            model = self._model
        model_outputs = model.predict_on_batch(board)
        return model_outputs

    def _prepare_input(self, board):
        if (board.ndim == 3):
            board = board.reshape((1,) + self._input_shape)
        board = self._normalize_board(board.copy())
        return board.copy()

    def _normalize_board(self, board):
        return board.astype(np.float32) / 4.0

    def save_model(self, file_path='', iteration=None):
        if iteration is not None:
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        self._model.save_weights("{}/model_{:04d}.h5".format(file_path, iteration))
        if self._use_target_net:
            self._target_net.save_weights("{}/model_{:04d}_target.h5".format(file_path, iteration))

    def move(self, board, legal_moves, value=None):
        # use the agent model to make the predictions
        model_outputs = self._get_model_outputs(board, self._model)
        return np.argmax(np.where(legal_moves == 1, model_outputs, -np.inf), axis=1)
