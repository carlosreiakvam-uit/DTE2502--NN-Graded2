import torch
from agents.agent import Agent
from replay_buffer import ReplayBuffer, ReplayBufferNumpy
import numpy as np
import torch
import torch.optim as optim
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
        t = torch.tensor(t)
        t = torch.stack([t[batch_idx].T for batch_idx in range(t.shape[0])])  # stack and retain shape
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

    def update_target_net(self):  # true
        # self._target_net.set_weights(self._model.get_weights())
        self._target_net.load_state_dict(
            self._model.state_dict())  # from simon # sets weights for _target_net from _model

    def train_agent(self, batch_size=32, num_games=1, reward_clip=False):
        # where self.buffer is a ReplayBuffer initialized from Agent
        # It is sampling batch_size number of examples from the buffer

        # getting batc_size number of: state, action, reward, next_state, done, legal_moves
        s, a, r, next_s, done, legal_moves = self._buffer.sample(batch_size)
        if reward_clip:  # true
            r = np.sign(r)  # returns batch_size numbers of: -1 if r<0, 0 if r== 0 or 1 if r > 0
            # this is in order to indicate if the reward is negative, positive or passive
            # essentially it is clipping the reward to discrete values

        # calculate the discounted reward, and then train accordingly
        # Decides wether or not to use target_net as current_model, even though target net is equal to current_model 🤔
        current_model = self._target_net if self._use_target_net else self._model  # goes through

        # returns 64 x 4 outputs of predicted labels for current model!!!!!
        # This is a training step!
        # reshapes the output
        next_model_outputs = self._get_model_outputs(next_s,
                                                     current_model)  # next_s er board inne i get_model_outputs!!

        # our estimate of expexted future discounted reward
        # discounted_reward is a 64x4 tensor, a modified reward tensor
        # gamma is discount, set to 0.99
        # np.max determines max output of
        # np.where: numpy.where(condition, [x, y, ]/), where true yield x, otherwise yield y
        #   In other words if legal_moves == 1: yield next_model_outuout, else yiedl negative infinity
        # the where part is reshaped to a single column
        # and finally multiplied with (1-done) which either yields 1 or 0
        #   meaning if not done, it makes the whole part 0
        tensor_copy = next_model_outputs.clone().detach().numpy()
        # tensor_copy = tensor_copy.detach().numpy()
        discounted_reward = r + (self._gamma * np.max(
            np.where(legal_moves == 1, tensor_copy, -np.inf),
            axis=1).reshape(-1, 1)) * (1 - done)

        # create the target variable, only the column with action has different value <- original comment
        # This is another go at get_model_outout, only this time, only state is input
        # state is 64x10x10x2, aka 64 examples of a game
        target = self._get_model_outputs(s)

        # we bother only with the difference in reward estimate at the selected action
        # a is a 64x4 tensor
        # target is the model outputs, or labels if you will
        # discounted reward is what it sounds like
        target = target.detach().numpy()
        target = target.astype(np.float32)
        target = (1 - a) * target + a * discounted_reward

        # EXTREMELY IMPORTANT STEP HERE
        # fit
        # train_on_batch() is a tensorflow method (not to be mistaken for predict_on_batch)
        # states are normalized and used as input X <---!!!
        # target is used for labels y <--- !!!
        # the training provides, as indicated, the loss which we intend to minimize
        # loss = self._model.train_on_batch(self._normalize_board(s), target) # tf
        s = self._prepare_input(s)
        s = s.detach().numpy()
        # s = nn.functional.normalize(s)
        loss = self.train_model(s, target, current_model)
        # loss = round(loss, 5)
        return loss

    def train_model(self, board, labels, model):
        self.optimizer = optim.Adam(model.parameters(), lr=0.01)
        labels = torch.from_numpy(labels)  # is 64, should be 32
        labels = labels.type(torch.float32)
        board = torch.from_numpy(board)
        preds = self._model(board)
        loss = torch.sqrt(F.mse_loss(preds, labels))  # RMSE
        self.optimizer.zero_grad()
        a = model.conv1.weight.grad
        loss.backward()
        a = model.conv1.weight.grad
        self.optimizer.step()
        return loss.item()

    def _get_model_outputs(self, board, model=None):
        # to correct dimensions and normalize
        board = self._prepare_input(board)  # needs to return a tensor? does it?
        # the default model to use
        if model is None:  # false
            model = self._model
        # model_outputs = model.predict_on_batch(board) # tf
        model_outputs = model(board)
        return model_outputs

    def _prepare_input(self, board):
        board = board.astype(np.float32) / 4  # normalization

        # board = torch.from_numpy(board)
        # board = nn.functional.normalize(board)
        # board = torch.reshape(board, (64, 2, 10, 10)) # do not use, does not move data in shape
        # board = torch.stack([board[batch_idx].T for batch_idx in range(board.shape[0])]) # stack and retain shape
        # board = board.permute(*torch.arange(-board.ndim 1, -1, -1)) # recommended py docu but gives wrong shape
        return board

    # def _normalize_board(self, board):
    # original is to return copy of numpy board cast to type np.float32 divided by 4 (probably because of 4 actions)
    # return board.astype(np.float32) / 4.0
    # return

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
