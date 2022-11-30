from agents.Agent import Agent
from agents.models.DQM import DQM
import numpy as np
import torch


class DeepQAgent(Agent):
    def __init__(self, board_size, frames, buffer_size, n_actions, version, use_target_net=True, gamma=0.99):
        super().__init__(board_size, frames, buffer_size, gamma, n_actions, use_target_net, version)

        self.board_size = board_size
        self.frames = frames
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.n_actions = 4
        self.use_target_net = use_target_net
        self.version = version

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.reset_models()

    def reset_models(self):
        self._model = self._agent_model()
        if self._use_target_net:
            self._target_net = self._agent_model()
            self.update_target_net()

    def _agent_model(self):
        self.model = DQM(version=self.version, frames=self._n_frames, n_actions=self.n_actions,
                         board_size=self.board_size, buffer_size=self.buffer_size,
                         gamma=self.gamma, use_target_net=self.use_target_net, device=self.device)
        return self.model.to(self.device)

    def train_agent(self, batch_size=32, num_games=1, reward_clip=False):

        s, a, r, next_s, done, legal_moves = self._buffer.sample(batch_size)
        if reward_clip: r = np.sign(r)

        # Set current model to target model if in use
        current_model = self._target_net if self._use_target_net else self._model

        # Get the next outputs
        next_model_outputs = self._get_model_outputs(next_s, current_model)

        # Calculate the current discounted reward
        discounted_reward = r + (self._gamma * np.max(
            np.where(legal_moves == 1, next_model_outputs.cpu().detach(), -np.inf), axis=1)
                                 .reshape(-1, 1)) * (1 - done)

        # Translate discounted reward into a Tensor
        discounted_reward = torch.from_numpy(discounted_reward).to(self.device)

        # Get targets from model
        targets = self._get_model_outputs(s)

        # Translate actions to Tensor and calculate new targets
        a = torch.tensor(a).to(self.device)
        targets = (1 - a) * targets + a * discounted_reward

        # Transpose states to shape (64,2,10,10)
        s = (np.transpose(self._normalize(s), (0, 3, 1, 2)))

        # Translate states to Tensor
        s = torch.from_numpy(s).to(self.device)

        # Calculate the loss with the given model using states and targets
        loss = self.train_model(s, targets, self._model)
        return loss

    def train_model(self, board, targets, model):

        # Use optimizer as declared in model (RMSProp)
        optimizer = model.optimizer

        # Set model to state of train
        model.train()

        # Translate target to Tensor and set type to float32
        targets = targets.type(torch.float32).to(self.device)

        # Get predictions from the model and translate to Tensor
        predicts = model(board).to(self.device)


        optimizer.zero_grad()

        # Set the model loss via the models criterion (mean huber loss)
        model.loss = model.criterion(predicts, targets)

        # Run backprop
        model.loss.backward()

        # Step the Optimizer a tiny step in the direction of smallest loss
        optimizer.step()

        # Get and return loss from model
        loss = model.loss.item()
        return loss

    def update_target_net(self):  # true
        if self._use_target_net:
            self._target_net.load_state_dict(self._model.state_dict())

    def _get_model_outputs(self, board, model=None):
        board = self._prepare_input(board)
        if model is None:  # false
            model = self._model.to(self.device)
        model_outputs = model((torch.tensor(np.transpose(board, (0, 3, 1, 2)))).to(self.device))
        return model_outputs

    def _prepare_input(self, board):
        board = self._normalize(board.copy())
        return board.copy()

    def _normalize(self, board):
        return board.astype(np.float32) / 4.0  # normalization

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
        if self._use_target_net:
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
            self._target_net.load_state_dict(torch.load("{}/model_{:04d}_target.h5".format(file_path, iteration)))

    def get_action_proba(self, board):
        model_outputs = self._get_model_outputs(board, self._model)
        model_outputs = np.clip(model_outputs, -10, 10)
        model_outputs = model_outputs - model_outputs.max(axis=1).reshape((-1, 1))
        model_outputs = np.exp(model_outputs)
        model_outputs = model_outputs / model_outputs.sum(axis=1).reshape((-1, 1))
        return model_outputs
