import numpy as np
from agents.PolicyGradientAgent import PolicyGradientAgent
from agents.DeepQAgent import DeepQAgent
from agents.models.AACM import AACM
import torch


class AdvantageActorCriticAgent(DeepQAgent):
    def __init__(self, board_size=10, frames=4, buffer_size=10000, gamma=0.99, n_actions=3, use_target_net=True,
                 version=''):
        super().__init__(board_size, frames, buffer_size, gamma, n_actions, use_target_net, version)

    def reset_models(self):
        self._model = AACM(model_type='model_logits')
        self._full_model = AACM(model_type='model_full')
        self._values_model = AACM(model_type='model_values')

        if self._use_target_net:
            _, _, self._target_net = self._agent_model()
            self.update_target_net()

    def _agent_model(self):
        model_logits = AACM(model_type='model_logits')
        model_full = AACM(model_type='model_full')
        model_values = AACM(model_type='model_values')
        return model_logits, model_full, model_values

    def save_model(self, file_path='', iteration=None):
        if iteration is not None:
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        torch.save(self._model.state_dict(), "{}/model_{:04d}.h5".format(file_path, iteration))
        torch.save(self._full_model.state_dict(), "{}/model_{:04d}_full.h5".format(file_path, iteration))
        if self._use_target_net:
            torch.save(self._values_model.state_dict(), "{}/model_{:04d}_values.h5".format(file_path, iteration))
            torch.save(self._target_net.state_dict(), "{}/model_{:04d}_target.h5".format(file_path, iteration))

    def load_model(self, file_path='', iteration=None):
        if iteration is not None:
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        self._model.load_weights("{}/model_{:04d}.h5".format(file_path, iteration))
        self._full_model.load_weights("{}/model_{:04d}_full.h5".format(file_path, iteration))
        if (self._use_target_net):
            self._values_model.load_weights("{}/model_{:04d}_values.h5".format(file_path, iteration))
            self._target_net.load_weights("{}/model_{:04d}_target.h5".format(file_path, iteration))

    def update_target_net(self):
        if self._use_target_net:
            if self._use_target_net:
                self._target_net.load_state_dict(self._model.state_dict())

    def train_agent(self, batch_size=32, beta=0.001, normalize_rewards=False,
                    num_games=1, reward_clip=False):
        # in policy gradient, only one complete episode is used for training
        s, a, r, next_s, done, _ = self._buffer.sample(self._buffer.get_current_size())
        s_prepared = self._prepare_input(s)
        next_s_prepared = self._prepare_input(next_s)
        # unlike DQN, the discounted reward is not estimated
        # we have defined custom actor and critic losses functions above
        # use that to train to agent model

        # normzlize the rewards for training stability, does not work in practice
        if normalize_rewards:
            if (r == r[0][0]).sum() == r.shape[0]:
                r -= r
            else:
                r = (r - np.mean(r)) / np.std(r)

        if reward_clip:
            r = np.sign(r)

        # calculate V values
        if self._use_target_net:
            next_s_pred = self._target_net(next_s_prepared)
        else:
            next_s_pred = self._values_model(next_s_prepared)
        s_pred = self._values_model(s_prepared)

        # prepare target
        future_reward = self._gamma * next_s_pred * (1 - done)
        # calculate target for actor (uses advantage), similar to Policy Gradient
        advantage = a * (r + future_reward - s_pred)

        # calculate target for critic, simply current reward + future expected reward
        critic_target = r + future_reward

        model = self._full_model
        loss = self.torch_trainer(model, advantage, critic_target)
        return loss

    def torch_trainer(self, model, inputs, labels, lr=0.0005):
        [w, b] = model.parameters().to(self.device)
        y_pred = model(inputs)
        model.loss = model(y_pred, labels).to(self.device)

        # compute gradients (grad)
        model.loss.backward()
        with torch.no_grad():
            w -= w.grad * lr
            b -= b.grad * lr
            w.grad.zero_()
            b.grad.zero_()

        model.optimizer.zero_grad()
        model.optimizer.step()
        loss = model.loss.item()
        return loss
