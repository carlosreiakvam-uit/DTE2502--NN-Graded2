from agents.DeepQAgent import DeepQAgent
from agents.models.PGAM import PGAM


class PolicyGradientAgent(DeepQAgent):

    def __init__(self, board_size=10, frames=4, buffer_size=10000, gamma=0.99, n_actions=3, use_target_net=False,
                 version=''):
        super().__init__(board_size, frames, buffer_size, gamma, n_actions, use_target_net)

        self.agent_model = PGAM()
