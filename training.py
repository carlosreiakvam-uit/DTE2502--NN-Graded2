from tqdm import tqdm
import pandas as pd
import time
from utils import play_game2
from game_environment import SnakeNumpy
import torch
from agents.DeepQAgent import DeepQAgent
from agents.AdvantageActorCriticAgent import AdvantageActorCriticAgent
import json

torch.manual_seed(42)
version = 'v17.1'

# get training configurations
with open('model_config/{:s}.json'.format(version), 'r') as f:
    m = json.loads(f.read())
    board_size = m['board_size']
    frames = m['frames']  # keep frames >= 2
    max_time_limit = m['max_time_limit']
    supervised = bool(m['supervised'])
    n_actions = m['n_actions']
    obstacles = bool(m['obstacles'])
    buffer_size = m['buffer_size']

episodes = 2 * (10 ** 5)
log_frequency = 500
games_eval = 8

agent_type = 'DQN'

if agent_type == 'DQN':
    agent = DeepQAgent(board_size=board_size, frames=frames, n_actions=n_actions, buffer_size=buffer_size,
                       version=version)

    epsilon, epsilon_end = 1, 0.01
    reward_type = 'current'
    sample_actions = False
    n_games_training = 8 * 16
    decay = 0.97

else:  # agent type is Advantage Actor Critic Agent
    agent = AdvantageActorCriticAgent(board_size=board_size, frames=frames, n_actions=n_actions,
                                      buffer_size=10000, version=version)
    epsilon, epsilon_end = -1, -1
    reward_type = 'current'
    sample_actions = True
    exploration_threshold = 0.1
    n_games_training = 32
    decay = 1

if agent_type == 'DQN':  # play some games initially to fill the buffer
    # setup the environment
    games = 512
    env = SnakeNumpy(board_size=board_size, frames=frames,
                     max_time_limit=max_time_limit, games=games,
                     frame_mode=True, obstacles=obstacles, version=version)
    ct = time.time()
    _ = play_game2(env, agent, n_actions, n_games=games, record=True,
                   epsilon=epsilon, verbose=True, reset_seed=False,
                   frame_mode=True, total_frames=games * 64)
    print('Playing {:d} frames took {:.2f}s'.format(games * 64, time.time() - ct))

# Setup new environments
env = SnakeNumpy(board_size=board_size, frames=frames,
                 max_time_limit=max_time_limit, games=n_games_training,
                 frame_mode=True, obstacles=obstacles, version=version)
env2 = SnakeNumpy(board_size=board_size, frames=frames,
                  max_time_limit=max_time_limit, games=games_eval,
                  frame_mode=True, obstacles=obstacles, version=version)

# Initialize model logs
model_logs = {'iteration': [], 'reward_mean': [],
              'length_mean': [], 'games': [], 'loss': []}

for index in tqdm(range(episodes)):
    if agent_type == 'DQN':
        # make small changes to the buffer and slowly train
        _, _, _ = play_game2(env, agent, n_actions, epsilon=epsilon,
                             n_games=n_games_training, record=True,
                             sample_actions=sample_actions, reward_type=reward_type,
                             frame_mode=True, total_frames=n_games_training,
                             stateful=True)

        loss = agent.train_agent(batch_size=64,
                                 num_games=n_games_training, reward_clip=True)
    else:  # Advanced Actor Critic Agent
        # play a couple of games and train on all
        _, _, total_games = play_game2(env, agent, n_actions, epsilon=epsilon,
                                       n_games=n_games_training, record=True,
                                       sample_actions=sample_actions, reward_type=reward_type,
                                       frame_mode=True, total_games=n_games_training * 2)
        buffer_size = agent.get_buffer_size()
        loss = agent.train_agent(batch_size=agent.get_buffer_size(),
                                 num_games=total_games, reward_clip=True)

    # for policy gradient algorithm, we only take current episodes for training
    if agent_type in ['PolicyGradientAgent', 'AdvantageActorCriticAgent']:
        agent.reset_buffer()

    # check performance every once in a while
    if (index + 1) % log_frequency == 0:
        # keep track of agent rewards_history
        current_rewards, current_lengths, current_games = \
            play_game2(env2, agent, n_actions, n_games=games_eval, epsilon=-1,
                       record=False, sample_actions=False, frame_mode=True,
                       total_frames=-1, total_games=games_eval)

        model_logs['iteration'].append(index + 1)
        model_logs['reward_mean'].append(round(int(current_rewards) / current_games, 2))
        model_logs['length_mean'].append(round(int(current_lengths) / current_games, 2))
        model_logs['games'].append(current_games)
        model_logs['loss'].append(loss)
        pd.DataFrame(model_logs)[['iteration', 'reward_mean', 'length_mean', 'games', 'loss']] \
            .to_csv('model_logs/{:s}.csv'.format(version), index=False)

        # copy weights to target network and save models
    if (index + 1) % log_frequency == 0:
        agent.update_target_net()
        agent.save_model(file_path='models/{:s}'.format(version), iteration=(index + 1))
        # keep some epsilon alive for training
        epsilon = max(epsilon * decay, epsilon_end)
