import tensorflow as tf

class AdvantageActorCriticAgent: # had (PolicyGradientAgent)
    """This agent uses the Advantage Actor Critic method to train
    the reinforcement learning agent, we will use Q actor critic here

    Attributes
    ----------
    _action_values_model : Tensorflow Graph
        Contains the network for the action values calculation model
    _actor_update : function
        Custom function to prepare the
    """

    def __init__(self, board_size=10, frames=4, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=True,
                 version=''):
        # DeepQLearningAgent.__init__(self, board_size=board_size, frames=frames,
        #                             buffer_size=buffer_size, gamma=gamma,
        #                             n_actions=n_actions, use_target_net=use_target_net,
        #                             version=version)
        self._optimizer = tf.keras.optimizers.RMSprop(5e-4)

    def _agent_model(self):
        """Returns the models which evaluate prob logits and action values
        for a given state input, Model is compiled in a different function
        Overrides parent

        Returns
        -------
        model_logits : TensorFlow Graph
            A2C model graph for action logits
        model_full : TensorFlow Graph
            A2C model complete graph
        """
        input_board = Input((self._board_size, self._board_size, self._n_frames,))
        x = Conv2D(16, (3, 3), activation='relu', data_format='channels_last')(input_board)
        x = Conv2D(32, (3, 3), activation='relu', data_format='channels_last')(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu', name='dense')(x)
        action_logits = Dense(self._n_actions, activation='linear', name='action_logits')(x)
        state_values = Dense(1, activation='linear', name='state_values')(x)

        model_logits = Model(inputs=input_board, outputs=action_logits)
        model_full = Model(inputs=input_board, outputs=[action_logits, state_values])
        model_values = Model(inputs=input_board, outputs=state_values)
        # updates are calculated in the train_agent function

        return model_logits, model_full, model_values

    def reset_models(self):
        """ Reset all the models by creating new graphs"""
        self._model, self._full_model, self._values_model = self._agent_model()
        if (self._use_target_net):
            _, _, self._target_net = self._agent_model()
            self.update_target_net()

    def save_model(self, file_path='', iteration=None):
        """Save the current models to disk using tensorflow's
        inbuilt save model function (saves in h5 format)
        saving weights instead of model as cannot load compiled
        model with any kind of custom object (loss or metric)

        Parameters
        ----------
        file_path : str, optional
            Path where to save the file
        iteration : int, optional
            Iteration number to tag the file name with, if None, iteration is 0
        """
        if (iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        self._model.save_weights("{}/model_{:04d}.h5".format(file_path, iteration))
        self._full_model.save_weights("{}/model_{:04d}_full.h5".format(file_path, iteration))
        if (self._use_target_net):
            self._values_model.save_weights("{}/model_{:04d}_values.h5".format(file_path, iteration))
            self._target_net.save_weights("{}/model_{:04d}_target.h5".format(file_path, iteration))

    def load_model(self, file_path='', iteration=None):
        """ load any existing models, if available """
        """Load models from disk using tensorflow's
        inbuilt load model function (model saved in h5 format)

        Parameters
        ----------
        file_path : str, optional
            Path where to find the file
        iteration : int, optional
            Iteration number the file is tagged with, if None, iteration is 0

        Raises
        ------
        FileNotFoundError
            The file is not loaded if not found and an error message is printed,
            this error does not affect the functioning of the program
        """
        if (iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        self._model.load_weights("{}/model_{:04d}.h5".format(file_path, iteration))
        self._full_model.load_weights("{}/model_{:04d}_full.h5".format(file_path, iteration))
        if (self._use_target_net):
            self._values_model.load_weights("{}/model_{:04d}_values.h5".format(file_path, iteration))
            self._target_net.load_weights("{}/model_{:04d}_target.h5".format(file_path, iteration))

    def update_target_net(self):
        """Update the weights of the target network, which is kept
        static for a few iterations to stabilize the other network.
        This should not be updated very frequently
        """
        if (self._use_target_net):
            self._target_net.set_weights(self._values_model.get_weights())

    def train_agent(self, batch_size=32, beta=0.001, normalize_rewards=False,
                    num_games=1, reward_clip=False):
        """Train the model by sampling from buffer and return the error
        The buffer is assumed to contain all states of a finite set of games
        and is fully sampled from the buffer
        Overrides parent

        Parameters
        ----------
        batch_size : int, optional
            Not used here, kept for consistency with other agents
        beta : float, optional
            The weight for the policy gradient entropy loss
        normalize_rewards : bool, optional
            Whether to normalize rewards for stable training
        num_games : int, optional
            Not used here, kept for consistency with other agents
        reward_clip : bool, optional
            Not used here, kept for consistency with other agents

        Returns
        -------
        error : list
            The current loss (total loss, actor loss, critic loss)
        """
        # in policy gradient, only one complete episode is used for training
        s, a, r, next_s, done, _ = self._buffer.sample(self._buffer.get_current_size())
        s_prepared = self._prepare_input(s)
        next_s_prepared = self._prepare_input(next_s)
        # unlike DQN, the discounted reward is not estimated
        # we have defined custom actor and critic losses functions above
        # use that to train to agent model

        # normzlize the rewards for training stability, does not work in practice
        if (normalize_rewards):
            if ((r == r[0][0]).sum() == r.shape[0]):
                # std dev is zero
                r -= r
            else:
                r = (r - np.mean(r)) / np.std(r)

        if (reward_clip):
            r = np.sign(r)

        # calculate V values
        if (self._use_target_net):
            next_s_pred = self._target_net.predict_on_batch(next_s_prepared)
        else:
            next_s_pred = self._values_model.predict_on_batch(next_s_prepared)
        s_pred = self._values_model.predict_on_batch(s_prepared)

        # prepare target
        future_reward = self._gamma * next_s_pred * (1 - done)
        # calculate target for actor (uses advantage), similar to Policy Gradient
        advantage = a * (r + future_reward - s_pred)

        # calculate target for critic, simply current reward + future expected reward
        critic_target = r + future_reward

        model = self._full_model
        with tf.GradientTape() as tape:
            model_out = model(s_prepared)
            policy = tf.nn.softmax(model_out[0])
            log_policy = tf.nn.log_softmax(model_out[0])
            # calculate loss
            J = tf.reduce_sum(tf.multiply(advantage, log_policy)) / num_games
            entropy = -tf.reduce_sum(tf.multiply(policy, log_policy)) / num_games
            actor_loss = -J - beta * entropy
            critic_loss = mean_huber_loss(critic_target, model_out[1])
            loss = actor_loss + critic_loss
        # get the gradients
        grads = tape.gradient(loss, model.trainable_weights)
        # grads = [tf.clip_by_value(grad, -5, 5) for grad in grads]
        # run the optimizer
        self._optimizer.apply_gradients(zip(grads, model.trainable_variables))

        loss = [loss.numpy(), actor_loss.numpy(), critic_loss.numpy()]
        return loss[0] if len(loss) == 1 else loss