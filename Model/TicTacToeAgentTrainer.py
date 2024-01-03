import tensorflow as tf
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.environments import tf_py_environment
from TicTacToeEnvironment import TicTacToeEnvironment
from RandomTicTacToePlayer import RandomTicTacToePlayer

class TicTacToeAgentTrainer:
    def __init__(self):
        self.environment = TicTacToeEnvironment()
        self.tf_env = tf_py_environment.TFPyEnvironment(self.environment)
        
        # Set up the Q-Network and DQN agent
        self.q_net = q_network.QNetwork(
            self.tf_env.observation_spec(),
            self.tf_env.action_spec(),
            fc_layer_params=(100, 50, 25))

        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)

        self.agent = dqn_agent.DqnAgent(
            self.tf_env.time_step_spec(),
            self.tf_env.action_spec(),
            q_network=self.q_net,
            optimizer=self.optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss)

        self.agent.initialize()
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=self.tf_env.batch_size,
            max_length=100000)


    def collect_data(self, opponent, num_episodes):
        for _ in range(num_episodes):
            time_step = self.tf_env.reset()
            while not time_step.is_last():
                if self.tf_env.current_time_step().is_first():
                    # Randomly decide who takes the first move
                    if np.random.rand() < 0.5:
                        action = opponent.select_action()
                        time_step = self.tf_env.step(action)

                if not time_step.is_last():
                    action_step = self.agent.collect_policy.action(time_step)
                    time_step = self.tf_env.step(action_step.action)
                    self.replay_buffer.add_batch(time_step)

                if not time_step.is_last():
                    action = opponent.select_action()
                    time_step = self.tf_env.step(action)

    def train(self, random_epochs, training_epochs, iterations, batch_size=64):
        random_opponent = RandomTicTacToePlayer(self.environment)

        # Train against random opponent for a fixed number of epochs
        for _ in range(random_epochs):
            self.collect_data(random_opponent, iterations)

            # Sample a batch of data from the buffer and update the agent's network
            for _ in range(iterations):
                experience, _ = next(iter(self.replay_buffer.as_dataset(
                    num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2).prefetch(3)))
                self.agent.train(experience)

        past_versions = [self.agent]

        # Train against past two versions and random opponent
        for epoch in range(training_epochs):
            opponents = past_versions[-2:] + [random_opponent]
            for opponent in opponents:
                self.collect_data(opponent, iterations)

                # Sample and train
                for _ in range(iterations):
                    experience, _ = next(iter(self.replay_buffer.as_dataset(
                        num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2).prefetch(3)))
                    self.agent.train(experience)

            # Save the current agent for future training
            past_versions.append(self.agent)

            # Optional: Evaluate agent's performance here
