import copy
import tensorflow as tf
import numpy as np
import time
import tensorflow as tf
from tf_agents.trajectories import trajectory
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.environments import tf_py_environment
from TicTacToeEnvironment import TicTacToeEnvironment
from RandomTicTacToePlayer import RandomTicTacToePlayer
from TicTacToeEnvironmentBucket import TicTacToeEnvironmentBucket
from FlattenAndConcatenateLayer import FlattenAndConcatenateLayer


class TicTacToeAgentTrainer:
    def __init__(self,fc_layer_params=(100, 50, 25),learning_rate=1e-3,buffer_max_length=100000):
        self.environment_bucket = TicTacToeEnvironmentBucket(max_environments=20,agent_player=0)
        self.past_versions = []
        self.evaluation_results = []
        self.env = self.environment_bucket.get_environment()
        
        # Set up the Q-Network and DQN agent
        self.q_net = q_network.QNetwork(
            input_tensor_spec = self.env.tf.observation_spec(),
            action_spec = self.env.tf.action_spec(),
            preprocessing_combiner=FlattenAndConcatenateLayer(),
            fc_layer_params=fc_layer_params)
        
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

        self.agent = dqn_agent.DqnAgent(
            self.env.tf.time_step_spec(),
            self.env.tf.action_spec(),
            q_network=self.q_net,
            optimizer=self.optimizer,
            observation_and_action_constraint_splitter=TicTacToeEnvironment.observation_and_action_constraint_splitter,
            td_errors_loss_fn=common.element_wise_squared_loss)

        self.agent.initialize()
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=self.env.tf.batch_size,
            max_length=buffer_max_length)

    def sign(self,x):
        return 1 if (x > 0) else -1 if (x < 0) else 0

    def collect_data(self, opponent, num_episodes):
        for _ in range(num_episodes):
            agent_is_first = (np.random.rand() < 0.5)
            time_step = self.env.tf.reset()

            self.env.py.set_next_player(0 if agent_is_first else 1)
            while not time_step.is_last():

                # print(f"Loop: {_}, player: {self.env.py.current_player}, step: {time_step}")
                if self.env.py.current_player == self.env.py.agent_player:
                    
                    action_step = self.agent.collect_policy.action(time_step)
                    next_time_step = self.env.tf.step(action_step.action)

                    step_trajectory = trajectory.from_transition(time_step, action_step, next_time_step)
                    self.replay_buffer.add_batch(step_trajectory)
                    time_step = next_time_step
                else:
                    action_step = opponent.policy.action(time_step)
                    time_step = self.env.tf.step(action_step.action)


    def train(self, random_epochs, training_epochs, iterations, batch_size=64, evaluation_num_episodes=1000, num_previous_versions=2):
        total_training_start_time = time.time()
        random_opponent = RandomTicTacToePlayer(self.env.py)

        # Train against random opponent for a fixed number of epochs
        for epoch in range(random_epochs):
            print(f"Starting random epoch: {epoch}")
            start_time = time.time()
            self.collect_data(random_opponent, iterations)

            # Sample a batch of data from the buffer and update the agent's network
            for _ in range(iterations):
                experience, _ = next(iter(self.replay_buffer.as_dataset(
                    num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2).prefetch(3)))
                self.agent.train(experience)
            post_training_time = time.time()
            print(f"Epoch duration: {post_training_time-start_time:.2f}")

        self.past_versions.append(self.agent)
        self.evaluation_results.append(self.evaluate_agent(evaluation_num_episodes,0,1,True))

        # Train against past two versions and random opponent
        for epoch in range(training_epochs):
            print(f"Starting main epoch: {epoch}")
            start_time = time.time()
            opponents = dict()
            opponents['Rnd'] = RandomTicTacToePlayer(self.env.py)
            previous_version = len(self.past_versions)-1
            for v in range(previous_version,max(-1,previous_version-num_previous_versions),-1):
                opponents[f"V{v}"] = copy.deepcopy(self.past_versions[v])
            for opponent_name,opponent in opponents.items():
                print(f"...collecting against opponent {opponent_name}")
                self.collect_data(opponent, iterations)

                print(f"...learning from opponent {opponent_name}")
                for _ in range(iterations):
                    experience, _ = next(iter(self.replay_buffer.as_dataset(
                        num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2).prefetch(3)))
                    self.agent.train(experience)

            # Save the current agent for future training
            post_training_time = time.time()
            self.past_versions.append(self.agent)
            # Evaluate the current agent
            self.evaluation_results.append(self.evaluate_agent(evaluation_num_episodes,len(self.past_versions)-1,1,True))
            print(f"Epoch duration: {post_training_time-start_time:.2f}")

        total_training_end_time = time.time()
        print(f"Total training duration: {total_training_end_time-total_training_start_time:.2f}")
        
        return self.evaluation_results


    def evaluate_agent(self, num_episodes,current_version,num_versions,include_random):
        print(f"V{current_version} evaluation start")
        start_time = time.time()
        results = {}
        env = self.environment_bucket.get_environment()
        agent = copy.deepcopy(self.past_versions[current_version])
        opponents = dict()
        if include_random:
            opponents['Rnd'] = RandomTicTacToePlayer(env.py)
        for v in range(current_version,max(-1,current_version-num_versions),-1):
            opponents[f"V{v}"] = copy.deepcopy(self.past_versions[v])

        
        for name, opponent in opponents.items():
            counts = [0,[0,0,0,0,0],[0,0,0,0,0]]
            rs = ""
            for no in range(num_episodes):
                agent_is_first = (np.random.rand() < 0.5)
                time_step = env.tf.reset()
                env.py.set_next_player(0 if agent_is_first else 1)
                counts[0] += (1 if agent_is_first else 0)
                fo=f"{name}[{no}]:{'A' if agent_is_first else 'O'}:"
                while not time_step.is_last():
                    if env.py.current_player == env.py.agent_player:
                        action_step = agent.collect_policy.action(time_step)
                    else:
                        action_step = opponent.policy.action(time_step)

                    time_step = env.tf.step(action_step.action)
                    fo+=str(int(action_step.action))
                        
                
                fo+=f":{float(time_step.reward)}"
                count_bucket = 1-self.sign(time_step.reward)
                if env.py.agent_caused_error:
                    count_bucket = 3
                if env.py.opponent_caused_error:
                    count_bucket = 4
                counts[2-agent_is_first][count_bucket]+=1
                if agent_is_first==0:
                    rs += "w" if time_step.reward > 0 else "l" if time_step.reward < 0 else "d"
                else:
                    rs += "W" if time_step.reward > 0 else "L" if time_step.reward < 0 else "D"
                # print(fo)

            results[name] = [num_episodes,counts[0],[x/counts[0]*100 for x in counts[1]],[x/(num_episodes-counts[0])*100 for x in counts[2]]]
            print(f"V{current_version} {name}:{rs}")

        self.evaluation_results.append(results)
        xxx = {k:([[f'{ii:.2f}' for ii in i] if isinstance(i,list) else i for i in v]) for k,v in self.evaluation_results[-1].items()}
        print(f"V{current_version} evaluation results: {xxx}")
        end_time = time.time()
        print(f"V{current_version} evaluation duration: {end_time-start_time:.2f}")
        self.environment_bucket.return_environment(env)
        return results


# Initialize the trainer
trainer = TicTacToeAgentTrainer(fc_layer_params=(100, 50, 25),learning_rate=1e-3,buffer_max_length=100000)

# Example usage
# evaluation_history = trainer.train(random_epochs=5, training_epochs=25, iterations=100, batch_size=64, evaluation_num_episodes=100)
evaluation_history = trainer.train(random_epochs=5, training_epochs=100, iterations=100, batch_size=64, evaluation_num_episodes=100)
