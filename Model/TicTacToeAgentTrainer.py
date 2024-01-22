import pickle
import traceback

import copy
import multiprocessing
import threading
import uuid
import tensorflow as tf
import numpy as np
import time
from tf_agents.trajectories import trajectory
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common
from tf_agents.replay_buffers import tf_uniform_replay_buffer,episodic_replay_buffer
from tf_agents.environments import tf_py_environment
from TicTacToeEnvironment import TicTacToeEnvironment
from RandomTicTacToePlayer import RandomTicTacToePlayer
from TicTacToeEnvironmentBucket import TicTacToeEnvironmentBucket


class TicTacToeAgentTrainer:
    def __init__(self,fc_layer_params=(100, 50, 25),learning_rate=1e-3,buffer_max_length=100000,num_steps= 2):
        self.init_parms = f"fc_layer_params={fc_layer_params}, learning_rate={learning_rate}, buffer_max_length={buffer_max_length}, num_steps={num_steps}"
        self.environment_bucket = TicTacToeEnvironmentBucket(max_environments=20,agent_player=0)
        self.past_versions = []
        self._episode_counter = TicTacToeAgentTrainer.ThreadSafeIdGenerator()

        env = self.environment_bucket.get_environment()
        
        q_net, optimizer, agent = self.setup_agent(fc_layer_params, learning_rate, num_steps, env.tf.observation_spec(), env.tf.action_spec(), env.tf.time_step_spec())

        agent.initialize()
        # self.replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(
        #     data_spec=agent.collect_data_spec,
        #     completed_only=True,
        #     buffer_size=8,
        #     capacity=buffer_max_length)
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=agent.collect_data_spec,
            batch_size=env.tf.batch_size,
            max_length=buffer_max_length)
        
        self.env = env
        self.q_net = q_net
        self.optimizer = optimizer
        self.agent = agent
        self.num_steps = num_steps

    class ThreadSafeIdGenerator:
        def __init__(self, initial_value=0):
            self.value = initial_value
            self.lock = threading.Lock()

        def get_id(self):
            with self.lock:
                self.value += 1
                return self.value

    def setup_agent(self, fc_layer_params, learning_rate, num_steps, observation_spec, action_spec, time_step_spec):
        q_net = q_network.QNetwork(
            input_tensor_spec = observation_spec,
            action_spec = action_spec,
            preprocessing_combiner=TicTacToeEnvironment.FlattenAndConcatenateLayer(),
            fc_layer_params=fc_layer_params)
        
        # optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

        agent = dqn_agent.DqnAgent(
            time_step_spec,
            action_spec,
            q_network=q_net,
            n_step_update = (num_steps-1),
            optimizer=optimizer,
            observation_and_action_constraint_splitter=TicTacToeEnvironment.observation_and_action_constraint_splitter,
            td_errors_loss_fn=common.element_wise_squared_loss)
            
        return q_net,optimizer,agent

    def sign(self,x):
        return 1 if (x > 0) else -1 if (x < 0) else 0

    def collect_data(self, env, agent, opponent, num_episodes):
        print(f"...collecting {num_episodes} iterations against opponent {opponent[1]}")
        buffer = []
        # Generate a new GUID for each episode

        for no in range(num_episodes):
            agent_is_first = (no%2==0)#(np.random.rand() < 0.5)
            time_step = env.tf.reset()
            tmp_buffer=[]
            env.py.set_next_player(0 if agent_is_first else 1)
            while not time_step.is_last():

                # print(f"Loop: {_}, player: {env.py.current_player}, step: {time_step}")
                if env.py.current_player == env.py.agent_player:
                    
                    action_step = agent.collect_policy.action(time_step)
                    next_time_step = env.tf.step(action_step.action)

                    step_trajectory = trajectory.from_transition(time_step, action_step, next_time_step)
                    tmp_buffer.append(step_trajectory)
                    time_step = next_time_step
                else:
                    action_step = (opponent[0]).policy.action(time_step)
                    time_step = env.tf.step(action_step.action)
            buffer.append((self._episode_counter.get_id(),tmp_buffer))
        return buffer


    def train(self, random_epochs, training_epochs, iterations, batch_size=64, evaluation_num_episodes=1000):
        self.train_parms = f"random_epochs={random_epochs}, training_epochs={training_epochs}, iterations={iterations}, batch_size={batch_size}, evaluation_num_episodes={evaluation_num_episodes}, iterations={iterations}"

        total_training_start_time = time.time()
        random_opponent = RandomTicTacToePlayer()

        # Train against random opponent for a fixed number of epochs
        for epoch in range(random_epochs):
            print('===================================================================')
            print(f"Starting random epoch: {epoch}")
            print(self.init_parms)
            print(self.train_parms)
            print('===================================================================')
            start_time = time.time()
            trajectories = self.collect_data(self.env, self.agent,(random_opponent,'Rnd'), iterations)
            for episode in trajectories:
                # episode_id = tf.constant([episode[0]], dtype=tf.int64)
                for t in episode[1]:
                    # self.replay_buffer.add_batch(t, episode_id)
                    self.replay_buffer.add_batch(t)
            # Sample a batch of data from the buffer and update the agent's network
            # experiences_raw = self.replay_buffer.as_dataset(num_parallel_calls=tf.data.AUTOTUNE)
            experiences_raw = self.replay_buffer.as_dataset(num_parallel_calls=tf.data.AUTOTUNE, sample_batch_size=batch_size, num_steps=self.num_steps)
            
            experiences = iter(experiences_raw.prefetch(tf.data.AUTOTUNE))
            for _ in range(iterations):
                experience, _ = next(experiences)
                self.agent.train(experience)
            post_training_time = time.time()
            print(f"Epoch duration: {post_training_time-start_time:.2f}")

        self.past_versions.append({"agent":self.agent})
        self.past_versions[-1]["training_results"]=self.evaluate_agent(evaluation_num_episodes,0,1,True)

        # Train against past two versions and random opponent
        for epoch in range(training_epochs):
            print('===================================================================')
            print(f"Starting main epoch: {epoch}")
            print(self.init_parms)
            print(self.train_parms)
            print('===================================================================')
            start_time = time.time()
            opponents = []
            opponents.append((RandomTicTacToePlayer(),'Rnd'))
            best_opponents = self.get_best_opponents()
            print("Best Opponents: ", [f"{n}({self.past_versions[n]['evaluation_results']['Rnd'][2][0]:.2f}, {self.past_versions[n]['evaluation_results']['Rnd'][3][0]:.2f})" for n in best_opponents])

            for v in best_opponents:#range(previous_version,max(-1,previous_version-num_previous_versions),-1):
                opponents.append((copy.deepcopy(self.past_versions[v]['agent']),f"V{v}"))

            collect_parms = list(zip([self.environment_bucket.get_environment() for _ in opponents], [copy.deepcopy(self.agent) for _ in opponents], opponents, [iterations for _ in opponents]))

            # with multiprocessing.Pool(len(collect_parms)) as pool:
            #     results = pool.starmap(self.collect_data, collect_parms)
            results = [self.collect_data(*v) for v in collect_parms]
            for trajectories in results:
                for episode in trajectories:
                    # episode_id = tf.constant([episode[0]], dtype=tf.int64)
                    for t in episode[1]:
                        # self.replay_buffer.add_batch(t, episode_id)
                        self.replay_buffer.add_batch(t)

            for env_tbd,_,_,_ in collect_parms:
                self.environment_bucket.return_environment(env_tbd)

            # for cp_agent,cp_opponent,cp_iterations in collect_parms:
            #     trajectories = self.collect_data(cp_agent,cp_opponent[0], cp_iterations)
            #     for t in trajectories:
            #         self.replay_buffer.add_batch(t)

            print(f"...learning from {len(opponents)} opponents")
            experiences = iter(self.replay_buffer.as_dataset(
                num_parallel_calls=tf.data.AUTOTUNE, sample_batch_size=batch_size, num_steps=self.num_steps).prefetch(tf.data.AUTOTUNE))
            for _ in range(iterations*len(opponents)):
                experience, _ = next(experiences)
                self.agent.train(experience)

            # Save the current agent for future training
            post_training_time = time.time()
            self.past_versions.append({"agent":self.agent})
            current_version = len(self.past_versions)-1
            print(f"Epoch {epoch} (V{current_version}) duration: {post_training_time-start_time:.2f}")
            # Evaluate the current agent
            self.past_versions[-1]["training_results"]=self.evaluate_agent(evaluation_num_episodes,current_version,1,True)


        total_training_end_time = time.time()
        print(f"Total training duration: {total_training_end_time-total_training_start_time:.2f}")

    def get_best_opponents(self):
        l = len(self.past_versions)
        best_start = sorted(range(l),key=lambda n: self.past_versions[n]['evaluation_results']['Rnd'][2][0], reverse=True)
        # print([f"{n}({self.past_versions[n]['evaluation_results']['Rnd'][2][0]})" for n in best_start])
        best_second = sorted(range(l),key=lambda n: self.past_versions[n]['evaluation_results']['Rnd'][3][0], reverse=True)
        # print([f"{n}({self.past_versions[n]['evaluation_results']['Rnd'][3][0]})" for n in best_second])
        return list({best_start[0] if best_start else None, best_second[0] if best_second else None} - {None})

    def evaluate_agent(self, num_episodes,current_version,opponents,include_random):
        print(f"V{current_version} evaluation start")
        start_time = time.time()
        opponents_list = opponents if isinstance(opponents,list) else range(current_version,max(-1,current_version-opponents),-1)
        results = {}
        env = self.environment_bucket.get_environment()
        agent = copy.deepcopy(self.past_versions[current_version]['agent'])
        opponents = dict()
        if include_random:
            opponents['Rnd'] = RandomTicTacToePlayer()
        for v in opponents_list:
            opponents[f"V{v}"] = copy.deepcopy(self.past_versions[v]['agent'])

        
        for name, opponent in opponents.items():
            counts = [0,[0,0,0],[0,0,0]]
            rs = ""
            for no in range(num_episodes):
                agent_is_first = (no%2==0)#(np.random.rand() < 0.5)
                time_step = env.tf.reset()
                env.py.set_next_player(0 if agent_is_first else 1)
                counts[0] += (1 if agent_is_first else 0)
                fo=f"{name}[{no}]:{'A' if agent_is_first else 'O'}:"
                while not time_step.is_last():
                    if env.py.current_player == env.py.agent_player:
                        action_step = agent.policy.action(time_step)
                    else:
                        action_step = opponent.policy.action(time_step)

                    time_step = env.tf.step(action_step.action)
                    fo+=f", {int(action_step.action)}({time_step.reward.numpy()[0]})"
                        
                
                fo+=f":{float(time_step.reward)}"
                count_bucket = 1-self.sign(time_step.reward)
                counts[2-agent_is_first][count_bucket]+=1
                if agent_is_first==0:
                    rs += "w" if time_step.reward > 0 else "l" if time_step.reward < 0 else "d"
                else:
                    rs += "W" if time_step.reward > 0 else "L" if time_step.reward < 0 else "D"
                # print(fo)
                # print(env.py._state)

            results[name] = [num_episodes,counts[0],counts[1],counts[2]]
            print(f"V{current_version} {name}:{rs}")

        self.past_versions[current_version]['evaluation_results']= results
        # xxx = {k:([[f'{ii:.2f}' for ii in i] if isinstance(i,list) else i for i in v]) for k,v in results.items()}
        print(f"V{current_version} evaluation results: {results}")
        end_time = time.time()
        print(f"V{current_version} evaluation duration: {end_time-start_time:.2f}")

        self.environment_bucket.return_environment(env)
        return results


# Initialize the trainer
trainer = TicTacToeAgentTrainer(fc_layer_params=(75,10),learning_rate=1e-5,buffer_max_length=10000, num_steps=5)

# evaluation_history = trainer.train(random_epochs=5, training_epochs=25, iterations=100, batch_size=64, evaluation_num_episodes=100)
# trainer.train(random_epochs=1, training_epochs=120, iterations=500, batch_size=64, evaluation_num_episodes=50)
trainer.train(random_epochs=5, training_epochs=10000, iterations=20, batch_size=64, evaluation_num_episodes=50)
