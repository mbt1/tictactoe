from TicTacToeEnvironment import TicTacToeEnvironment
import numpy as np
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts, trajectory, policy_step

class AgentHandler:

    def __init__(self,environment, agent):
        self._environment = environment
        self._agent = agent
        self._time_step = None
        self._buffer = []

    def reset(self):
        self._time_step = self._environment.tf.reset()
    
    def next_action(self, opponent_action):
        if opponent_action is not None:
            action_step=policy_step.PolicyStep(opponent_action)
            self._time_step = self._exec_step(action_step,own_turn=False)

        action_step=self._agent.collect_policy.action(self._time_step)
        next_time_step = self._exec_step(action_step,own_turn=True)
        self._append_to_buffer(self._time_step,action_step,next_time_step)
        self._time_step = next_time_step

    def learn(self):
        return "todo"/0
    
    def _flush_buffer():
        return "todo"/0
        self._buffer = []
    
    def _append_to_buffer(self,time_step,action_step,next_time_step):
        self._buffer.append(trajectory.from_transition(time_step, action_step, next_time_step))
    
    def _exec_step(self,action_step,own_turn):
        self._environment.set_next_is_agent_or_opponent(is_agent = own_turn)
        return self._environment.tf.step(action_step.action)

    def setup_agent(self, fc_layer_params, learning_rate, observation_spec, action_spec, time_step_spec):
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
            optimizer=optimizer,
            observation_and_action_constraint_splitter=TicTacToeEnvironment.observation_and_action_constraint_splitter,
            td_errors_loss_fn=common.element_wise_squared_loss)
            
        return q_net,optimizer,agent
    
    def collect_data(self, env, agent, opponent, num_episodes):
        print(f"...collecting {num_episodes} iterations against opponent {opponent[1]}")
        buffer = []
        for _ in range(num_episodes):
            agent_is_first = (np.random.rand() < 0.5)
            time_step = env.tf.reset()

            env.py.set_next_player(0 if agent_is_first else 1)
            while not time_step.is_last():

                # print(f"Loop: {_}, player: {env.py.current_player}, step: {time_step}")
                if env.py.current_player == env.py.agent_player:
                    
                    action_step = agent.collect_policy.action(time_step)
                    next_time_step = env.tf.step(action_step.action)

                    step_trajectory = trajectory.from_transition(time_step, action_step, next_time_step)
                    buffer.append(step_trajectory)
                    time_step = next_time_step
                else:
                    action_step = (opponent[0]).policy.action(time_step)
                    time_step = env.tf.step(action_step.action)
        return buffer
