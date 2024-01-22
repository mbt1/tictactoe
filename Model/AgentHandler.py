from TicTacToeEnvironment import TicTacToeEnvironment
import numpy as np
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts, trajectory, policy_step

class AgentHandler:

    def __init__(self,environment, agent, replay_buffer, learn_iterations, batch_size, num_steps):
        self._environment = environment
        self._agent = agent
        self._time_step = None
        self._replay_buffer = replay_buffer
        self._tmp_buffer = []
        self._learn_iterations = learn_iterations
        self._batch_size = batch_size
        self._num_steps = num_steps

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
        self._flush_buffer()
        experiences = iter(self.replay_buffer.as_dataset(
            num_parallel_calls=tf.data.AUTOTUNE, sample_batch_size=self._batch_size, num_steps=self._num_steps).prefetch(tf.data.AUTOTUNE))
        for _ in range(self._learn_iterations):
            experience, _ = next(experiences)
            self.agent.train(experience)
    
    def _flush_buffer(self):
        for t in self._tmp_buffer:
            self._replay_buffer.add_batch(t)
        self._tmp_buffer = []
    
    def _append_to_buffer(self,time_step,action_step,next_time_step):
        self._tmp_buffer.append(trajectory.from_transition(time_step, action_step, next_time_step))
    
    def _exec_step(self,action_step,own_turn):
        self._environment.set_next_is_agent_or_opponent(is_agent = own_turn)
        return self._environment.tf.step(action_step.action)



