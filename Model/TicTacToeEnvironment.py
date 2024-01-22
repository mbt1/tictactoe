import math
import numpy as np
import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec, BoundedTensorSpec, BoundedArraySpec
from tf_agents.trajectories import time_step as ts

class TicTacToeEnvironment(py_environment.PyEnvironment):
    def __init__(self, agent_player=0):
        # The state will be a 1D array of 9 elements (3x3 board), each can be 0 (empty), 1 (player 1), or 2 (player 2)
        self._state = np.zeros((3, 3), dtype=np.int32)
        self._episode_ended = False
        self._current_player = agent_player
        self._agent_player = agent_player
        self._player_symbols = (1,2)
        self._player_caused_error = None
        self._winning_tuples = [t for j in range(3) for t in [[[j,i] for i in range(3)],[[i,j] for i in range(3)]]]+[[[i,i] for i in range(3)],[[i,2-i] for i in range(3)]]
        self._prime_factors = (2,3,5)

        # Define the action and observation specs
        self._action_spec = BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=8, name='action')
        self._observation_spec = (
            BoundedArraySpec(shape=(3, 3), dtype=np.int32, minimum=0, maximum=2, name='board_observation'),
            BoundedArraySpec(shape=(9, ), dtype=np.int32, minimum=0, maximum=1, name='action_mask')
        )
        self._tf = tf_py_environment.TFPyEnvironment(self)

    @classmethod
    def observation_and_action_constraint_splitter(cls, observation):
        return observation, observation[1]
    
    class FlattenAndConcatenateLayer(tf.keras.layers.Layer):
        def __init__(self):
            super(TicTacToeEnvironment.FlattenAndConcatenateLayer, self).__init__()
            self.flatten = tf.keras.layers.Flatten()

        def call(self, inputs):
            flattened_inputs = [self.flatten(input_tensor) for input_tensor in inputs]
            concatenated_output = tf.keras.layers.Concatenate(axis=-1)(flattened_inputs)
            return concatenated_output


    @property
    def tf(self):
        return self._tf

    @property
    def current_player(self):
        return self._current_player
    
    def set_next_player(self,next_player):
        self._current_player = next_player

    def set_next_is_agent_or_opponent(self,is_agent):
        self.set_next_player(self._agent_player if is_agent else 1 - self._agent_player)

    @property
    def agent_player(self):
        return self._agent_player

    @property
    def agent_caused_error(self):
        return self._player_caused_error == self._agent_player
    
    @property
    def opponent_caused_error(self):
        return self._player_caused_error == (1-self._agent_player)
    
    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def calculate_action_mask(self):
        return np.array([1 if cell == 0 else 0 for cell in self._state.flatten()], dtype=np.int32)
    
    def get_observation(self):
        return (self._state, self.calculate_action_mask())

    def _reset(self):
        self._state = np.zeros((3, 3), dtype=np.int32)
        self._episode_ended = False
        self._current_player = 0
        self._player_caused_error = None
        return ts.restart(self.get_observation())

    def _step(self, action):
        # Check if the game is already over
        if self._episode_ended:
            return self.reset()

        # Action is the index of the cell where the player makes their move (0-8)
        row, col = divmod(action, 3)

        # Check if the action is invalid (cell already filled)
        if self._state[row][col] != 0:
            reward_value = -100 if self._current_player == self._agent_player else 0
            self._player_caused_error = self._current_player
            return ts.termination(self.get_observation(), reward=reward_value)

        pre_move_win_opportunity = self._check_for_win_opportunity(self._current_player)
        # Apply the action
        self._state[row][col] = self._player_symbols[self._current_player]

        # Check for a win or draw
        if self._check_for_win(self._current_player):
            self._episode_ended = True           
            reward_value = 1 if self._current_player == self._agent_player else -2
            return ts.termination(self.get_observation(), reward=reward_value)
        elif np.all(self._state != 0):
            self._episode_ended = True
            reward_value = 0
            return ts.termination(self.get_observation(), reward=reward_value)

        self._current_player = 1 - self._current_player
        opponent_win_opportunity = self._check_for_win_opportunity(self._current_player)

        reward = 0
        if pre_move_win_opportunity or opponent_win_opportunity:
            reward = -1
        return ts.transition(self.get_observation(), reward=reward)

    def _check_for_win(self, player):
        winning_product = self._prime_factors[self._player_symbols[player]]**len(self._winning_tuples[0])
        for wt in self._winning_tuples:
            if math.prod([self._prime_factors[self._state[te[0]][te[1]]] for te in wt]) == winning_product:
                return True
        return False

    def _check_for_win_opportunity(self, player):
        win_opportunity_product = self._prime_factors[self._player_symbols[player]]**(len(self._winning_tuples[0]) -1)*self._prime_factors[0]
        for wt in self._winning_tuples:
            if math.prod([self._prime_factors[self._state[te[0]][te[1]]] for te in wt]) == win_opportunity_product:
                return True
        return False
