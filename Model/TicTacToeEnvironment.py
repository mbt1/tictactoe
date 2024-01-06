import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
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

        # Define the action and observation specs
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=8, name='action')
        self._observation_spec = (
            array_spec.BoundedArraySpec(
                shape=(3, 3), dtype=np.int32, minimum=0, maximum=2, name='board_observation'),
            array_spec.BoundedArraySpec(
                shape=(9, ), dtype=np.int32, minimum=0, maximum=1, name='action_mask')
        )

    @classmethod
    def observation_and_action_constraint_splitter(cls, observation):
        return observation, observation[1]
    
    @property
    def current_player(self):
        return self._current_player
    
    def set_next_player(self,current_player):
        self._current_player = current_player

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

        # Apply the action
        self._state[row][col] = self._player_symbols[self._current_player]

        # Check for a win or draw
        if self._check_for_win(self._current_player):
            self._episode_ended = True           
            reward_value = 1 if self._current_player == self._agent_player else -1
            return ts.termination(self.get_observation(), reward=reward_value)
        elif np.all(self._state != 0):
            self._episode_ended = True
            reward_value = 0
            return ts.termination(self.get_observation(), reward=reward_value)

        # Switch to the other player
        self._current_player = 1 - self._current_player

        return ts.transition(self.get_observation(), reward=0)

    def _check_for_win(self, player):
        symbol = self._player_symbols[player]
        # Check rows, columns, and diagonals for a win
        for i in range(3):
            if all(self._state[i, :] == symbol) or all(self._state[:, i] == symbol):
                return True
        if self._state[0, 0] == self._state[1, 1] == self._state[2, 2] == symbol or \
           self._state[0, 2] == self._state[1, 1] == self._state[2, 0] == symbol:
            return True
        return False
