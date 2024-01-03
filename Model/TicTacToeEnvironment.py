import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

class TicTacToeEnvironment(py_environment.PyEnvironment):
    def __init__(self):
        # The state will be a 1D array of 9 elements (3x3 board), each can be 0 (empty), 1 (player 1), or 2 (player 2)
        self._state = np.zeros((3, 3), dtype=np.int32)
        self._episode_ended = False
        self._current_player = 1

        # Define the action and observation specs
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=8, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(3, 3), dtype=np.int32, minimum=0, maximum=2, name='observation')

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = np.zeros((3, 3), dtype=np.int32)
        self._episode_ended = False
        self._current_player = 1
        return ts.restart(np.array(self._state, dtype=np.int32))

    def _step(self, action):
        # Check if the game is already over
        if self._episode_ended:
            return self.reset()

        # Action is the index of the cell where the player makes their move (0-8)
        row, col = divmod(action, 3)

        # Check if the action is invalid (cell already filled)
        if self._state[row][col] != 0:
            return ts.termination(np.array(self._state, dtype=np.int32), reward=-1)

        # Apply the action
        self._state[row][col] = self._current_player

        # Check for a win or draw
        if self._check_for_win(self._current_player):
            self._episode_ended = True
            return ts.termination(np.array(self._state, dtype=np.int32), reward=1)
        elif np.all(self._state != 0):
            self._episode_ended = True
            return ts.termination(np.array(self._state, dtype=np.int32), reward=0)

        # Switch to the other player
        self._current_player = 2 if self._current_player == 1 else 1

        return ts.transition(np.array(self._state, dtype=np.int32), reward=0)

    def _check_for_win(self, player):
        # Check rows, columns, and diagonals for a win
        for i in range(3):
            if all(self._state[i, :] == player) or all(self._state[:, i] == player):
                return True
        if self._state[0, 0] == self._state[1, 1] == self._state[2, 2] == player or \
           self._state[0, 2] == self._state[1, 1] == self._state[2, 0] == player:
            return True
        return False
