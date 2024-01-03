import numpy as np
from TicTacToeEnvironment import TicTacToeEnvironment

class RandomTicTacToePlayer:
    def __init__(self, environment: TicTacToeEnvironment):
        self._environment = environment

    def select_action(self):
        valid_actions = [i for i, x in enumerate(self._environment._state.flatten()) if x == 0]
        return np.random.choice(valid_actions)
