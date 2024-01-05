import numpy as np
import tf_agents
from tf_agents.trajectories import policy_step
from TicTacToeEnvironment import TicTacToeEnvironment

class RandomTicTacToePlayer:
    def __init__(self, environment:TicTacToeEnvironment):
        self.environment = environment

    @property
    def policy(self):
        return self

    def action(self, time_step):
        # The action method aligns with the DqnAgent's policy structure
        action = self.select_action()
        return policy_step.PolicyStep(action)

    def select_action(self):
        valid_actions = [i for i, x in enumerate(self.environment._state.flatten()) if x == 0]
        return np.random.choice(valid_actions)
