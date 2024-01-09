import numpy as np
from tf_agents.trajectories import policy_step

class RandomTicTacToePlayer:

    @property
    def policy(self):
        return self

    def action(self, time_step):
        valid_actions = self.get_valid_actions(time_step.observation[0])
        action = np.random.choice(valid_actions)
        return policy_step.PolicyStep(action)

    def get_valid_actions(self, board_state):
        return [i for i, x in enumerate(board_state.numpy().flatten()) if x == 0]
