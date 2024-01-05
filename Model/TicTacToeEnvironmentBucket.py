from tf_agents.environments import tf_py_environment
from TicTacToeEnvironment import TicTacToeEnvironment
from collections import namedtuple
from queue import Queue
import threading

class TicTacToeEnvironmentBucket:
    def __init__(self, max_environments, agent_player = 0):
        self.max_environments = max_environments
        self.environments = Queue(max_environments)
        self.environment_count = 0
        self._agent_player = agent_player
        self.lock = threading.Lock()

    def get_environment(self):
        with self.lock:
            if self.environments.empty() and self.environment_count < self.max_environments:
                env = self.create_tic_tac_toe_environment(self._agent_player)
                self.environment_count += 1
                print(f"Environment {self.environment_count} created!")
            else:
                # Wait for an environment to become available
                env =  self.environments.get()
            env.tf.reset()
            return env

    def return_environment(self, env):
        with self.lock:
            self.environments.put(env)

    def create_tic_tac_toe_environment(self,agent_player):
        py_env = TicTacToeEnvironment(agent_player=agent_player)
        return (namedtuple('env',['py','tf']))(py=py_env,tf=tf_py_environment.TFPyEnvironment(py_env))
