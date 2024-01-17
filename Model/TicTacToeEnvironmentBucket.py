from tf_agents.environments import tf_py_environment
from TicTacToeEnvironment import TicTacToeEnvironment
from collections import namedtuple
from queue import Queue, Empty
import threading

class TicTacToeEnvironmentBucket:
    def __init__(self, max_environments, agent_player = 0):
        self.max_environments = max_environments
        self.environments = Queue(max_environments)
        self.environment_count = 0
        self._agent_player = agent_player
        self.lock = threading.Lock()

    def get_counts(self):
        return (self.environments.qsize(),self.environment_count,self.max_environments)
    
    def get_environment(self):
        env = None
        with self.lock:
            try:
                # Try to get an environment without waiting
                env = self.environments.get_nowait()
            except Empty:
                # Only create a new environment if the maximum hasn't been reached
                if self.environment_count < self.max_environments:
                    env = self.create_tic_tac_toe_environment(self._agent_player)
                    self.environment_count += 1

        if env is None:
            # If no environment is available, wait for one to be returned
            env = self.environments.get()


        env.tf.reset()
        return env


    def return_environment(self, env):
        self.environments.put(env)

    def create_tic_tac_toe_environment(self,agent_player):
        py_env = TicTacToeEnvironment(agent_player=agent_player)
        return (namedtuple('env',['py','tf']))(py=py_env,tf=py_env.tf)
