import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

class PPOAgent:
    """
    A PPO agent that uses the stable-baselines3 library.
    """
    def __init__(self, env):
        """
        Initializes the PPO agent.
        """
        self.env = DummyVecEnv([lambda: env])
        self.model = PPO("MlpPolicy", self.env, verbose=1)

    def learn(self, total_timesteps):
        """
        Trains the PPO model.
        """
        self.model.learn(total_timesteps=total_timesteps)

    def predict(self, obs):
        """
        Returns an action for the given observation.
        """
        action, _ = self.model.predict(obs, deterministic=True)
        return action

    def save(self, path):
        """
        Saves the trained model.
        """
        self.model.save(path)

    def load(self, path):
        """
        Loads a trained model.
        """
        self.model = PPO.load(path, env=self.env)