import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
import numpy as np
from citylearn.citylearn import CityLearnEnv
from custom_rewards import GridConsumptionReward
import config

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

class SingleBuildingEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # The action space is the standardized 3-element action
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        # The observation space is the single building's observation space
        self.observation_space = env.observation_space[0]
        self.building_metadata = self.env.buildings[0].action_metadata

    def step(self, action):
        # Translate the standardized action to the building's action space
        building_action = []
        if self.building_metadata['cooling_storage']:
            building_action.append(action[0])
        if self.building_metadata['dhw_storage']:
            building_action.append(action[1])
        if self.building_metadata['electrical_storage']:
            building_action.append(action[2])
        
        # The environment expects a list of actions
        obs, reward, terminated, truncated, info = self.env.step([np.array(building_action)])
        
        # Return the single observation, reward, etc.
        return obs[0], reward[0], terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs[0], info

def run_ppo_training(schema_path):
    """
    Trains a PPO agent.
    """
    # --- Training ---
    # Create a single-building environment for training
    train_env = CityLearnEnv(
        schema_path,
        central_agent=False, # Must be false for custom reward
        reward_function=GridConsumptionReward
    )
    # Select the first building for training
    train_env.buildings = [train_env.buildings[0]]
    train_env = SingleBuildingEnvWrapper(train_env)

    # Create and train the PPO agent
    agent = PPOAgent(train_env)
    agent.learn(total_timesteps=config.PPO_TRAINING_TIMESTEPS)
    agent.save(config.PPO_MODEL_PATH)

    print("PPO training finished.")
    # Note: The environment does not render during training, so no output files are generated yet.
    # This is just a placeholder for when evaluation is added back.
    # copy_output_files(Path('citylearn_output'), 'ppo_training')