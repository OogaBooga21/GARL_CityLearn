import types
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
import numpy as np
from citylearn.citylearn import CityLearnEnv
from custom_rewards import GridConsumptionReward
import config
from pathlib import Path
from utils import copy_output_files # Import copy_output_files


def _fast_community_as_dict(self):
    """
    O(1) replacement for CityLearnEnv.as_dict().

    The default implementation calls total_self_consumption, energy_to_electrical_storage
    and solar_generation — all of which rebuild a full cross-building DataFrame on every
    call, producing O(n²) simulation time as the episode grows. Those columns are not
    consumed by kpi_calculator, so we skip them here.
    """
    if not self.net_electricity_consumption:
        idx = 0
    else:
        idx = max(0, min(self.time_step, len(self.net_electricity_consumption) - 1))
    return {
        "Net Electricity Consumption-kWh": self.net_electricity_consumption[idx],
        "Self Consumption-kWh": 0,
        "Stored energy by community- kWh": 0,
        "Total Solar Generation-kWh": 0,
        "CO2-kg_co2": self.net_electricity_consumption_emission[idx],
        "Price-$": self.net_electricity_consumption_cost[idx],
    }

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
        # Store reference to unwrapped environment
        self._base_env = env

    def step(self, action):
        # Clip actions to ensure they're within [-1, 1] range
        action = np.clip(action, -1.0, 1.0)
        
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

    @property
    def terminated(self):
        """
        Expose the terminated property from the base environment.
        """
        return self._base_env.terminated

    def close(self):
        """
        Closes the wrapped environment properly to trigger rendering.
        """
        if hasattr(self, '_base_env') and self._base_env is not None:
            return self._base_env.close()
        return self.env.close()

def run_ppo_training(schema_path, run_id: str):
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

    # Close training environment
    train_env.close()

    print("PPO training finished.")

def run_ppo_evaluation(schema_path, run_id: str):
    """
    Evaluates a trained PPO agent.
    """
    print("\n--- PPO Evaluation ---")

    # Use the run_id to create a unique output directory
    output_dir = Path(config.BASE_OUTPUT_DIR) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_env = CityLearnEnv(
        schema_path,
        central_agent=False,
        episode_time_steps=config.EPISODE_TIME_STEPS,  # CRITICAL: Set episode length
        reward_function=GridConsumptionReward,
        render_mode='end', # Ensure render_mode is set to 'end'
        render_directory=Path.cwd() / Path(config.BASE_OUTPUT_DIR), # Base directory for all runs
        render_session_name=run_id # This will create the unique subdirectory
    )
    eval_env.as_dict = types.MethodType(_fast_community_as_dict, eval_env)

    eval_env.buildings = [eval_env.buildings[0]]
    
    # Keep reference to base environment before wrapping
    base_eval_env = eval_env
    eval_env = SingleBuildingEnvWrapper(eval_env)

    # Load the trained PPO agent
    agent = PPOAgent(eval_env)
    agent.load(config.PPO_MODEL_PATH)
    
    # Run evaluation simulation - CRITICAL: Run until environment terminates naturally
    observations = eval_env.reset()[0]

    step_count = 0
    # Use the same pattern as RBC: run until the environment is terminated
    while not eval_env.terminated:
        actions = agent.predict(observations)
        observations, rewards, terminated, truncated, info = eval_env.step(actions)
        step_count += 1
        
        # Safety check to prevent infinite loop
        if step_count >= config.EPISODE_TIME_STEPS:
            print(f"Warning: Reached maximum steps ({config.EPISODE_TIME_STEPS}) but episode not terminated")
            break

    print(f"PPO evaluation ran for {step_count} steps")
    print(f"Episode terminated: {eval_env.terminated}")
    
    # CRITICAL: Close the environment to trigger rendering
    eval_env.close()
    
    print(f"PPO evaluation finished. Simulation data saved to {output_dir}")
    
    # Return the base environment for KPI calculation
    return base_eval_env