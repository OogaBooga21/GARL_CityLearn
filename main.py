from pathlib import Path
import citylearn.data
from citylearn.citylearn import CityLearnEnv
from rbc_agent import SimpleRBC
from stateful_rbc_agent import StatefulRBC
from translation_layer import TranslationLayer
from utils import copy_output_files
import config
from custom_rewards import GridConsumptionReward
from ppo_agent import PPOAgent
import gymnasium as gym
import numpy as np

SCHEMA_PATH = '/home/oli/Documents/Work/EC_RL/schema.json'

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

def run_ppo_training():
    """
    Trains a PPO agent.
    """
    # --- Training ---
    # Create a single-building environment for training
    train_env = CityLearnEnv(
        SCHEMA_PATH,
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


def run_rbc_simulation(episode_time_steps: int, central_agent: bool):
    """
    Runs a CityLearn simulation with the given parameters using an RBC agent.
    """
    output_dir = Path('citylearn_output')
    run_name = 'my_first_run'
    
    env = CityLearnEnv(
        SCHEMA_PATH,
        central_agent=central_agent,
        episode_time_steps=episode_time_steps,
        render_mode='end',
        render_directory=output_dir,
        render_session_name=run_name,
    )

    # Initialize the translation layer
    translator = TranslationLayer(env.buildings)

    # Initialize agents
    # To use the SimpleRBC, uncomment the following line
    # agents = [SimpleRBC(building.action_space) for building in env.buildings]
    # To use the StatefulRBC, uncomment the following line
    agents = [StatefulRBC(building.action_space, building.active_observations) for building in env.buildings]

    observations, _ = env.reset()
    while not env.terminated:
        # Get standardized actions from agents
        standard_actions = [agent.predict(obs) for agent, obs in zip(agents, observations)]
        
        # Translate actions for the environment
        env_actions = translator.translate_actions(standard_actions)
        
        observations, _, _, _, _ = env.step(env_actions)

    print("Simulation finished.")
    copy_output_files(output_dir, run_name)

def main():
    """
    This is the main script to run a CityLearn simulation.
    """
    # available_datasets = citylearn.data.DataSet().get_dataset_names()
    # print("Available datasets:", available_datasets)
    
    if config.AGENT_TYPE == 'RBC':
        run_rbc_simulation(
            episode_time_steps=config.EPISODE_TIME_STEPS,
            central_agent=config.CENTRAL_AGENT
        )
    elif config.AGENT_TYPE == 'PPO':
        run_ppo_training()

if __name__ == '__main__':
    main()