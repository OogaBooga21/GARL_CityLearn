import numpy as np
from pathlib import Path
from citylearn.citylearn import CityLearnEnv
from translation_layer import TranslationLayer
import config # Import config

class SimpleRBC:
    """
    A simple Rule-Based Controller that uses a standardized action space.
    """
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, observations):
        """
        Returns a standardized 3-element action vector [cooling, dhw, electrical].
        The logic is to charge/discharge the battery based on the time of day.
        """
        hour = observations[2]
        
        # Standard action: [cooling, dhw, electrical]
        action = np.zeros(3)
        
        if 7 <= hour < 16:
            # Charge the battery
            action[2] = 1.0
        elif 16 <= hour < 20:
            # Discharge the battery
            action[2] = -1.0
        
        return action

def run_rbc_simulation(schema_path, episode_time_steps: int, central_agent: bool, run_id: str):
    """
    Runs a CityLearn simulation with the given parameters using an RBC agent.
    """
    # Use the run_id to create a unique output directory
    output_dir = Path(config.BASE_OUTPUT_DIR) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    env = CityLearnEnv(
        schema_path,
        central_agent=central_agent,
        episode_time_steps=episode_time_steps,
        render_mode='end',
        render_directory=Path.cwd() / Path(config.BASE_OUTPUT_DIR), # Base directory for all runs
        render_session_name=run_id # This will create the unique subdirectory
    )

    # Initialize the translation layer
    translator = TranslationLayer(env.buildings)

    # Initialize agents
    agents = [SimpleRBC(building.action_space) for building in env.buildings]

    observations, _ = env.reset()
    while not env.terminated:
        # Get standardized actions from agents
        standard_actions = [agent.predict(obs) for agent, obs in zip(agents, observations)]
        
        # Translate actions for the environment
        env_actions = translator.translate_actions(standard_actions)
        
        observations, _, _, _, _ = env.step(env_actions)
    
    env.close() # Ensure environment is closed to finalize output files

    print(f"RBC Simulation finished. Output saved to {output_dir}")