import numpy as np
from pathlib import Path
from citylearn.citylearn import CityLearnEnv
from translation_layer import TranslationLayer
import config # Import config

import json
import numpy as np
from pathlib import Path
from citylearn.citylearn import CityLearnEnv
from translation_layer import TranslationLayer
import config # Import config

class IntelligentRBC:
    """
    A more intelligent Rule-Based Controller that uses a standardized action space.
    """
    def __init__(self, action_space, observation_names):
        self.action_space = action_space
        self.observation_indices = {obs: i for i, obs in enumerate(observation_names)}

    def predict(self, observations):
        """
        Returns a standardized 3-element action vector [cooling, dhw, electrical].
        The logic is to charge/discharge the battery based on the time of day,
        solar generation, and electrical storage state of charge.
        """
        hour = observations[self.observation_indices['hour']]
        
        if 'solar_generation' in self.observation_indices:
            solar_generation = observations[self.observation_indices['solar_generation']]
        else:
            solar_generation = 0
            
        electrical_storage_soc = observations[self.observation_indices['electrical_storage_soc']]
        
        # Standard action: [cooling, dhw, electrical]
        action = np.zeros(3)
        
        # Peak hours: 16:00 - 20:00
        if 16 <= hour < 20:
            if electrical_storage_soc > 0.2:
                action[2] = -1.0  # Discharge
        # Off-peak hours: 00:00 - 07:00
        elif 0 <= hour < 7:
            if electrical_storage_soc < 0.8:
                action[2] = 1.0  # Charge
        # Daytime hours: 07:00 - 16:00
        elif 7 <= hour < 16:
            if solar_generation > 0.2 and electrical_storage_soc < 0.95:
                action[2] = 1.0  # Charge

        return action

def run_rbc_2_simulation(schema_path, episode_time_steps: int, central_agent: bool, run_id: str):
    """
    Runs a CityLearn simulation with the given parameters using an RBC agent.
    """
    # Create a unique directory for this run's output
    output_dir = Path(config.BASE_OUTPUT_DIR)
    kpi_output_dir = Path(config.KPI_OUTPUT_DIR) / run_id
    
    # The render_directory is the base, and render_session_name is the unique run_id
    env = CityLearnEnv(
        schema_path,
        central_agent=central_agent,
        episode_time_steps=episode_time_steps,
        render_mode='end',
        render_directory=Path.cwd() / output_dir, 
        render_session_name=run_id 
    )

    # Initialize the translation layer
    translator = TranslationLayer(env.buildings)

    # Initialize agents
    agents = [IntelligentRBC(building.action_space, building.active_observations) for building in env.buildings]

    observations, _ = env.reset()
    while not env.terminated:
        # Get standardized actions from agents
        standard_actions = [agent.predict(obs) for agent, obs in zip(agents, observations)]
        
        # Translate actions for the environment
        env_actions = translator.translate_actions(standard_actions)
        
        observations, _, _, _, _ = env.step(env_actions)
    
    env.close() # Ensure environment is closed to finalize output files

    run_output_dir = output_dir / run_id
    print(f"RBC Simulation finished. Output saved to {run_output_dir}")