import numpy as np
from pathlib import Path
from citylearn.citylearn import CityLearnEnv
from translation_layer import TranslationLayer
from utils import copy_output_files

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

def run_rbc_simulation(schema_path, episode_time_steps: int, central_agent: bool):
    """
    Runs a CityLearn simulation with the given parameters using an RBC agent.
    """
    output_dir = Path('citylearn_output')
    kpi_output_dir = Path('calculated_kpis')
    run_name = 'my_first_run'
    
    env = CityLearnEnv(
        schema_path,
        central_agent=central_agent,
        episode_time_steps=episode_time_steps,
        render_mode='end',
        render_directory=output_dir,
        render_session_name=run_name,
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

    print("Simulation finished.")

    # Copy the output files from the CityLearn environment
    copy_output_files(output_dir, run_name)

    # Process the simulation output to calculate and save custom KPIs
    from utils import process_simulation_output
    process_simulation_output(output_dir, kpi_output_dir, env)