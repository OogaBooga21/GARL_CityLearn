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

def run_rbc_simulation(schema_path, episode_time_steps: int, central_agent: bool):
    """
    Runs a CityLearn simulation with the given parameters using an RBC agent.
    """
    output_dir = Path(config.BASE_OUTPUT_DIR) # Base output directory
    kpi_output_dir = Path(config.KPI_OUTPUT_DIR)
    
    # Clear output directory before simulation
    if output_dir.exists():
        for item in output_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                import shutil
                shutil.rmtree(item)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = CityLearnEnv(
        schema_path,
        central_agent=central_agent,
        episode_time_steps=episode_time_steps,
        render_mode='end',
        render_directory=Path.cwd() / output_dir, # Files go directly here
        render_session_name='' # Empty string = no subdirectory
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

    # CityLearn might create a timestamp subdirectory even with render_session_name=''
    # So we need to move files from any subdirectories to the main output_dir
    if output_dir.exists():
        for subdir in output_dir.iterdir():
            if subdir.is_dir():
                # Found a subdirectory (likely timestamp-based)
                print(f"Found subdirectory: {subdir.name}, moving files to {output_dir}")
                for file in subdir.iterdir():
                    if file.is_file():
                        # Move file to parent directory
                        import shutil
                        shutil.move(str(file), str(output_dir / file.name))
                # Remove the empty subdirectory
                subdir.rmdir()
                print(f"Moved files from {subdir.name} to {output_dir}")

    print(f"RBC Simulation finished. Output saved to {output_dir}")

    # Files are already in the right location, no need to copy

    # Process the simulation output to calculate and save custom KPIs
    from kpi_calculator import calculate_and_save_kpis
    calculate_and_save_kpis(output_dir, kpi_output_dir, env)