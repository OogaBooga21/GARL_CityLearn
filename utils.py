
import shutil
from pathlib import Path
import citylearn
import numpy as np

def inspect_action_space(env):
    """
    Inspects the action space of the environment and prints information about it.
    """
    print("--- Action Space Inspection ---")
    
    if isinstance(env.action_space, list):
        print("Action space is a list of spaces (decentralized agent).")
        for i, space in enumerate(env.action_space):
            print(f"  Building {i}:")
            print(f"    Action space: {space}")
            print(f"    Sample action: {space.sample()}")
    else:
        print("Action space is a single space (centralized agent).")
        print(f"  Action space: {env.action_space}")
        print(f"  Sample action: {env.action_space.sample()}")

    print("--- End of Action Space Inspection ---")

def get_action_meanings(env):
    """
    Prints a human-readable description of the actions for each building.
    """
    print("--- Action Meanings ---")
    for i, building in enumerate(env.buildings):
        print(f"  Building {i}: {building.name}")
        print(f"    Action space: {building.action_space}")
        print(f"    Action names: {building.action_metadata}")
    print("--- End of Action Meanings ---")

def copy_output_files(output_dir: Path, run_name: str):
    """
    Copies the output files from the CityLearn environment to the specified output directory.

    Args:
        output_dir (Path): The base directory where CityLearn saves output.
        run_name (str): The name of the simulation run.
    """
    try:
        # Construct the source path
        source_path = Path(citylearn.__file__).parent.parent / output_dir / run_name
        destination_path = Path.cwd() / output_dir
        
        # Create destination directory if it doesn't exist
        destination_path.mkdir(parents=True, exist_ok=True)

        # Copy files
        if source_path.exists() and source_path.is_dir():
            for item in source_path.iterdir():
                shutil.copy(item, destination_path)
            print(f"Output files copied to: {destination_path}")
        else:
            print(f"Could not find the output directory at: {source_path}")

    except ImportError:
        print("Could not import citylearn library to determine output path.")
    except Exception as e:
        print(f"An error occurred while copying files: {e}")
