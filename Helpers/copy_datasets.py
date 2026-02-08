import os
import shutil

# The root directory of the CityLearn data
# THIS IS A HARDCODED PATH - IT MIGHT BREAK IF THE ENVIRONMENT CHANGES
# In a real application, this should be handled more robustly.
CITYLEARN_DATA_ROOT = '/home/oli/.local/share/virtualenvs/EC_RL_py311/lib/python3.11/site-packages/citylearn/data'


# List of datasets to copy
DATASETS_TO_COPY = [
    'citylearn_challenge_2020_climate_zone_1',
    'citylearn_challenge_2020_climate_zone_2',
    'citylearn_challenge_2020_climate_zone_3',
]

# Destination directory
DEST_DIR = 'data'

def copy_datasets():
    """Copies specified CityLearn datasets to a local directory."""
    print("Starting dataset copy process...")
    # Ensure the destination directory exists
    os.makedirs(DEST_DIR, exist_ok=True)

    for dataset_name in DATASETS_TO_COPY:
        try:
            print(f"--> Processing dataset: {dataset_name}")
            
            # Construct the full source path for the dataset
            source_path = os.path.join(CITYLEARN_DATA_ROOT, dataset_name)
            
            # Construct the destination path
            dest_path = os.path.join(DEST_DIR, dataset_name)
            
            # Check if the source directory exists
            if os.path.isdir(source_path):
                # If destination exists, remove it first to avoid errors with copytree
                if os.path.exists(dest_path):
                    shutil.rmtree(dest_path)
                    print(f"    Removed existing directory: {dest_path}")

                # Copy the entire dataset directory
                print(f"    Copying from: {source_path}")
                print(f"    Copying to:   {dest_path}")
                shutil.copytree(source_path, dest_path)
                print(f"    Successfully copied {dataset_name}")
            else:
                print(f"    ERROR: Could not find source directory: {source_path}")

        except Exception as e:
            print(f"    An error occurred while copying {dataset_name}: {e}")
            
    print("\nDataset copy process finished.")

if __name__ == "__main__":
    copy_datasets()