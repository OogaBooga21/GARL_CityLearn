import citylearn.data
from citylearn.citylearn import CityLearnEnv
from utils import print_schema_details
import config
from ppo_agent import run_ppo_training, run_ppo_evaluation
from rbc_agent import run_rbc_simulation
from plot_kpis import generate_plots
from kpi_calculator import calculate_and_save_kpis
from pathlib import Path

SCHEMA_PATH = '/home/oli/Documents/Work/EC_RL/schema.json'

# --- Schema and Dataset Information ---
print("--- Initializing CityLearn Environment for Schema Inspection ---")
temp_env = CityLearnEnv(SCHEMA_PATH)
print_schema_details(temp_env)

print("\n--- Available CityLearn Datasets ---")
available_datasets = citylearn.data.DataSet().get_dataset_names()
for name in sorted(available_datasets):
    print(f"- {name}")
print("--- End of Available Datasets ---\n")
# ------------------------------------

def main():
    """
    This is the main script to run a CityLearn simulation.
    """
    if config.AGENT_TYPE == 'RBC':
        run_rbc_simulation(
            schema_path=SCHEMA_PATH,
            episode_time_steps=config.EPISODE_TIME_STEPS,
            central_agent=config.CENTRAL_AGENT
        )
    elif config.AGENT_TYPE == 'PPO':
        run_ppo_training(schema_path=SCHEMA_PATH)
        eval_env = run_ppo_evaluation(schema_path=SCHEMA_PATH)
        
        # Calculate and save KPIs
        output_dir = Path(config.BASE_OUTPUT_DIR)
        kpi_output_dir = Path(config.KPI_OUTPUT_DIR)
        calculate_and_save_kpis(output_dir, kpi_output_dir, eval_env)

if __name__ == '__main__':
    main()
    generate_plots()