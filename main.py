import citylearn.data
from citylearn.citylearn import CityLearnEnv
from utils import print_schema_details, generate_run_id
import config
from ppo_agent import run_ppo_training, run_ppo_evaluation
from rbc_agent import run_rbc_simulation
from plot_kpis import generate_plots
from kpi_calculator import calculate_and_save_kpis
from pathlib import Path

# --- Schema and Dataset Information ---
print("--- Initializing CityLearn Environment for Schema Inspection ---")
temp_env = CityLearnEnv(config.SCHEMA_PATH)
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
    run_id = generate_run_id(config.AGENT_TYPE)
    print(f"Generated Run ID: {run_id}")

    # Create unique output directories for this run
    run_output_dir = Path(config.BASE_OUTPUT_DIR) / run_id
    run_kpi_output_dir = Path(config.KPI_OUTPUT_DIR) / run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)
    run_kpi_output_dir.mkdir(parents=True, exist_ok=True)

    if config.AGENT_TYPE == 'RBC':
        run_rbc_simulation(
            schema_path=config.SCHEMA_PATH,
            episode_time_steps=config.EPISODE_TIME_STEPS,
            central_agent=config.CENTRAL_AGENT,
            run_id=run_id  # Pass run_id
        )
        # For RBC, KPIs are calculated within the simulation loop, so we need to tell it where to save
        # This might require a refactor of rbc_agent.py to take the kpi_dir as an argument
        # For now, let's assume the rbc_agent saves to its own output dir, and we'll calculate KPIs after
        env = CityLearnEnv(config.SCHEMA_PATH) # Create a dummy env to pass to kpi calculator
        calculate_and_save_kpis(run_output_dir, run_kpi_output_dir, env)


    elif config.AGENT_TYPE == 'PPO':
        run_ppo_training(schema_path=config.SCHEMA_PATH, run_id=run_id)
        eval_env = run_ppo_evaluation(schema_path=config.SCHEMA_PATH, run_id=run_id)
        
        # Calculate and save KPIs
        calculate_and_save_kpis(run_output_dir, run_kpi_output_dir, eval_env)

if __name__ == '__main__':
    main()
    generate_plots()