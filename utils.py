
import shutil
from pathlib import Path
import citylearn
import numpy as np
import pandas as pd

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

def print_schema_details(env):
    """
    Prints detailed information about the CityLearn environment schema.
    """
    print("\n--- CityLearn Environment Schema Details ---")
    print(f"Total number of buildings: {len(env.buildings)}")

    for i, building in enumerate(env.buildings):
        print(f"\nBuilding {i+1}: {building.name}")
        
        # Actions
        print(f"  Action Space: {building.action_space}")
        print(f"  Available Actions:")
        for action_name, is_active in building.action_metadata.items():
            if is_active:
                print(f"    - {action_name}")
        
        # Storages
        if building.cooling_storage is not None:
            print(f"  Cooling Storage:")
            print(f"    Capacity: {building.cooling_storage.capacity} kWh")
            if hasattr(building.cooling_storage, 'efficiency'):
                print(f"    Efficiency: {building.cooling_storage.efficiency}")
        
        if building.dhw_storage is not None:
            print(f"  DHW Storage:")
            print(f"    Capacity: {building.dhw_storage.capacity} kWh")
            if hasattr(building.dhw_storage, 'efficiency'):
                print(f"    Efficiency: {building.dhw_storage.efficiency}")
            
        if building.electrical_storage is not None:
            print(f"  Electrical Storage:")
            print(f"    Capacity: {building.electrical_storage.capacity} kWh")
            if hasattr(building.electrical_storage, 'efficiency'):
                print(f"    Efficiency: {building.electrical_storage.efficiency}")
            if hasattr(building.electrical_storage, 'nominal_power'):
                print(f"    Nominal Power: {building.electrical_storage.nominal_power} kW")
            
        # PV
        if building.pv is not None:
            print(f"  PV System:")
            print(f"    Nominal Power: {building.pv.nominal_power} kW")
        
        # Other relevant building parameters (can be expanded as needed)
        # print(f"  Building Type: {building.building_type}")
        # print(f"  Area: {building.area} m2")
        # print(f"  Number of Occupants: {building.n_occupants}")
        
    print("\n--- End of Schema Details ---")

def process_simulation_output(output_dir: Path, kpi_output_dir: Path, env):
    """
    Reads the simulation output from CityLearn, combines it with logged actions,
    and calculates and saves the final KPIs.
    """
    num_buildings = len(env.buildings)
    building_ids = [i + 1 for i in range(num_buildings)]

    # --- Read and merge building-level observation files ---
    all_building_dfs = []
    for bid in building_ids:
        try:
            df = pd.read_csv(output_dir / f'exported_data_building_{bid}_ep0.csv', index_col='timestamp')
            all_building_dfs.append(df)
        except FileNotFoundError:
            print(f"Could not find exported_data_building_{bid}_ep0.csv in {output_dir}")
            return
            
    # --- Prepare DataFrames for each KPI ---
    
    # Grid Consumption
    grid_consumption_df = pd.concat([df[['Net Electricity Consumption-kWh']].rename(columns={'Net Electricity Consumption-kWh': f'Building_{i+1}'}) for i, df in enumerate(all_building_dfs)], axis=1)

    # Load
    load_df = pd.concat([df[['Non-shiftable Load-kWh']].rename(columns={'Non-shiftable Load-kWh': f'Building_{i+1}'}) for i, df in enumerate(all_building_dfs)], axis=1)

    # PV Generation
    pv_df = pd.concat([df[['Energy Production from PV-kWh']].rename(columns={'Energy Production from PV-kWh': f'Building_{i+1}'}) for i, df in enumerate(all_building_dfs)], axis=1)
    pv_df = pv_df.abs()

    # --- Read battery files ---
    all_battery_dfs = []
    for bid in building_ids:
        try:
            df = pd.read_csv(output_dir / f'exported_data_building_{bid}_battery_ep0.csv', index_col='timestamp')
            all_battery_dfs.append(df)
        except FileNotFoundError:
            # Not all buildings have batteries
            all_battery_dfs.append(pd.DataFrame())
            
    # SOC
    soc_df = pd.concat([df[['Battery Soc-%']].rename(columns={'Battery Soc-%': f'Building_{i+1}'}) if not df.empty else pd.DataFrame(index=all_building_dfs[0].index, columns=[f'Building_{i+1}']) for i, df in enumerate(all_battery_dfs)], axis=1)
    soc_df = soc_df.fillna(0)

    # Action Power
    action_df = pd.concat([df[['Battery (Dis)Charge-kWh']].rename(columns={'Battery (Dis)Charge-kWh': f'Building_{i+1}'}) if not df.empty else pd.DataFrame(index=all_building_dfs[0].index, columns=[f'Building_{i+1}']) for i, df in enumerate(all_battery_dfs)], axis=1)
    action_df = action_df.fillna(0)

    # Cost
    price = 0.33 # Assuming static price for now
    cost_df = grid_consumption_df.multiply(price)

    # Carbon Emissions
    try:
        community_df = pd.read_csv(output_dir / f'exported_data_community_ep0.csv', index_col='timestamp')
        carbon_intensity = community_df['Carbon Intensity-kg_CO2/kWh']
        carbon_df = grid_consumption_df.multiply(carbon_intensity, axis='index')
    except (FileNotFoundError, KeyError):
        print("Could not find carbon intensity data. Carbon emissions will not be calculated.")
        carbon_df = pd.DataFrame(index=grid_consumption_df.index)

    # --- Save KPIs ---
    kpi_dfs = {
        'grid_consumption': grid_consumption_df,
        'load': load_df,
        'cost': cost_df,
        'carbon_emissions': carbon_df,
        'pv_generation': pv_df,
        'electrical_storage_soc': soc_df,
        'electrical_storage_action': action_df,
    }

    # Create the KPI output directory if it doesn't exist
    kpi_output_dir.mkdir(parents=True, exist_ok=True)
    # Clear the directory
    for item in kpi_output_dir.iterdir():
        if item.is_file():
            item.unlink()

    for kpi_name, df in kpi_dfs.items():
        if not df.empty:
            df.to_csv(kpi_output_dir / f'{kpi_name}.csv', index_label='timestamp')
            
            # Save total KPI
            total_df = pd.DataFrame(df.sum(axis=1), columns=[kpi_name])
            total_df.to_csv(kpi_output_dir / f'total_{kpi_name}.csv', index_label='timestamp')

    print(f"Custom KPIs processed and saved to: {kpi_output_dir}")
