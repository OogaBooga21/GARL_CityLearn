
import pandas as pd
from pathlib import Path

def calculate_and_save_kpis(output_dir: Path, kpi_output_dir: Path, env):
    """
    Reads the simulation output from CityLearn, calculates and saves the final KPIs.
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
    
    calculate_and_save_summary_kpis(kpi_output_dir)


def calculate_and_save_summary_kpis(kpi_output_dir: Path):
    """
    Calculates summary KPIs from the simulation results and saves them to a CSV file.
    """
    summary_data = {}

    # Total Cost
    total_cost_df = pd.read_csv(kpi_output_dir / 'total_cost.csv')
    summary_data['total_cost'] = total_cost_df['cost'].sum()

    # Total Carbon Emissions
    try:
        total_carbon_df = pd.read_csv(kpi_output_dir / 'total_carbon_emissions.csv')
        summary_data['total_carbon_emissions'] = total_carbon_df['carbon_emissions'].sum()
    except FileNotFoundError:
        summary_data['total_carbon_emissions'] = 0

    # Max Consumption
    total_grid_consumption_df = pd.read_csv(kpi_output_dir / 'total_grid_consumption.csv')
    summary_data['max_consumption'] = total_grid_consumption_df['grid_consumption'].max()

    # Max Load
    total_load_df = pd.read_csv(kpi_output_dir / 'total_load.csv')
    summary_data['max_load'] = total_load_df['load'].max()

    # Total PV Generation
    total_pv_generation_df = pd.read_csv(kpi_output_dir / 'total_pv_generation.csv')
    summary_data['total_pv_generation'] = total_pv_generation_df['pv_generation'].sum()

    # Battery Charge/Discharge
    electrical_storage_action_df = pd.read_csv(kpi_output_dir / 'electrical_storage_action.csv')
    
    total_charged = 0
    total_discharged = 0

    for building in electrical_storage_action_df.columns[1:]:
        charged = electrical_storage_action_df[electrical_storage_action_df[building] > 0][building].sum()
        discharged = electrical_storage_action_df[electrical_storage_action_df[building] < 0][building].sum()
        
        summary_data[f'{building}_charged'] = charged
        summary_data[f'{building}_discharged'] = discharged
        
        total_charged += charged
        total_discharged += discharged

    summary_data['total_charged'] = total_charged
    summary_data['total_discharged'] = abs(total_discharged)

    # Create DataFrame and save to CSV
    summary_df = pd.DataFrame([summary_data])
    summary_df.to_csv(kpi_output_dir / 'summary_kpis.csv', index=False)

    print("Summary KPIs calculated and saved to 'calculated_kpis/summary_kpis.csv'")
