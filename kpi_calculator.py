import pandas as pd
from pathlib import Path
import numpy as np

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
    
    # Net Energy Exchange (formerly Grid Consumption)
    net_energy_exchange_df = pd.concat([df[['Net Electricity Consumption-kWh']].rename(columns={'Net Electricity Consumption-kWh': f'Building_{i+1}'}) for i, df in enumerate(all_building_dfs)], axis=1)

    # Energy Export
    energy_export_df = net_energy_exchange_df.copy()
    energy_export_df[energy_export_df > 0] = 0
    energy_export_df = energy_export_df.abs()

    # Energy Imported from Grid
    energy_imported_df = net_energy_exchange_df.copy()
    energy_imported_df[energy_imported_df < 0] = 0

    # --- Read battery files ---
    all_battery_dfs = []
    for bid in building_ids:
        try:
            df = pd.read_csv(output_dir / f'exported_data_building_{bid}_battery_ep0.csv', index_col='timestamp')
            all_battery_dfs.append(df)
        except FileNotFoundError:
            all_battery_dfs.append(pd.DataFrame())
            
    # Action Power
    action_df = pd.concat([df[['Battery (Dis)Charge-kWh']].rename(columns={'Battery (Dis)Charge-kWh': f'Building_{i+1}'}) if not df.empty else pd.DataFrame(index=all_building_dfs[0].index, columns=[f'Building_{i+1}']) for i, df in enumerate(all_battery_dfs)], axis=1)
    action_df = action_df.fillna(0)

    # Non-shiftable Load (Old Load)
    non_shiftable_load_df = pd.concat([df[['Non-shiftable Load-kWh']].rename(columns={'Non-shiftable Load-kWh': f'Building_{i+1}'}) for i, df in enumerate(all_building_dfs)], axis=1)

    # Battery Charging
    battery_charging_df = action_df.copy()
    battery_charging_df[battery_charging_df < 0] = 0

    # New Load (Non-shiftable + Battery Charging)
    load_df = non_shiftable_load_df.add(battery_charging_df)

    # PV Generation
    pv_df = pd.concat([df[['Energy Production from PV-kWh']].rename(columns={'Energy Production from PV-kWh': f'Building_{i+1}'}) for i, df in enumerate(all_building_dfs)], axis=1)
    pv_df = pv_df.abs()
            
    # SOC
    soc_df = pd.concat([df[['Battery Soc-%']].rename(columns={'Battery Soc-%': f'Building_{i+1}'}) if not df.empty else pd.DataFrame(index=all_building_dfs[0].index, columns=[f'Building_{i+1}']) for i, df in enumerate(all_battery_dfs)], axis=1)
    soc_df = soc_df.fillna(0)

    # Battery Discharge
    battery_discharge_df = action_df.copy()
    battery_discharge_df[battery_discharge_df > 0] = 0
    battery_discharge_df = battery_discharge_df.abs()

    # Cost
    price = 0.33 # Default static price
    dynamic_price_df = pd.DataFrame(index=net_energy_exchange_df.index)

    price_file_found = False
    for price_filename in ['price.csv', 'prices.csv']:
        price_filepath = output_dir / price_filename
        if price_filepath.exists():
            try:
                dynamic_price_df = pd.read_csv(price_filepath, index_col='timestamp')
                if 'price' in dynamic_price_df.columns:
                    price = dynamic_price_df['price']
                    price_file_found = True
                    print(f"Using dynamic pricing from {price_filename}")
                    break
                else:
                    print(f"Warning: '{price_filename}' found but no 'price' column. Using static price.")
            except Exception as e:
                print(f"Error reading {price_filename}: {e}. Using static price.")
    
    if not price_file_found:
        try:
            community_df = pd.read_csv(output_dir / f'exported_data_community_ep0.csv', index_col='timestamp')
            if 'Electricity Pricing-$/kWh' in community_df.columns:
                price = community_df['Electricity Pricing-$/kWh']
                print("Using dynamic pricing from exported_data_community_ep0.csv")
            else:
                print("No dynamic pricing found in exported_data_community_ep0.csv. Using static price.")
        except (FileNotFoundError, KeyError):
            print("Could not find community data for dynamic pricing. Using static price.")

    cost_df = net_energy_exchange_df.multiply(price, axis='index')

    # Carbon Emissions
    try:
        community_df = pd.read_csv(output_dir / f'exported_data_community_ep0.csv', index_col='timestamp')
        carbon_intensity = community_df['Carbon Intensity-kg_CO2/kWh']
        carbon_df = net_energy_exchange_df.multiply(carbon_intensity, axis='index')
    except (FileNotFoundError, KeyError):
        print("Could not find carbon intensity data. Carbon emissions will not be calculated.")
        carbon_df = pd.DataFrame(index=net_energy_exchange_df.index)

    # --- Save KPIs ---
    kpi_dfs = {
        'net_energy_exchange': net_energy_exchange_df,
        'energy_export': energy_export_df,
        'energy_imported': energy_imported_df,
        'load': load_df,
        'non_shiftable_load': non_shiftable_load_df,
        'cost': cost_df,
        'carbon_emissions': carbon_df,
        'pv_generation': pv_df,
        'electrical_storage_soc': soc_df,
        'electrical_storage_action': action_df,
        'battery_discharge': battery_discharge_df,
    }

    kpi_output_dir.mkdir(parents=True, exist_ok=True)

    for kpi_name, df in kpi_dfs.items():
        if not df.empty:
            df.to_csv(kpi_output_dir / f'{kpi_name}.csv', index_label='timestamp')
            
            if kpi_name == 'electrical_storage_soc':
                battery_capacities = [b.electrical_storage.capacity if b.electrical_storage is not None else 0 for b in env.buildings]
                total_capacity = sum(battery_capacities)
                
                if total_capacity > 0:
                    weighted_soc = (df.multiply(battery_capacities, axis=1)).sum(axis=1) / total_capacity
                    total_df = pd.DataFrame(weighted_soc, columns=['weighted_average_soc'])
                    total_df.to_csv(kpi_output_dir / f'total_{kpi_name}.csv', index_label='timestamp')
                else:
                    total_df = pd.DataFrame(0, index=df.index, columns=['weighted_average_soc'])
                    total_df.to_csv(kpi_output_dir / f'total_{kpi_name}.csv', index_label='timestamp')
            else:
                total_df = pd.DataFrame(df.sum(axis=1), columns=[kpi_name])
                total_df.to_csv(kpi_output_dir / f'total_{kpi_name}.csv', index_label='timestamp')

    print(f"Custom KPIs processed and saved to: {kpi_output_dir}")
    
    calculate_and_save_summary_kpis(kpi_output_dir)


def calculate_and_save_summary_kpis(kpi_output_dir: Path):
    """
    Calculates summary KPIs from the simulation results and saves them to a CSV file.
    """
    summary_data = {}

    total_cost_df = pd.read_csv(kpi_output_dir / 'total_cost.csv')
    summary_data['total_cost'] = total_cost_df['cost'].sum()

    try:
        total_carbon_df = pd.read_csv(kpi_output_dir / 'total_carbon_emissions.csv')
        summary_data['total_carbon_emissions'] = total_carbon_df['carbon_emissions'].sum()
    except FileNotFoundError:
        summary_data['total_carbon_emissions'] = 0

    total_net_energy_exchange_df = pd.read_csv(kpi_output_dir / 'total_net_energy_exchange.csv')
    summary_data['max_consumption'] = total_net_energy_exchange_df['net_energy_exchange'].max()

    total_load_df = pd.read_csv(kpi_output_dir / 'total_load.csv')
    summary_data['max_load'] = total_load_df['load'].max()
    summary_data['total_load_energy'] = total_load_df['load'].sum()

    total_non_shiftable_load_df = pd.read_csv(kpi_output_dir / 'total_non_shiftable_load.csv')
    summary_data['total_non_shiftable_energy'] = total_non_shiftable_load_df['non_shiftable_load'].sum()

    total_pv_generation_df = pd.read_csv(kpi_output_dir / 'total_pv_generation.csv')
    summary_data['total_pv_generation'] = total_pv_generation_df['pv_generation'].sum()

    try:
        weighted_soc_df = pd.read_csv(kpi_output_dir / 'total_electrical_storage_soc.csv')
        summary_data['average_electrical_storage_soc'] = weighted_soc_df['weighted_average_soc'].mean()
    except FileNotFoundError:
        summary_data['average_electrical_storage_soc'] = 0

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

    summary_df = pd.DataFrame([summary_data])
    summary_df.to_csv(kpi_output_dir / 'summary_kpis.csv', index=False)

    print(f"Summary KPIs calculated and saved to '{kpi_output_dir / 'summary_kpis.csv'}'")