This table explains how the main simulation outputs from CityLearn are calculated at each timestep. These calculations are performed by the CityLearn environment itself.

gemi
---

### Summary of Aggregated KPIs

The following KPIs are calculated by the `kpi_calculator.py` script *after* the simulation is complete. They are aggregations of the timestep data described above.

| Metric Name | Calculation Logic |
| :--- | :--- |
| **Total Net Energy Exchange** | Sum of `Net Energy Exchange` for all buildings. |
| **Total Building Load** | Sum of `Building Load` for all buildings. |
| **Total PV Generation** | Sum of `PV Generation` for all buildings. |
| **Total Electricity Cost** | Sum of `Electricity Cost` for all buildings. |
| **Total Carbon Emissions** | Sum of `Carbon Emissions` for all buildings. |
| **Total Energy Export** | Sum of all negative `Net Energy Exchange` values for all buildings. |
| **Total Battery Discharge** | Sum of all negative `Electrical Storage Action` values for all buildings. |
| **Average Electrical Storage SoC** | Weighted average of `Electrical Storage SoC` across all buildings, based on battery capacity. |
