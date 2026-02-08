# CityLearn v2 Developer Notes

This document provides a quick reference and guide for developing with CityLearn v2.

## 1. Initializing an Environment

There are two primary ways to initialize a CityLearn environment.

### 1.1 Using Built-in Datasets

CityLearn comes with several pre-packaged datasets. You can load them by passing their string name to the `CityLearnEnv` constructor. The dataset will be automatically downloaded on the first use.

```python
from citylearn.citylearn import CityLearnEnv

# Example: Load one of the 2020 challenge datasets
env = CityLearnEnv('citylearn_challenge_2020_climate_zone_1') 
```

### 1.2 Using a Custom Schema File

For custom setups, you define your environment in a `schema.json` file. This file points to all your data files (weather, building loads, etc.). You then initialize the environment by providing the path to this schema.

```python
from citylearn.citylearn import CityLearnEnv

# Example: Load an environment from a custom schema file
schema_path = 'path/to/your/custom_dataset/schema.json'
env = CityLearnEnv(schema_path)
```

## 2. Datasets & Data Handling

### 2.1 List of Built-in Datasets

You can programmatically get a list of all available built-in dataset names.

```python
from citylearn.data import DataSet

dataset_names = DataSet().get_dataset_names()
print("Available CityLearn Datasets:")
for name in sorted(dataset_names):
    print(f"- {name}")
```
This is the best way to find the "shortcuts" you mentioned.

### 2.2 Training vs. Validation/Testing Data

CityLearn doesn't have a built-in train/test split function. The common practice is to use different datasets or different time-series periods for training and evaluation. For example, the CityLearn challenges often provide separate datasets for different phases (e.g., `citylearn_challenge_2022_phase_1`, `citylearn_challenge_2022_phase_2`).

For your own data, you would typically:
1.  Create a full time-series dataset.
2.  Create two `schema.json` files:
    *   `train_schema.json`: Sets the simulation period to the first part of the time-series.
    *   `test_schema.json`: Sets the simulation period to the latter part of the time-series.

### 2.3 Creating a Custom Dataset

To create your own dataset, you need to structure your data into several CSV files and a master `schema.json` that orchestrates them. The key files are:
- **Building Data (`building.csv`):** Time-series of building-specific data (loads, solar generation, occupancy).
- **Weather Data (`weather.csv`):** Time-series of outdoor weather variables.
- **Carbon Intensity Data (`carbon_intensity.csv`):** Time-series of CO2 emission rates.
- **Pricing Data (`pricing.csv`):** Time-series for electricity prices.

The `schema.json` then points to these files and defines simulation parameters.

## 3. Environment Details & Visualization

### 3.1 Visualizing the Schema (CityLearn UI)

The official **CityLearn UI** is a web-based dashboard that allows you to visually inspect, create, and edit `schema.json` files. This is the best way to understand the structure of an environment. You can find it on the CityLearn GitHub page.

### 3.2 Inspecting Observation and Action Spaces

CityLearn follows the Gymnasium (formerly Gym) API. You can inspect the observation and action spaces directly from the environment object. This is crucial for designing your agent.

```python
env = CityLearnEnv('citylearn_challenge_2020_climate_zone_1')

# Get the number of buildings (agents)
num_agents = len(env.buildings)
print(f"Number of agents: {num_agents}")

# Inspect the observation space for the first agent
obs_space = env.observation_space[0]
print(f"Observation space for agent 0: {obs_space}")
print(f"Observation space shape: {obs_space.shape}")

# Inspect the action space for the first agent
act_space = env.action_space[0]
print(f"Action space for agent 0: {act_space}")
print(f"Action space shape: {act_space.shape}")
```

### 3.3 Technical Breakdown: Observation Space (Final)

The structure of the observation space was the source of previous errors. The diagnostic script clarified the exact format.

**`CartPole-v1` Observation (for comparison):**
- **Format:** A single NumPy array, e.g., `array([-0.02, 0.01, 0.03, -0.04])`.
- **Access:** Use a numerical index, e.g., `pole_angle = obs[2]`.

**CityLearn Decentralized Observation (`citylearn_challenge_2020_climate_zone_1`):**
- **`env.reset()` Return Format:** A `tuple` of length 2. The actual observations are the first element.
  - `obs_tuple = env.reset()`
  - `all_observations = obs_tuple[0]`
- **`all_observations` Format:** A `list` where each item is the observation for one building (agent).
  - `obs_for_building_1 = all_observations[0]`
- **Single Observation Format:** A flat `list` of 28 numbers.
- **Access:** You must use the correct numerical index for the value you want.

**Observation Index Mapping:**
Based on the `schema.json` for this dataset, the 28 values in the observation list for each agent are:

*   **Shared Observations (Indices 0-19):**
    *   `0`: `month`
    *   `1`: `day_type`
    *   `2`: `hour`
    *   `3`: `outdoor_dry_bulb_temperature`
    *   `4`: `outdoor_dry_bulb_temperature_predicted_1`
    *   `5`: `outdoor_dry_bulb_temperature_predicted_2`
    *   `6`: `outdoor_dry_bulb_temperature_predicted_3`
    *   `7`: `outdoor_relative_humidity`
    *   `8`: `outdoor_relative_humidity_predicted_1`
    *   `9`: `outdoor_relative_humidity_predicted_2`
    *   `10`: `outdoor_relative_humidity_predicted_3`
    *   `11`: `diffuse_solar_irradiance`
    *   `12`: `diffuse_solar_irradiance_predicted_1`
    *   `13`: `diffuse_solar_irradiance_predicted_2`
    *   `14`: `diffuse_solar_irradiance_predicted_3`
    *   `15`: `direct_solar_irradiance`
    *   `16`: `direct_solar_irradiance_predicted_1`
    *   `17`: `direct_solar_irradiance_predicted_2`
    *   `18`: `direct_solar_irradiance_predicted_3`
    *   `19`: `carbon_intensity`
*   **Building-Specific Observations (Indices 20-27):**
    *   `20`: `indoor_dry_bulb_temperature`
    *   `21`: `indoor_relative_humidity`
    *   `22`: `non_shiftable_load`
    *   `23`: `solar_generation`
    *   `24`: `cooling_storage_soc`
    *   `25`: `dhw_storage_soc`
    *   **`26`: `electrical_storage_soc`**
    *   **`27`: `net_electricity_consumption`**

- **Example Access:**
  - `current_soc = obs_for_building_1[26]`
  - `current_net_consumption = obs_for_building_1[27]`

### 3.4 Technical Breakdown: Action Space

**`CartPole-v1` Action:**
- **Format:** A single integer.
- **Example:** `1`
- **Structure:** Each integer has a fixed meaning.
  - `0`: Push cart to the Left
  - `1`: Push cart to the Right

**CityLearn Decentralized Action:**
- **Format:** You must provide a `list` of actions to `env.step()`, one for each building.
  - `all_actions = [action_for_building_1, action_for_building_2, ...]`
- **Structure of a Single Action:** The action for one building is *also* a list of numbers. The length of this list depends on the number of controllable devices for that building, as defined in the `schema.json` under the `"actions"` key.
- **Example from `citylearn_challenge_2020_climate_zone_1`:**
  - The active actions are `cooling_storage`, `dhw_storage`, and `electrical_storage`, in that order.
  - An action for one building must therefore be a list of 3 numbers, where each value is typically between -1.0 and 1.0.
  - `[-1.0, 0.0, 0.5]` would mean:
    - `cooling_storage`: -1.0 (maximum discharge/use)
    - `dhw_storage`: 0.0 (do nothing)
    - `electrical_storage`: 0.5 (charge at 50% of nominal power)

## 4. Customizing the Environment

### 4.1 Custom Reward Functions

You can create your own reward function by defining a Python class and referencing it in your `schema.json`.

**Step 1: Create your custom reward class in a Python file (e.g., `my_rewards.py`)**

```python
# my_rewards.py
from citylearn.reward_function import RewardFunction

class MyCustomReward(RewardFunction):
    def __init__(self, env):
        super().__init__(env)
    
    def calculate(self):
        # Custom reward logic here.
        # Example: Reward for low grid consumption
        # env.net_electricity_consumption is a list of consumption for each building
        total_consumption = sum(self.env.net_electricity_consumption)
        
        # Penalize high consumption
        reward = -abs(total_consumption) 
        
        # Return a list of rewards, one for each agent
        return [reward] * len(self.env.buildings)
```

**Step 2: Update your `schema.json` to point to this class.**

```json
{
    "reward_function": {
        "class_name": "MyCustomReward",
        "module_name": "my_rewards" 
    },
    ...
}
```
Make sure `my_rewards.py` is in your Python path.

### 4.2 Overriding Environment Logic

For more advanced customizations, you can subclass `CityLearnEnv` and override its methods.

```python
from citylearn.citylearn import CityLearnEnv

class MyCustomEnv(CityLearnEnv):
    def __init__(self, schema):
        super().__init__(schema)
        # Add any custom initializations here
    
    def step(self, actions):
        # Implement custom logic before or after the default step
        print("Custom logic before step!")
        
        observations, rewards, done, info = super().step(actions)
        
        # Implement custom logic after the step (e.g., modify rewards)
        print("Custom logic after step!")
        
        return observations, rewards, done, info

# Usage
# env = MyCustomEnv('path/to/your/schema.json')
```
This allows you to fundamentally change the simulation's dynamics.
