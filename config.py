# -*- coding: utf-8 -*-
"""
This file contains the configuration parameters for the CityLearn simulation.
"""

# --- Simulation Parameters ---
SCHEMA_PATH = '/home/oli/Documents/Work/EC_RL/schema.json'
DATASET_NAME = 'citylearn_challenge_2020_climate_zone_1'
CENTRAL_AGENT = False
EPISODE_TIME_STEPS = 8760 #8760 max
BASE_OUTPUT_DIR = 'citylearn_output'
# ---------------------------

# --- Agent Parameters ---
AGENT_TYPE = 'RBC' # 'RBC', 'PPO', or 'RBC_2'
# ---------------------------

# --- PPO Parameters ---
PPO_TRAINING_TIMESTEPS = 10000
PPO_MODEL_PATH = 'ppo_model.zip'
KPI_OUTPUT_DIR = 'calculated_kpis'
# ---------------------------
