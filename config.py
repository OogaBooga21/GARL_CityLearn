# -*- coding: utf-8 -*-
"""
This file contains the configuration parameters for the CityLearn simulation.
"""

# --- Simulation Parameters ---
DATASET_NAME = 'citylearn_challenge_2020_climate_zone_1'
CENTRAL_AGENT = False
EPISODE_TIME_STEPS = 500
# ---------------------------

# --- Agent Parameters ---
AGENT_TYPE = 'RBC' # 'RBC' or 'PPO'
# ---------------------------

# --- PPO Parameters ---
PPO_TRAINING_TIMESTEPS = 10000
PPO_MODEL_PATH = 'ppo_model.zip'
# ---------------------------
