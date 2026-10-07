# src/main.py

import os
from pathlib import Path
import warnings

import mlflow
import numpy as np
import pandas as pd
from preprocess import load_split_raw_data

# Import from your modules
from train import (
    custom_hyperparameter_tuning,
    load_data,
    load_split_data,
    optuna_tuning,
)
import yaml

if __name__ == "__main__":
  warnings.filterwarnings("ignore")
  np.random.seed(42)

  # Load the parameters from params.yaml file
  with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

  # Force MLflow to store tracking logs in SQLite database
  mlflow.set_tracking_uri("sqlite:///mlflow.db")

  # Fetching data paths from params.yaml (Fixed key typo: removed leading space)
  train_data_path = Path(params["data"]["train_data_path"])
  test_data_path = Path(params["data"]["test_data_path"])

  # Fetching the target feature from params.yaml
  target_feature_name = params["data"]["target_feature"]

  # Fixed: Use os.path.exists() properly
  if not os.path.exists(train_data_path) or not os.path.exists(test_data_path):
    print("Processed datasets not found. Running raw data split & prep...")
    load_split_raw_data()

  # Load train and test data and split them into features and targets sets
  train_x, test_x, train_y, test_y = load_split_data(
      train_path=train_data_path,
      test_path=test_data_path,
      target_feature=target_feature_name,
  )

  # Check for optuna hyperparameter tuning selection flag
  use_optuna = params["train"]["use_optuna"]

  if use_optuna:
    print("Starting Optuna optimization workflow...")
    optuna_tuning(train_x, train_y, test_x, test_y)
  else:
    print("Starting Custom Grid Search workflow...")
    custom_hyperparameter_tuning(
        train_x,
        train_y,
        test_x,
        test_y,
        n_estimators=params["train"]["n_estimators"],
        max_depth=params["train"]["max_depth"],
    )