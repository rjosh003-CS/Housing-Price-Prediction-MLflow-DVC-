# src/train.py

from itertools import product
import os
from pathlib import Path
import sys
import yaml

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

# Import our reusable model class
from model import HousingModelTrainer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def load_data(data_path):
    try:
        df = pd.read_csv(data_path)
        return df
    except Exception as e:
        print(f"Error loading dataset from {data_path}: {e}", file=sys.stderr)
        sys.exit(1)


def load_split_data(train_path, test_path, target_feature):
  if target_feature is None:
    raise ValueError(
        "Target feature name must be provided for splitting the dataset."
    )
  if not isinstance(target_feature, list):
    target_feature = [target_feature]
  if train_path is None or test_path is None:
    raise ValueError("Train and test paths must be provided for loading.")
  if not isinstance(train_path, Path):
    train_path = Path(train_path)
  if not isinstance(test_path, Path):
    test_path = Path(test_path)

  train = pd.read_csv(train_path)
  test = pd.read_csv(test_path)

  train_x = train.drop(target_feature, axis=1)
  train_y = train[target_feature]
  test_x = test.drop(target_feature, axis=1)
  test_y = test[target_feature]

  return train_x, test_x, train_y, test_y


def objective(trial, train_x, train_y, test_x, test_y):
  params = {
      "n_estimators": trial.suggest_int("n_estimators", 10, 200),
      "max_depth": trial.suggest_int("max_depth", 1, 20),
      "random_state": 42,
  }

  with mlflow.start_run(nested=True):
    # Use HousingModelTrainer class inside Optuna trial
    trainer = HousingModelTrainer(params)
    trainer.train(train_x, train_y.values.ravel())
    metrics = trainer.evaluate(test_x, test_y)

    # Log metrics and parameters to MLflow Tracking
    trainer.log_to_mlflow(metrics)

    return metrics["rmse"]


def optuna_tuning(train_x, train_y, test_x, test_y):
  import optuna

  mlflow.set_experiment("DVC_MLflow_Optuna_Optimization")

  with mlflow.start_run(run_name="Optuna_Optimization_Parent"):
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, train_x, train_y, test_x, test_y),
        n_trials=10,
    )

    best_params = study.best_params
    best_params["random_state"] = 42
    print(f"Best parameters found by Optuna: {best_params}")

    # Retrain Final Model with Best Parameters using the class
    trainer = HousingModelTrainer(best_params)
    trainer.train(train_x, train_y.values.ravel())
    metrics = trainer.evaluate(test_x, test_y)

    # Save the final model to disk for DVC output tracking
    os.makedirs("models", exist_ok=True)
    joblib = __import__("joblib")
    joblib.dump(trainer.model, "models/model.pkl")

    # Log best model with auto-inferred signature using a training sample subset
    trainer.log_to_mlflow(metrics, X_sample=train_x.head(5))
    print(
        "Final best model trained, evaluated, signed, and saved to"
        " models/model.pkl successfully!"
    )


def custom_hyperparameter_tuning(
    train_x, train_y, test_x, test_y, n_estimators, max_depth
):
  if not isinstance(n_estimators, list):
    n_estimators = [n_estimators]
  if not isinstance(max_depth, list):
    max_depth = [max_depth]

  mlflow.set_experiment("DVC_MLflow_GridSearch")

  base_rmse = float("inf")
  best_trainer = None
  best_params = {}
  best_metrics = {}

  with mlflow.start_run(run_name="GridSearch_Parent"):
    for n_est, m_depth in product(n_estimators, max_depth):
      params = {"n_estimators": n_est, "max_depth": m_depth, "random_state": 42}
      with mlflow.start_run(nested=True):
        # Use HousingModelTrainer class inside Grid Search loop
        trainer = HousingModelTrainer(params)
        trainer.train(train_x, train_y.values.ravel())
        metrics = trainer.evaluate(test_x, test_y)

        print(
            f"Random Forest Housing Model ({params}): RMSE: {metrics['rmse']},"
            f" R2: {metrics['r2']}"
        )

        # Log parameters & metrics for this trial
        trainer.log_to_mlflow(metrics)

        # Track best model
        if metrics["rmse"] < base_rmse:
          base_rmse = metrics["rmse"]
          best_trainer = trainer
          best_params = params
          best_metrics = metrics

    # Log overall best metrics/params to parent run
    mlflow.log_params(best_params)
    mlflow.log_metric("best_rmse", base_rmse)

    # Save best model locally for DVC tracking
    os.makedirs("models", exist_ok=True)
    joblib = __import__("joblib")
    joblib.dump(best_trainer.model, "models/model.pkl")

    # Log best model along with the signature
    best_trainer.log_to_mlflow(best_metrics, X_sample=train_x.head(5))
    print("Housing grid search and MLflow signature logging completed!")