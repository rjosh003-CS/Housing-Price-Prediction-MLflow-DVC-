from itertools import product
import joblib
from pathlib import Path

import warnings
import sys
import os
import yaml

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import mlflow
import mlflow.sklearn

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def load_data(data_path):
    try:
        df = pd.read_csv(data_path)
        return df
    except Exception as e:
        print(f"Error loading dataset from {data_path}: {e}", file=sys.stderr)
        sys.exit(1)

def metric_log(**kwargs):
    # Log parameters and metrics to MLflow Tracking
    for key, value in kwargs.items():
        if key in ["n_estimators", "max_depth"]:
            mlflow.log_param(key, value)
        else:
            mlflow.log_metric(key, value)

def objective(trial, train_x, train_y, test_x, test_y):
    n_estimators = trial.suggest_int("n_estimators", 10, 200)
    max_depth = trial.suggest_int("max_depth", 1, 20)

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )

    with mlflow.start_run(nested=True):
        rf.fit(train_x, train_y.values.ravel())
        predicted = rf.predict(test_x)
        rmse, mae, r2 = eval_metrics(test_y, predicted)

        # Log metrics and parameters to MLflow Tracking
        metric_log(n_estimators=n_estimators, max_depth=max_depth,
                   rmse=rmse, mae=mae, r2=r2)

        return rmse

def load_split_data(train_path, test_path, target_feature):
    if target_feature is None:
        raise ValueError("Target feature name must be provided for splitting the dataset.") 
    if not isinstance(target_feature, list):
        target_feature = [target_feature]
    if train_path is None or test_path is None:
        raise ValueError("Train and test paths must be provided for loading the dataset.")   
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

def optuna_tuning(train_x, train_y, test_x, test_y):
    import optuna
    mlflow.set_experiment("DVC_MLflow_Optuna_Optimization")
    
    with mlflow.start_run(run_name="Optuna_Optimization_Parent"):
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial, train_x, train_y, test_x, test_y), n_trials=10)

        best_params = study.best_params
        print(f"Best parameters found by Optuna: {best_params}")

        # Log best parameters to parent run
        mlflow.log_params(best_params)

        # Retrain Final Model with Best Parameters
        best_n_estimators = best_params["n_estimators"]
        best_max_depth = best_params["max_depth"]

        best_rf = RandomForestRegressor(
            n_estimators=best_n_estimators,
            max_depth=best_max_depth,
            random_state=42
        )
        best_rf.fit(train_x, train_y.values.ravel())

        # Save the final model to disk for DVC output tracking
        os.makedirs("models", exist_ok=True)
        joblib.dump(best_rf, "models/model.pkl")

        # Fixed: Using artifact_path instead of name
        mlflow.sklearn.log_model(
            best_rf,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]    
        )
        print("Final best model trained and saved to models/model.pkl successfully!")


def custom_hyperparameter_tuning(train_x, train_y, test_x, test_y,
                                  n_estimators, max_depth):
    
    #Ensure n_estimators and max_depth are lists for grid search
    if not isinstance(n_estimators, list):
        n_estimators = [n_estimators]
    if not isinstance(max_depth, list):
        max_depth = [max_depth]
    
    # set up MLflow experiment for grid search
    mlflow.set_experiment("DVC_MLflow_GridSearch")

    base_rmse = float("inf")
    base_model = None
    base_params = {}
    
    with mlflow.start_run(run_name="GridSearch_Parent"):
        # Use itertools.product to create a grid of hyperparameter combinations
        for n_estimators, max_depth in product(n_estimators, max_depth):
            with mlflow.start_run(nested=True):
                # initialize Random Forest Regressor with current hyperparameters
                rf = RandomForestRegressor(
                    n_estimators=n_estimators, 
                    max_depth=max_depth, 
                    random_state=42
                )
                # Train the model on the training set
                rf.fit(train_x, train_y.values.ravel())

                # predict on test set and evaluate metrics
                predicted = rf.predict(test_x)
                rmse, mae, r2 = eval_metrics(test_y, predicted)

                # Log on terminal
                print(f"Random Forest Housing Model (n_estimators={n_estimators}, max_depth={max_depth}):")
                print(f"  RMSE: {rmse}")
                print(f"  R2: {r2}")

                # Log metrics and parameters to MLflow Tracking
                metric_log(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    rmse=rmse, mae=mae, r2=r2
                   )

                # Keep track of the best model based on RMSE
                if rmse < base_rmse:
                    base_rmse = rmse
                    base_model = rf
                    base_params = {
                        "n_estimators": n_estimators,
                        "max_depth": max_depth
                    }

        # Log overall best metrics/params to parent run
        mlflow.log_params(base_params)
        mlflow.log_metric("best_rmse", base_rmse)

        # Save model locally for DVC tracking
        os.makedirs("models", exist_ok=True)
        joblib.dump(rf, "models/model.pkl")

        # Fixed: Using artifact_path instead of name
        mlflow.sklearn.log_model(
            rf,
            name="housing_model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )
        print("Housing model training and MLflow logging completed!")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    # Force MLflow to store tracking logs in SQLite database
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    data_path = params["data"]["train_data_path"]
    df = load_data(data_path)

    target_feature_name = params["data"]["target_feature"]
    
    train_x, test_x, train_y, test_y = load_split_data(
        train_path="data/processed/train.csv",
        test_path="data/processed/test.csv",
        target_feature=[target_feature_name]
    )

    use_optuna = params["train"]["use_optuna"]

    if use_optuna:
        optuna_tuning(train_x, train_y, test_x, test_y) 
    else:
        custom_hyperparameter_tuning(
            train_x, train_y, test_x, test_y,
            n_estimators=params["train"]["n_estimators"],
            max_depth=params["train"]["max_depth"]      
        )