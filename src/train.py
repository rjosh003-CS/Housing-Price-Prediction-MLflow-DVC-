import argparse
import warnings
import sys

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

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()

    # Load local housing data guaranteed by DVC
    data_path = "data/housing.csv"
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"Error loading dataset from {data_path}: {e}", file=sys.stderr)
        sys.exit(1)

    # Split dataset (Target column in California housing is 'MedHouseValue')
    target_feature_name="MedHouseValue"
    train, test = train_test_split(df, test_size=0.25, random_state=42)
    train_x = train.drop([target_feature_name], axis=1)
    test_x = test.drop([target_feature_name], axis=1)
    train_y = train[[target_feature_name]]
    test_y = test[[target_feature_name]]

    # Start MLflow Tracking
    mlflow.set_experiment("Housing_Price_Experiment")

    with mlflow.start_run():
        # Train model
        rf = RandomForestRegressor(
            n_estimators=args.n_estimators, 
            max_depth=args.max_depth, 
            random_state=42
        )
        rf.fit(train_x, train_y.values.ravel())

        # Evaluate
        predicted = rf.predict(test_x)
        (rmse, mae, r2) = eval_metrics(test_y, predicted)

        print(f"Random Forest Housing Model (n_estimators={args.n_estimators}, max_depth={args.max_depth}):")
        print(f"  RMSE: {rmse}")
        print(f"  R2: {r2}")

        # Log parameters and metrics to MLflow Tracking
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        # Log model format for downstream Model Registry usage
        mlflow.sklearn.log_model(rf, "housing_model")
        print("Housing model training and MLflow logging completed!")