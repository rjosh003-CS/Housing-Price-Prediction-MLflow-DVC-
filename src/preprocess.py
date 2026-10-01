import warnings

import os
import sys
import yaml
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # Load params
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    # Load data path and test size from params.yaml
    data_path = params["data"]["data_path"]
    # test size is defined in params.yaml under base section
    test_size = params["base"]["test_size"]

    # Load local housing data guaranteed by DVC
    df = pd.read_csv(data_path)

    # Split dataset into train and test sets
    train, test = train_test_split(
                            df,
                            test_size=test_size,
                            random_state=42
                        )

    # Save train and test sets to CSV files
    processed_data_dir = Path("data/processed")
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    os.makedirs(processed_data_dir, exist_ok=True)
    test.to_csv(processed_data_dir / "test.csv", index=False)
    train.to_csv(processed_data_dir / "train.csv", index=False)
    print(f"Processed data saved to {processed_data_dir}")