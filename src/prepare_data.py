# src/prepare_data.py

import os
import pandas as pd
from sklearn.datasets import fetch_california_housing

os.makedirs("data", exist_ok=True)
print("Fetching California housing dataset...")
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Save to local CSV file
output_path = "data/housing.csv"
df.to_csv(output_path, index=False)
print(f"Dataset successfully saved to {output_path}")