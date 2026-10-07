# src/model.py

import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

class HousingModelTrainer:
    def __init__(self, params:dict):
        self.params = params
        self.model = None

    def build_model(self):
        self.model = RandomForestRegressor(**self.params)
        return self

    def train(self, X_train, y_train):
        if self.model is None:
            self.build_model()
        self.model.fit(X_train, y_train)
        return self

    def evaluate(self, X_val , y_val):
        preds = self.model.predict(X_val)
        mse = mean_squared_error(y_val, preds)
        rmse = mse**0.5
        r2 = r2_score(y_val, preds)
        return {"rmse": rmse, "mse": mse, "r2": r2}

    def log_to_mlflow(self, metrics, X_sample=None):
        """
        Logs params, metrics, models, and automatically handles the signature

        if an X_sample DataFram is provided.
        """

        mlflow.log_params(self.params)
        mlflow.log_metrics(metrics)

        signature = None
        if X_sample is not None:
            # Automatically infer signature from sample input and predictions
            predictions = self.model.predict(X_sample)
            signature = mlflow.models.infer_signature(X_sample, predictions)

        # Log model with the generated signature (if available)
        mlflow.sklearn.log_model(
            self.model,
            name="model",
            signature=signature,
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

