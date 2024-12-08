import os
import sys
import pickle
import dill
import requests
import pandas as pd
import time
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import mlflow
import mlflow.sklearn
import warnings
from src.exception import CustomException
from src.logger import logging
import gzip

warnings.filterwarnings("ignore")


def save_object(file_path, obj):
    """Save an object to a compressed pickle file using gzip."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with gzip.open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """Load an object from a compressed pickle file using gzip."""
    try:
        with gzip.open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param_grids):
    """Evaluate multiple models with hyperparameter tuning and log results in MLflow."""
    try:
        report = {}
        best_params = {}

        # Set the MLflow experiment
        mlflow.set_experiment("Resale Price Prediction")
        
        # Start a new MLflow run
        with mlflow.start_run():
            for model_name, model in models.items():
                logging.info(f"Training model: {model_name}")
                
                with mlflow.start_run(nested=True):
                    mlflow.log_param("model_name", model_name)

                    param_grid = param_grids.get(model_name, {})
                    if param_grid:
                        search = GridSearchCV(
                            estimator=model,
                            param_grid=param_grid,
                            scoring="r2",
                            cv=5,
                            n_jobs=-1,
                            verbose=2,
                        )
                        search.fit(X_train, y_train)
                        best_model = search.best_estimator_
                        best_params[model_name] = search.best_params_
                        mlflow.log_params(search.best_params_)
                    else:
                        best_model = model
                        best_model.fit(X_train, y_train)

                    y_train_pred = best_model.predict(X_train)
                    y_test_pred = best_model.predict(X_test)

                    train_r2 = r2_score(y_train, y_train_pred)
                    test_r2 = r2_score(y_test, y_test_pred)
                    report[model_name] = test_r2

                    # Log metrics and model to MLflow
                    mlflow.log_metrics({
                        "train_r2": train_r2,
                        "test_r2": test_r2,
                        "mae": mean_absolute_error(y_test, y_test_pred),
                        "mse": mean_squared_error(y_test, y_test_pred),
                    })
                    mlflow.sklearn.log_model(best_model, model_name)

        return report, best_params

    except Exception as e:
        raise CustomException(e, sys)



class SingaporeData:
    def __init__(self):
        self.base_url = 'https://api-production.data.gov.sg/v2/public/api/'
        self.collection_id = 189
        self.data = self._get_all_records()

    def _get_dataset_ids(self):
        """Fetch dataset IDs from the API."""
        collection_url = f"collections/{self.collection_id}/metadata"
        response = requests.get(self.base_url + collection_url)
        response.raise_for_status()
        return response.json()['data']['collectionMetadata']['childDatasets']

    def _get_all_records(self, max_polls=5, delay=5):
        """Fetch and combine all datasets into a DataFrame."""
        dataset_ids = self._get_dataset_ids()
        dfs = []
        for dataset_id in dataset_ids:
            for _ in range(max_polls):
                response = requests.get(
                    f"https://api-open.data.gov.sg/v1/public/api/datasets/{dataset_id}/poll-download"
                )
                data = response.json().get("data", {})
                if "url" in data:
                    try:
                        dfs.append(pd.read_csv(data["url"]))
                        break
                    except Exception as e:
                        logging.error(f"Error loading dataset {dataset_id}: {e}")
                time.sleep(delay)
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()