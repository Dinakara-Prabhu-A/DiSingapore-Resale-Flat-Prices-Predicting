import os
import sys
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifact", "model.pkl.gz")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Starting model training process.")

            # Split features and target
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Define models and hyperparameter grids
            models = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(),
                "Lasso Regression": Lasso(),
                "Decision Tree Regressor": DecisionTreeRegressor(),
                "Gradient Boosting Regressor": GradientBoostingRegressor(),
                "K-Nearest Neighbors Regressor": KNeighborsRegressor(),
                "XGBoost Regressor": XGBRegressor(),
            }

            param_grids = {
                "Ridge Regression": {"alpha": [0.1, 1.0, 10.0]},
                "Lasso Regression": {"alpha": [0.1, 1.0, 10.0]},
                "Decision Tree Regressor": {"max_depth": [2, 4, 6], "min_samples_split": [2, 5]},
                "Gradient Boosting Regressor": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1],
                },
                "K-Nearest Neighbors Regressor": {"n_neighbors": [3, 5]},
                "XGBoost Regressor": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1],
                    "max_depth": [3, 5],
                },
            }

            # Evaluate models and find the best one
            report, best_params = evaluate_models(X_train, y_train, X_test, y_test, models, param_grids)

            best_model_name = max(report, key=report.get)
            best_model = models[best_model_name]
            best_model.set_params(**best_params.get(best_model_name, {}))
            best_model.fit(X_train, y_train)

            # Save the best model
            save_object(self.model_trainer_config.trained_model_file_path, best_model)
            logging.info(f"Best model saved: {best_model_name}")

            return {
                "best_model_name": best_model_name,
                "best_model_score": report[best_model_name],
                "all_model_reports": report,
            }

        except Exception as e:
            raise CustomException(e, sys)