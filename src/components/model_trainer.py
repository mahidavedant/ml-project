"""
src/components/model_trainer.py
"""
import sys
import os
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_filepath = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Split training and testing input data.")
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            # Dictionary of models
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors": KNeighborsRegressor(),
                "XGBoost": XGBRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False),
            }

            # models evalution scores
            model_report = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            # best model score and name from dict
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            # thresold for best model score
            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info(
                "Best found model on best training and testing dataset.")

            # save model object
            save_object(
                filepath=self.model_trainer_config.trained_model_filepath,
                obj=best_model
            )

            # Prediction test
            predicted = best_model.predict(X_test)
            r_square = r2_score(y_test, predicted)

            return r_square

        except Exception as e:
            raise CustomException(e, sys)
