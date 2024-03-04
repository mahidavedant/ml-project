"""
src/utils.py

Module for utility functions in the project.
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
import dill


def save_object(filepath, obj):
    try:
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path, exist_ok=True)

        with open(filepath, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for i in range(len(list(models))):
            # models and hyperparameters
            model = list(models.values())[i]
            model_params = params[list(models.keys())[i]]

            # grid search for hyperparameter tuning
            grid = GridSearchCV(model, model_params, cv=3)
            grid.fit(X_train, y_train)

            # train model (with best hyperparameters)
            model.set_params(**grid.best_params_)
            model.fit(X_train, y_train)

            # prediction
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # evaluation
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # append in report dict, also add best hyperparameters from gridsearch
            report[list(models.keys())[i]] = (
                test_model_score, grid.best_params_)

        return report

    except Exception as e:
        raise CustomException(e, sys)
