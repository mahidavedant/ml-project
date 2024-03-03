"""
src/utils.py

Module for utility functions in the project.
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
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


def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for i in range(len(list(models))):
            # train
            model = list(models.values())[i]
            model.fit(X_train, y_train)
            # prediction
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            # evaluation
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            # append in report dict
            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
