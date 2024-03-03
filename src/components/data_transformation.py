"""
src/components/data_transformation.py
"""
import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_filepath = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    # Method to get the data transformer object
    def get_data_transformer_obj(self):
        """
        Method to get the data transformer object.

        Constructs a data transformer object that preprocesses numerical and
        categorical features using pipelines.
        """
        try:
            numerical_cols = ['writing score', 'reading score']
            categorical_cols = [
                'gender',
                'race/ethnicity',
                'parental level of education',
                'lunch',
                'test preparation course'
            ]

            # Numerical features processing pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            # Categorical features processing pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Numerical columns: {numerical_cols}")
            logging.info(f"Categorical columns: {categorical_cols}")

            # Apply transformations to columns using ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_cols),
                    ('cat_pipeline', cat_pipeline, categorical_cols),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    # Method to initiate data transformation
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed.")
            logging.info("Obtaining prepocessing object.")

            preprocessing_obj = self.get_data_transformer_obj()
            target_column = 'math score'
            numerical_cols = ['writing score', 'reading score']

            # Separate input features and target feature for training data
            input_feature_train_df = train_df.drop(target_column, axis=1)
            target_feature_train_df = train_df[target_column]

            # Separate input features and target feature for testing data
            input_feature_test_df = test_df.drop(target_column, axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info(
                "Applying preprocessing object on training and testing dataframe."
            )

            # Transform training input features using the preprocessing object
            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df)
            # Transform testing input features using the preprocessing object
            input_feature_test_arr = preprocessing_obj.transform(
                input_feature_test_df)

            # Concatenate training input features with the target feature
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            # Concatenate testing input features with the target feature
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info("Saved preprocessing object.")

            # Save the preprocessing object to a file
            save_object(
                filepath=self.data_transformation_config.preprocessor_obj_filepath,
                obj=preprocessing_obj
            )

            # Return the preprocessed training and testing data along with the
            # filepath of the saved preprocessing object
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_filepath
            )

        except Exception as e:
            raise CustomException(e, sys)
