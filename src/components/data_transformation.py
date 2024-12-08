import os
import pickle
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler,LabelEncoder,RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
from src.utils import save_object
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifact', "preprocessor.pkl")

class LabelEnconderTransformer(BaseEstimator,TransformerMixin):
    def __init__(self):
        self.label_encoders = {}
        self.column_names = []

    def fit(self, X, y=None):
        self.column_names = [f"col_{i}" for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=self.column_names)
        for column in X_df.columns:
            le = LabelEncoder()
            le.fit(X_df[column])
            self.label_encoders[column] = le
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X, columns=self.column_names)
        X_transformed = X_df.copy()
        for column in X_df.columns:
            le = self.label_encoders[column]
            X_transformed[column] = le.transform(X_df[column])
        return X_transformed.to_numpy()


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):

        try:
            numeric_column = ['block', 'floor_area_sqm', 'remaining_lease', 'year',
                                'storey_start', 'storey_end']
            discrete_categorical_columns = ['flat_type', 'flat_model']
            continuous_categorical_columns = ['town']

            # numeric pipeline
            num_pipeline = Pipeline([
                ("imputer",SimpleImputer(strategy='median')),
                ('scaler',RobustScaler())
            ])

            # Discrete Categorical pipeline (One-Hot Encoding)
            discrete_cat_pipeline = Pipeline([
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder(handle_unknown='ignore',sparse_output=False))
            ])

            # Continuous Categorical Pipeline (Label Encoding)
            continuous_cat_pipeline = Pipeline([
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("label_encoder",LabelEnconderTransformer())
            ])

            # Combine all pipelines into a single ColumnTransformer
            preprocessor = ColumnTransformer([
                ("num", num_pipeline, numeric_column),
                ("discrete_cat", discrete_cat_pipeline, discrete_categorical_columns),
                ("continuous_cat", continuous_cat_pipeline, continuous_categorical_columns)
            ])

            return preprocessor
        
        except Exception as e:
            print(f"Error in creating transformer: {str(e)}")
            
    def inititate_data_transformer(self,train_path,test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")
            train_df.columns = train_df.columns.str.strip().str.lower()
            test_df.columns = test_df.columns.str.strip().str.lower()

            target_column_name = 'resale_price'

            # if target_column_name not in train_df.columns:
            #   raise Exception(f"Target column '{target_column_name}' not found in training data")
            # if target_column_name not in test_df.columns:
            #   raise Exception(f"Target column '{target_column_name}' not found in testing data")
            
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            preprocessing_obj = self.get_data_transformer_object()
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            save_object(self.data_transformation_config.preprocessor_obj_file_path,preprocessing_obj)
            logging.info(f"Saved preprocessing object.")
            return train_arr,test_arr
            

            pass
        except Exception as e:
            print(f" Error in data transformation: {str(e)}")