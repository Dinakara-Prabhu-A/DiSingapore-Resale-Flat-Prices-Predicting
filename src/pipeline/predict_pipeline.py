import sys
import pandas as pd
from src.exception import CustomException
from sklearn.exceptions import NotFittedError
from src.utils import load_object
import os

class PredictPipeline:
    def __init__(self):
        self.model_path = 'artifact/model.pkl.gz'
        self.preprocessor_path = 'artifact/preprocessor.pkl'

    def predict(self, features):
        try:
            # Load the model and preprocessor
            model = load_object(file_path=self.model_path)
            preprocessor = load_object(file_path=self.preprocessor_path)

            # Validate that the model has been fitted
            if not hasattr(model, "predict"):
                raise NotFittedError("The loaded model is not fitted. Train and save the model before using it.")

            # Scale the features and make predictions
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except NotFittedError as e:
            raise CustomException(f"Model Error: {e}", sys)
        except Exception as e:
            raise CustomException(e, sys)
        

class CustomData:
    def __init__(  self,
        town: str,
        flat_type:str,
        block:int,
        floor_area_sqm:int,
        flat_model:str,
        remaining_lease:int,
        year:int,
        storey_start:int,
        storey_end:int):

        self.town = town

        self.flat_type = flat_type

        self.block = block

        self.floor_area_sqm = floor_area_sqm

        self.flat_model = flat_model

        self.remaining_lease = remaining_lease

        self.year = year

        self.storey_start = storey_start

        self.storey_end = storey_end

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "town": [self.town],
                "flat_type": [self.flat_type],
                "block": [self.block],
                "floor_area_sqm": [self.floor_area_sqm],
                "flat_model": [self.flat_model],
                "remaining_lease": [self.remaining_lease],
                "year": [self.year],
                "storey_start": [self.storey_start],
                "storey_end": [self.storey_end]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)