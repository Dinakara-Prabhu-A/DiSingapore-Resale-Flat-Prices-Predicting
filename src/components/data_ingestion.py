import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from src.utils import SingaporeData
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifact','train.csv')
    test_data_path: str=os.path.join('artifact','test.csv')
    raw_data_path: str=os.path.join('artifact','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingenstion_config = DataIngestionConfig()

    def correct_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform additional data cleaning and transformations.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        logging.info("Starting additional data corrections.")

        try:
            # Add year column from month (assuming 'month' is a date column)
            df['year'] = pd.to_datetime(df['month']).dt.year
            # Calculate remaining lease based on 'lease_commence_date' and 'year'
            df['remaining_lease'] = df['lease_commence_date'] + 99 - df['year']
            # Title-case 'flat_model' column values
            df['flat_model'] = df['flat_model'].str.title()
            # Replace incorrect 'flat_type' category value
            df['flat_type'] = df['flat_type'].replace('MULTI-GENERATION', 'MULTI GENERATION')
            #  split the block column
            df['block'] = df['block'].str.split(r'(\D)', expand=True)[0].astype('int64')
            # adding additional columns
            df[['storey_start', 'storey_end']] = df['storey_range'].str.split("TO", expand=True).astype(int)
            # drop month , lease_commence_date,street_name column
            df.drop(columns=['month','lease_commence_date','street_name','storey_range'], inplace=True)
            logging.info("Additional data corrections completed successfully.")
            return df
        
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or components")
        try:
            
            # Instantiate the class and access the concatenated DataFrame
            # Instantiate the class and access the concatenated DataFrame
            
            singapore = SingaporeData()
            df = singapore.data  # Access the concatenated DataFrame
            logging.info("Read the Dataset as DataFrame")

            # Correct data using the new method
            df = self.correct_data(df)

            # Save raw data after corrections
            os.makedirs(os.path.dirname(self.ingenstion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingenstion_config.raw_data_path, index=False, header=True)

            # Train-test split
            logging.info("Train-test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingenstion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingenstion_config.test_data_path, index=False, header=True)

            logging.info("Train and Test datasets saved successfully")
            return (
                self.ingenstion_config.train_data_path,
                self.ingenstion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)

