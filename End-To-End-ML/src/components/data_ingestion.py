# Data ingestion is process of importing or collecting data from various sources and preparing it for analysis or storage.

import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split

# This used to create class variable without using __init__() it manages that
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer

#Using dataclass we are able to directly define variables in class without need of traditional method

@dataclass
class DataIngestionConfig:
    train_data_path : str= os.path.join('artifacts','train.csv')
    test_data_path : str= os.path.join('artifacts','test.csv')
    raw_data_path : str= os.path.join('artifacts','data.csv')



class DataIngestion:
    """
    In this class actuall ingestion process happens of collecting data and storing them. It returns train & test .csv files paths
    1) We make folder artifacts folder
    2) We add collected raw data into data.csv of artifacts
    3) We add training data to train.csv
    4) We add test data to test.csv
    """
    
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Entered in the data ingestion method or component')
        try:
            df = pd.read_csv('notebook\data\StudentsPerformance.csv')
            logging.info('Read the dataset as dataframe')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index = False, header=True)
            logging.info('Train test split initiated')


            train_set, test_set = train_test_split(df,test_size=0.2,random_state=43)
            train_set.to_csv(self.ingestion_config.train_data_path,index = False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index = False, header=True)

            logging.info('Ingestion of data completed')
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data,test_data)
    
    model_trainig = ModelTrainer()
    best_model = model_trainig.initiate_model_trainig(train_arr,test_arr)
    print(best_model)
    
    # To check the result
    # print(train_arr[0])
    # print(train_arr.shape,test_arr.shape)