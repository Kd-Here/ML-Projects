""" 
    Prediction_pipeline is for making prediction on data given in html form
    So this will handle process of collecting input-data of user from html form 
    map it with our model variables and preprocess then predict & return prediction to htmlpage 

"""

import sys
import os
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from src.utils import load_object

class PredictPiple:
    def __init__(self):
        pass
       

    # This method is taking model&preprocessor path giving to load_object function & retrived data is model , preprocessor
    def predict(self,features):
        try:
            logging.info('Dataframe is passed to model')
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'

            logging.info('Model & Preprocessor objects are loaded from pickle file')
            preprocessor = load_object(path = preprocessor_path)
            model = load_object(path = model_path)

            # preprocessor is object return from preprocessor.pkl which does transformation to it
            # model is object return from model.pkl
            data_scaled = preprocessor.transform(features)
            prediction = model.predict(data_scaled)
            logging.info('input data is given to model for making prediction')
            return prediction
        
        except Exception as e:
            raise CustomException(e,sys)


class DataCollector:
    logging.info('Userinput data is assigned to class attributes')
    def __init__(self,
                 gender:str,
                 race_ethnicity:str,
                 parental_level_of_education : str,
                 lunch :str,
                 test_preparation_course:str,
                 reading_score : int,
                 writing_score:int
                ) -> None:
        
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score


    # This method creates dict of input data & return them as dataframe
    def give_data_as_dataframe(self):
        try:
            user_input_data = {
                "gender": [self.gender],
                "race/ethnicity": [self.race_ethnicity],
                "parental level of education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test preparation course": [self.test_preparation_course],
                "reading score": [self.reading_score],
                "writing score": [self.writing_score],
            }
            return pd.DataFrame(user_input_data)

        except Exception as e:
            raise CustomException(e,sys)


