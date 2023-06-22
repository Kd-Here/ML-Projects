import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

class ModelTrainerConfig:

    """
    This class is for modeltraining path
    """
    def __init__(self):
        self.model_trainer_path = os.path.join('artifacts','model.pkl')    

class ModelTrainer:

    """
    This class performs actual model training. 
    It has method initiate_model_trainig with dict of models & hyper-parameter.
    We pass them to evaluate method that returns a report for models.
    Consider only model with highest performance score and return it.

    """
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    
    def initiate_model_trainig(self,train_data,test_data):
        try:
            logging.info('Split training and test input data')
            train_X, train_Y, test_X, test_Y = train_data[:,:-1],train_data[:,-1],test_data[:,:-1],test_data[:,-1]



            models = { 
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                    }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            evaluation_result = evaluate_models(train_X, train_Y, test_X, test_Y,models,param = params)
            # evaluate_models takes training,testing data with input /output features , models, parameters & return a dictionary


            best_model_name = max(evaluation_result , key = evaluation_result.get)
            # evaluation_result is dict we find maximum based on values <evaluation_result.get> from dict , returns the key of maximum value 

            best_model_score = evaluation_result[best_model_name]
            # Here with key <best_model_name> with extract value 

            best_model = models[best_model_name]
            # Here with key <best_model_name> we extract value in models dictionary
            # best_model will have model object i.e. LinearRegression() which is value for the best_model_name key

            logging.info('Best model found is %s with evaluation score %s and the model function is %s', best_model_name,best_model_score,best_model)

            if best_model_score < 0.6:
                raise CustomException("No model found")
            
            save_object(
                file_path = self.model_trainer_config.model_trainer_path,
                obj = best_model
            )


            predicted = best_model.predict(test_X)
            r2_square = r2_score(test_Y,predicted)
            return r2_square
        
            
        except Exception as e:
            raise CustomException(e,sys)
