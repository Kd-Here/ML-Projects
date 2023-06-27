import os 
import sys

import numpy as np
import pandas as pd
import dill
import pickle

from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file:
            pickle.dump(obj,file)
    
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(train_X, train_Y, test_X, test_Y,models,param):
    
    """
    Takes training & testing data with input/output features , with models & hyperparameter
    we run for loop using items() iteration, 
    using GridSearch tune parameter then fit the model
    
    return report which is dictionary with key as model name & value as model r2_score
    """
    try:
        report = {}
        for name, model in models.items():
            
            para=param[name]
            gs = GridSearchCV(model,para,cv=3)
            gs.fit(train_X,train_Y)

            model.set_params(**gs.best_params_)
            model.fit(train_X,train_Y)

            y_train_pred = model.predict(train_X)
            y_test_pred = model.predict(test_X)

            train_model_score = r2_score(train_Y, y_train_pred)
            test_model_score = r2_score(test_Y, y_test_pred)

            report[name] = test_model_score

        return report
    
    except Exception as e:
        raise CustomException(e, sys)

"""
    This function is to retrieve pickle file data using pickle.load()
    The fucntion argument is file_path
"""

def load_object(path):   
    try:
        with open(path,'rb') as file_obj :
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)