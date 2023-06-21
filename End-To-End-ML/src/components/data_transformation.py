"""
In ML pipelines chain together multiple steps so that output of each step is 
used as input to next step.
Pipeline is very useful when we want to do many preprocessing steps on data.
eg. When we have dataset with categorical & numerical value with missing values.
    1) Step is fill missing values with impute
    2) OneHotEnconding is applied to Updated dataset on categorical 
    3) StandardScaler is applied to numerical

"""
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import os


class DataTransformationConfig:
    def __init__(self) -> None:
        self.preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    """
    This class is having 2 methods 
    1) Creates pipeline & preprocessor object
    2) Applies preprocessor object on selected features 
    """
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ["writing score", "reading score"]
            categorical_columns = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course",
            ]


            # Pipeline pipe 1
            num_pipeline = Pipeline(
                steps=[
                    # This imputer to fill balnk values with median
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            # Pipeline pipe 2
            cat_pipeline = Pipeline(
                steps=[
                    # This imputer to fill balnk values with mode
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))

                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")
                        
            #Pipeline pipe 3        
            # Using ColumnTransformer which can transform subset of columns with different transforming methods in one command.
            # Here ColumnTransformer class  combine the num_pipeline and cat_pipeline pipelines.
            preprocessor=ColumnTransformer(
                    [
                        ("num_pipeline",num_pipeline,numerical_columns),
                        ("cat_pipelines",cat_pipeline,categorical_columns)

                    ]
                )
            logging.info("Preprocessor pipeline created successfully.")

            return preprocessor



        except Exception as e:
            raise CustomException(e,sys)


    def initiate_data_transformation(self,train_path,test_path):
        
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Read train and test data completed")


            preprocessing_obj=self.get_data_transformer_object()
            logging.info("Obtained preprocessing object")

            target_column_name="math score"
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]
                        
                        
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")


            # np.c_ concatenated the input feature array and the target feature array into a single array with two columns
            # train_arr contains the input features, and the second column contains the target features
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]


            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info(f"Saved preprocessing object.")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e,sys)
