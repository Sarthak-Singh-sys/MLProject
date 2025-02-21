import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.exception import CustomException
from src.logger import logger
import os

@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path= os.path.join(os.getcwd(),"artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer_object(self):
        
        try:
            numerical_features=["writing_score","reading_score"]
            categorical_features=[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("std_scaler",StandardScaler())
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("onehot",OneHotEncoder())
                    ("std_scaler",StandardScaler())

                ]
            )
            logger.info("Categorical and numerical pipeline created and encoding and scaling done")

            preprocessor=ColumnTransformer(
                transformers=[
                    ("num_pipeline",num_pipeline,numerical_features),
                    ("cat_pipeline",cat_pipeline,categorical_features)
                ]
            )

            return preprocessor
            

        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logger.info("Read the train and test data as dataframe")
            logger.info("Obtaining preprocessing object")

            preprocessor=self.get_data_transformer_object()
            target_column_name="math_score"
            numerical_features=["writing_score","reading_score"]

            input_feature_train_df=train_df.drop(target_column_name,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(target_column_name,axis=1)
            target_feature_test_df=test_df[target_column_name]
        except:
            pass




