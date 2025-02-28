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
from src.utlils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path= os.path.join(os.getcwd(),"artifact","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer_object(self):
        """
        This function is used for Data transformation
        """
        
        try:
            numerical_features=["writing score","reading score"]
            categorical_features=[
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course"
            ]

            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("std_scaler",StandardScaler(with_mean=False))
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("onehot",OneHotEncoder()),
                    ("std_scaler",StandardScaler(with_mean=False))

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

            preprocessor_obj=self.get_data_transformer_object()
            target_column_name="math score"
            numerical_features=["writing score","reading score"]

            input_feature_train_df=train_df.drop(target_column_name,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(target_column_name,axis=1)
            target_feature_test_df=test_df[target_column_name]

            logger.info("Applying the preprocessing object on the train and test data")

            input_feature_train_array=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_array=preprocessor_obj.transform(input_feature_test_df)
            logger.info("Tranformation of the data is completed")

            train_arr=np.c_[input_feature_train_array,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_array,np.array(target_feature_test_df)]

            logger.info("Saving the preprocessor object")

            save_object(self.data_transformation_config.preprocessor_obj_file_path,preprocessor_obj)

            return train_arr,test_arr
        except Exception as e:
            raise CustomException(e,sys)




