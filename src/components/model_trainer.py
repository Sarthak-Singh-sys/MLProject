import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.utlils import evaluate_model

from src.logger import logger
from src.exception import CustomException
from src.utlils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_path=os.path.join("artifact","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logger.info("Splitting training and test")
            X_train,Y_train,X_test,Y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
                )
            logger.info("Training the model")
            models={
                "Random Forrest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Linear Regression":LinearRegression(),
                "KNN":KNeighborsRegressor(),
                "AdaBoost":AdaBoostRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "XGBoost":XGBRegressor(),
                "CatBoost":CatBoostRegressor()

            }
            model_report:dict=evaluate_model(X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test,models=models)
            logger.info("Model evaluation completed")

            best_score_model=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_score_model)
            ]
            best_model=models[best_model_name]
            if best_score_model<0.6:
                raise CustomException("No Best Model Found")
            logger.info("Best Model Found")

            save_object(file_path=self.model_trainer_config.trained_model_path,obj=best_model)

            predicted=best_model.predict(X_test)
            r2=r2_score(Y_test,predicted)
            return r2

        except Exception as e:
            raise CustomException(e,sys)
