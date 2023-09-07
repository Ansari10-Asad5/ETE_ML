import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_pkl_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr,test_arr):
        try:
            x_train,y_train,x_test,y_test=(
                train_arr[:,:-1],train_arr[:,-1],
                test_arr[:,:-1],test_arr[:,-1]
            )

            logging.info("Splitting into Train and Test is done")

            models={
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "KNN": KNeighborsRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False),
                "XGBoost": XGBRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
            }

            params = {
                "Linear Regression":{},
                "Decision Tree":{
                    'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2']
                },
                "Random Forest":{
                    # 'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features': ['sqrt', 'log2'],
                    'n_estimators':[8,16,32,64,128]
                },
                "KNN":{
                    'n_neighbors':[1,2,4,8,16]
                },
                "Gradient Boosting": {
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [.1, .01, 0.5, .001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }
            model_report:dict=evaluate_model(x_train=x_train,x_test=x_test,y_train=y_train,y_test=y_test,models=models,param=params)

            ## Get the best model score
            best_score=max(sorted(model_report.values()))

            ## Get the best model
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_score)
            ]

            best_model=models[best_model_name]

            if best_score<0.6:
                raise CustomException("No best model Found")
            logging.info("Best model found!")

            save_pkl_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model
            )

            predict=best_model.predict(x_test)

            r2score=r2_score(y_test,predict)
            return r2score

        except Exception as e:
            raise CustomException(e,sys)
