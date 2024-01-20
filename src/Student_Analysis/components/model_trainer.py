
import pandas as pd
import numpy as np
import os
import sys
from src.Student_Analysis.logger import logging
from src.Student_Analysis.exception import myexception
from dataclasses import dataclass
from src.Student_Analysis.utils.utils import save_object
from src.Student_Analysis.utils.utils import evaluate_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import accuracy_score,r2_score,mean_absolute_error,mean_squared_error


@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')
    
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
            'LinearRegression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=1.0),
    'SVR': SVR(C=1.0, kernel='rbf', epsilon=0.1, gamma='scale'),
    'DecisionTreeRegressor': DecisionTreeRegressor(max_depth=10, min_samples_split=2, min_samples_leaf=1, max_features=None),
    'RandomForestRegressor': RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', bootstrap=True),
    'GradientBoostingRegressor': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2, min_samples_leaf=1, max_features=None, subsample=0.8),
    'KNeighborsRegressor': KNeighborsRegressor(n_neighbors=5, weights='uniform', p=2, algorithm='auto'),
    'XGBRegressor': XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, subsample=0.8, colsample_bytree=0.8, gamma=0, reg_alpha=0, reg_lambda=1, objective='reg:squarederror'),
    'LGBMRegressor': LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, subsample=0.8, colsample_bytree=0.8, min_child_samples=20, reg_alpha=0, reg_lambda=1, objective='regression'),
    'CatBoostRegressor': CatBoostRegressor(n_estimators=100, learning_rate=0.1, depth=6, subsample=0.8, colsample_bylevel=0.8, min_child_samples=20, reg_lambda=1)
            
        }
            
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )
          

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise myexception(e,sys)

        
    