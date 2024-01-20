import os
import sys
import pandas as pd
from src.Student_Analysis.exception import myexception
from src.Student_Analysis.logger import logging
from src.Student_Analysis.utils.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
            model_path=os.path.join("artifacts","model.pkl")
            
            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)
            
            scaled_data=preprocessor.transform(features)
            
            pred=model.predict(scaled_data)
            
            return pred
            
            
        
        except Exception as e:
            raise myexception(e,sys)
    
    
    
class CustomData:
    def __init__(self,
                 math:float,
                 reading:float,
                 writing:float,
                 gender:str,
                 race:str,
                 level_of_education:str,
                 lunch:str,
                 course:str):
        
        self.math=math
        self.reading=reading
        self.writing=writing
        self.gender=gender
        self.race=race
        self.level_of_education=level_of_education
        self.lunch=lunch
        self.course=course
            
                
    def get_data_as_dataframe(self):
            try:
                custom_data_input_dict = {
                    'math score':[self.math],
                    'reading score':[self.reading],
                    'writing score':[self.writing],
                    'gender':[self.gender],
                    'race/ethnicity':[self.race],
                    'parental level of education':[self.level_of_education],
                    'lunch':[self.lunch],
                    'test preparation course':[self.course]
                }
                df = pd.DataFrame(custom_data_input_dict)
                logging.info('Dataframe Gathered')
                return df
            except Exception as e:
                logging.info('Exception Occured in prediction pipeline')
                raise myexception(e,sys)