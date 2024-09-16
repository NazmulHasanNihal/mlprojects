import os 
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


from src.exception import CustomException 


def save_object(file_path, obj):
    """This function saves a Python object to a given file path.

    Args:
        file_path (str): The path to save the object to.
        obj (object): The Python object to be saved.

    Raises:
        CustomException: If there is an issue with saving to the given file path.
    """

    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, X_test, y_train, y_test, models, param):
    """
    Evaluate a list of models on given data and parameters.

    Args:
        X_train (pandas.DataFrame): The feature data for training.
        X_test (pandas.DataFrame): The feature data for testing.
        y_train (pandas.Series): The target data for training.
        y_test (pandas.Series): The target data for testing.
        models (dict): A dictionary of models to evaluate. The keys are the model names and the values are the model objects.
        param (dict): A dictionary of parameters for GridSearchCV. The keys are the model names and the values are dictionaries of parameters.

    Returns:
        dict: A dictionary with the model names as keys and the R2 scores as values.

    Raises:
        CustomException: If there is an issue with evaluating the models.
    """
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            # model.fit(X_train, y_train) # Train Model

            y_train_pred = model.predict(X_train)
            
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path):
        """
        Loads a pickled object from a file.

        Args:
            file_path (str): The path to the file to load.

        Returns:
            object: The loaded object.

        Raises:
            CustomException: If there is an issue with loading the object.
        """
        try:
            with open(file_path, "rb") as file_obj:
                return dill.load(file_obj)

        except Exception as e:
            raise CustomException(e, sys)