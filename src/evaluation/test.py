import numpy as np
import xgboost as xgb
from src.load import get_data, get_data_frame
import os

script_dir = os.path.dirname(__file__)
model_path = os.path.join(script_dir + "/../model/")

def test_basic_pat(files_test, pat_type, feature_type, model_name):
    print("Load Model")
    loaded_model = xgb.Booster()
    loaded_model.load_model(model_path + model_name)
    num_boosting_rounds = loaded_model.num_boosted_rounds()
    print(f"Number of boosting rounds: {num_boosting_rounds}")
    print("Model loaded \n")

    print("Load Data")
    features_test, pat_test, _, _ = get_data(files_test, pat_type=f"{pat_type}", feature_type=feature_type)
    print("Data loaded \n")

    print("Predicting the target variable")
    dtest = xgb.DMatrix(features_test)
    predictions = loaded_model.predict(data=dtest)
    print("Prediction finished")
    return predictions, pat_test

def test_basic_bp(features_test, target_test, bp_type, model_name):
    print("Load Model")
    loaded_model = xgb.Booster()
    loaded_model.load_model(model_path + model_name)
    num_boosting_rounds = loaded_model.num_boosted_rounds()
    print(f"Number of boosting rounds: {num_boosting_rounds}")
    print("Model loaded \n")

    print("Predicting the target variable")
    dtest = xgb.DMatrix(features_test)
    predictions = loaded_model.predict(data=dtest)
    print("Prediction finished")
    return predictions, np.array(target_test)