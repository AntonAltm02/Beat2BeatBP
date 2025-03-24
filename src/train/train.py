import sklearn
import xgboost as xgb
import os
from src.evaluation.eval import plot_hist
from src.data.statistics import ks_test_bp

def train_basic_pat(files_train, pat_type, feature_type, model_name):
    """
    This function trains the XGBoost models either with basic features alone, ePAT alone or even with ePAT and features
    in conjunction.
    - If the train_type is set to "pat" in main.py, the model is trained using the desired
    feature type as input and pat type as label or reference
    - If the train_type is set to "bp" in main.py, the model is trained using the desired feature set alone, the
    ePAT type alone or ePAT and feature set concatenated. This both for SBP and DBP estimation
    """
    print("Load Data")
    files_train, files_val = sklearn.model_selection.train_test_split(files_train, test_size=0.1, random_state=0)
    features_train, target_train, _, _ = get_data(files_train, pat_type=f"{pat_type}", feature_type=feature_type)
    features_val, target_val, _, _ = get_data(files_val, pat_type=f"{pat_type}", feature_type=feature_type)
    eval_set = [(features_train, target_train), (features_val, target_val)]
    print("Data loaded \n")

    print("Training the model")
    num_boost_round = 5000
    xgb_reg = xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="mae",
        n_estimators=num_boost_round,
        # early_stopping_rounds=20,
        learning_rate=0.01,
        device="cpu",
    )
    xgb_reg.fit(features_train, target_train, eval_set=eval_set, verbose=True)
    print("Training finished \n")

    print("Saving the trained model")
    script_dir = os.path.dirname(__file__)
    model_path = os.path.join(script_dir + "/../model/")
    xgb_reg.save_model(model_path + model_name)
    print("Model saved \n")

def train_basic_bp(features_train, features_val, target_train, target_val, bp_type, model_name):
    """
    This function trains the XGBoost models either with basic features alone, ePAT alone or even with ePAT and features
    in conjunction.
    - If the train_type is set to "pat" in main.py, the model is trained using the desired
    feature type as input and pat type as label or reference
    - If the train_type is set to "bp" in main.py, the model is trained using the desired feature set alone, the
    ePAT type alone or ePAT and feature set concatenated. This both for SBP and DBP estimation
    """

    eval_set = [(features_train, target_train), (features_val, target_val)]
    plot_hist(target_train, target_val)

    print("Training the model")
    num_boost_round = 5000
    xgb_reg = xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="mae",
        n_estimators=num_boost_round,
        # early_stopping_rounds=20,
        learning_rate=0.01,
        device="cpu",
    )
    xgb_reg.fit(features_train, target_train, eval_set=eval_set, verbose=True)

    print("Saving the trained model")
    script_dir = os.path.dirname(__file__)
    model_path = os.path.join(script_dir + "/../model/")
    xgb_reg.save_model(model_path + model_name)