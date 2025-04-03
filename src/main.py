from data.preprocessing import Processor
import data.statistics as stats
import os
from train.train import train_basic_pat, train_basic_bp
from sklearn.model_selection import train_test_split
from src.evaluation.test import test_basic_pat, test_basic_bp
import src.evaluation.eval as eval
from src.load import process


script_dir = os.path.dirname(__file__)
data_path = os.path.join(script_dir)

if __name__ == "__main__":
    """
    Processor to extract the feature subsets and systolic as well as diastolic 
    blood pressure (BP) from each subject. Additionally, the PPG signals have been
    segmented by an algorithm of pyPPG to extract the fiducial points of each PPG beat. 
    The matching beat of fiducial points included in the raw features has been used to
    extract the various pulse arrival time (PAT) types.   
    """
    pre_processor = Processor(data_path=data_path, replace=True, config_filter=None)
    pre_processor.process()

    """
    Statistics to analyze the extracted PAT, BP or features. The file statistics.py holds
    functions to plot the PAT for a selected subject, calculate and plot the correlation 
    between PAT and BP, etc.
    """
    path_main = data_path + "/../data/processed/"
    files = os.listdir(os.path.join(data_path + "/../data/processed/DataFrame/"))
    show_stats = False
    if show_stats:
        # stats.plot_pat_bp(path_main=path_main, sub_files=files)
        stats.calculate_corr_bp_pat(path_main=path_main, sub_files=files, pat_type=["ON", "IT", "U", "SP", "DN", "DP"])
        stats.calculate_mean_std_pat(path_main=path_main, sub_files=files, pat_type=["ON", "IT", "U", "SP", "DN", "DP"])
        stats.plot_bp_pat_boxplot(path_main=path_main, sub_files=files, pat_type=["ON", "IT", "U", "SP", "DN", "DP"])

    """
    Building XGBoost models to train and eval the prediction of PAT across 
    all types using the feature sets (AF, OF, RF, KF)
    """
    eval_pat = False
    if eval_pat:
        pat_type = "DN"
        feature_type = "AF"

        # Conducting statistical tests to have balanced splits of train and test data
        fixed_state = 5
        conduct_test = False
        if conduct_test:
            for state in range(100):
                train_files, test_files = train_test_split(files, test_size=0.3, random_state=state)
                # !!!! Here it needs to be refined
                fixed_state = stats.t_test_pat(files_train=train_files, files_test=test_files, pat_type=None, random_state=state)
        train_files, test_files = train_test_split(files, test_size=0.3, random_state=fixed_state)
        train_files, val_files = train_test_split(train_files, test_size=0.3, random_state=fixed_state)
        features_train, features_val, features_test, target_train, target_val, target_test = process(
            files_train=train_files, files_val=val_files, files_test=test_files)

        model_name = f"PAT({pat_type})_{feature_type}"
        train_basic_pat(files_train=train_files, pat_type=pat_type, feature_type=feature_type, model_name=model_name)
        predictions, ref = test_basic_pat(files_test=test_files, pat_type=pat_type, feature_type=feature_type, model_name=model_name)
        eval.error_metrics(predictions, ref)
        eval.corr_plot(predictions, ref, train_type="pat", pat_type=pat_type, bp_type=None)

    """
    Building XGBoost models to train and eval the prediction of BP across 
    all types using the feature sets (AF, OF, RF, KF) and the PATs (ON, IT, MD, SP, DN, DP)
    """
    eval_bp = True
    if eval_bp:
        train_type = "BP"
        bp_type = "SBP"
        feature_type = "ALL"

        # Conducting statistical tests to have balanced splits of train and test data
        fixed_state = 42
        conduct_test = False
        if conduct_test:
            print("Conducting a T-Test for statistical difference measure")
            for state in range(100):
                train_files, test_files = train_test_split(files, test_size=0.3, random_state=state)
                fixed_state = stats.t_test_bp(files_train=train_files, files_test=test_files, bp_type=bp_type, random_state=state)
                if fixed_state is not None:
                    break

        print("Splitting the subjects into train, val and test data")
        train_files, temp_files = train_test_split(files, test_size=0.3, random_state=fixed_state)
        test_files, val_files = train_test_split(temp_files, test_size=0.1, random_state=fixed_state)

        stats.descriptive_stats(files_train=train_files, files_test=test_files, bp_type=bp_type)

        print("Extracting the feature and target data for each subject splits")
        features_train, features_val, features_test, target_train, target_val, target_test = process(
            files_train=train_files, files_val=temp_files, files_test=temp_files, bp_type=bp_type)

        model_name = f"{bp_type}_{feature_type}"
        print("Training model")
        train_basic_bp(features_train=features_train, features_val=features_val, target_train=target_train,
                       target_val=target_val, model_name=model_name)
        print("Inference model")
        predictions, ref = test_basic_bp(features_test=features_test, target_test=target_test, bp_type=bp_type, model_name=model_name)
        eval.error_metrics(predictions, ref)
        eval.corr_plot(predictions, ref, train_type=train_type, pat_type=None, bp_type=bp_type)
        eval.plot_inference(pred=predictions, ref=ref, train_type=train_type, pat_type=None, bp_type=bp_type)
