from data.preprocessing import Processor
import data.statistics as stats
import os
from train.train import train_basic_pat
from sklearn.model_selection import train_test_split
from src.evaluation.test import test_basic_pat
import src.evaluation.eval as eval


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
    files = os.listdir(os.path.join(data_path + "/../data/processed/ExtractedBP/SBP/"))
    show_stats = True
    if show_stats:
        # stats.plot_pat_bp(path_main=path_main, sub_files=files, pat_type=["ON"])
        # stats.calculate_corr_bp_pat(path_main=path_main, sub_files=files, pat_type=["ON", "IT", "U", "SP", "DN", "DP"])
        # stats.calculate_mean_std_pat(path_main=path_main, sub_files=files, pat_type=["ON", "IT", "U", "SP", "DN", "DP"])
        stats.plot_bp_pat_boxplot(path_main=path_main, sub_files=files, pat_type=["ON", "IT", "U", "SP", "DN", "DP"])
    print("Stop here")

    """
    Building XGBoost models to train and eval the prediction of PAT across 
    all types using the feature sets (AF, OF, RF, KF)

    train_files, test_files = train_test_split(files, test_size=0.2, shuffle=True, random_state=10)
    stats.t_test_pat(path_main=path_main, files_train=train_files, files_test=test_files, pat_type="ON")
    model_name = "PAT(ON)_AF"
    # train_basic_pat(files_train=train_files, pat_type="ON", feature_type="AF", model_name=model_name)
    pred, ref = test_basic_pat(files_test=test_files, pat_type="ON", feature_type="AF", model_name=model_name)
    eval.error_metrics(pred, ref)
    eval.corr_plot(pred, ref, train_type="pat", pat_type="ON", bp_type=None)
    print("Stop here")
    """
