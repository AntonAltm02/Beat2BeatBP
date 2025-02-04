from data.preprocessing import Processor
import data.statistics as stats
import os


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
    stats.plot_pat_bp(path_main=path_main, sub_files=files, pat_type=["ON"])
    stats.calculate_corr_bp_pat(path_main=path_main, sub_files=files, pat_type=["ON", "IT", "U", "SP", "DN", "DP"])
    stats.calculate_mean_std_pat(path_main=path_main, sub_files=files, pat_type=["ON", "IT", "U", "SP", "DN", "DP"])
    stats.plot_bp_pat_boxplot(path_main=path_main, sub_files=files, pat_type=["ON", "IT", "U", "SP", "DN", "DP"])
    print("Stop here")
