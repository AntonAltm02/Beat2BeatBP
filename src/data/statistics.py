import numpy as np
import scipy
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


def load_data(path_main, file, pat_type):
    sbp = np.load(path_main + "ExtractedBP/SBP/" + file + ".npy")
    dbp = np.load(path_main + "ExtractedBP/DBP/" + file + ".npy")
    pat = np.load(path_main + f"ExtractedPAT/{pat_type}/" + file + ".npy", allow_pickle=True)
    no_zeros = (pat != 0) | (pat != np.nan)
    sbp = sbp[no_zeros]
    dbp = dbp[no_zeros]
    pat = pat[no_zeros]
    return sbp, dbp, pat

def plot_PAT(path_main, sub_file, pat_type):
    """
    Plot the extracted PAT of one selected subject
    :param path_main: processed, extracted PAT
    :param sub_file: selected subject file
    :param pat_type: PAT type (here: "ON", "DP", "DN", "SP", ...)
    :return:
    """
    sbp, dbp, pat = load_data(path_main, sub_file, pat_type)
    plt.figure()
    plt.plot(pat)
    plt.xlabel("Samples")
    plt.ylabel("PAT in ms")
    plt.title(f"PAT of subject {sub_file}")
    # plt.show()

def calculate_mean_std_PAT(path_main, sub_files, pat_type):
    """

    :param path_main:
    :param sub_files:
    :param pat_type:
    :return:
    """
    results = {
        "mean": [],
        "std": [],
    }
    for pat_type in pat_type:
        for file in sub_files:
            sbp, dbp, pat = load_data(path_main, file[:10], pat_type)
            metrics = {
                "mean": np.mean(pat),
                "std": np.std(pat),
            }
            for key, value in metrics.items():
                results[key].append(value)

        mean = np.vstack(results["mean"])
        std = np.vstack(results["std"])

        print(f"Overall mean of subject-wise metrics of PAT({pat_type}) - Mean: {np.round(np.mean(mean), 2)} and SD: {np.round(np.mean(np.std(std)), 2)}")

def calculate_corr_BP_PAT(path_main, sub_files, pat_type):
    results = {
        "pearson_sbp": [],
        "pearson_dbp": [],
        "spearman_sbp": [],
        "spearman_dbp": []
    }
    for file in sub_files:
        sbp, dbp, pat = load_data(path_main, file[:10], pat_type)
        correlations = {
            "pearson_sbp": scipy.stats.pearsonr(pat, sbp)[0],
            "pearson_dbp": scipy.stats.pearsonr(pat, dbp)[0],
            "spearman_sbp": scipy.stats.spearmanr(pat, sbp)[0],
            "spearman_dbp": scipy.stats.spearmanr(pat, dbp)[0]
        }
        for key, value in correlations.items():
            results[key].append(value)

    r_pearson_sbp = np.vstack(results["pearson_sbp"])
    r_pearson_dbp = np.vstack(results["pearson_dbp"])
    r_spearman_sbp = np.vstack(results["spearman_sbp"])
    r_spearman_dbp = np.vstack(results["spearman_dbp"])

    print(f"Pearson R - Overall mean of subject-wise R between PAT({pat_type}) and SBP - Mean: {np.round(np.mean(r_pearson_sbp), 2)} and SD: {np.round(np.std(r_pearson_sbp), 2)}")
    print(f"Pearson R - Overall mean of subject-wise R between PAT({pat_type}) and DBP - Mean: {np.round(np.mean(r_pearson_dbp), 2)} and SD: {np.round(np.std(r_pearson_dbp), 2)}")
    print(f"Spearman R - Overall mean of subject-wise R between PAT({pat_type}) and SBP - Mean: {np.round(np.mean(r_spearman_sbp), 2)} and SD: {np.round(np.std(r_spearman_sbp), 2)}")
    print(f"Spearman R - Overall mean of subject-wise R between PAT({pat_type}) and DBP - Mean: {np.round(np.mean(r_spearman_dbp), 2)} and SD: {np.round(np.std(r_spearman_dbp), 2)} \n")

