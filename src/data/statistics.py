import numpy as np
import scipy
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import seaborn as sns


def load_data(path_main, file, pat_type):
    sbp = np.load(path_main + "ExtractedBP/SBP/" + file + ".npy")
    dbp = np.load(path_main + "ExtractedBP/DBP/" + file + ".npy")
    pat = np.load(path_main + f"ExtractedPAT/{pat_type}/" + file + ".npy", allow_pickle=True)
    no_zeros = (pat != 0)
    sbp = sbp[no_zeros]
    dbp = dbp[no_zeros]
    pat = pat[no_zeros]
    return sbp, dbp, pat

def plot_PAT_BP(path_main, sub_files, pat_type):
    """
    Plot the extracted PAT of one selected subject
    :param path_main: processed, extracted PAT
    :param sub_files: selected subject file
    :param pat_type: PAT type (here: "ON", "DP", "DN", "SP", ...)
    :return:
    """
    for pat_type in pat_type:
        for file in sub_files:
            sbp, dbp, pat = load_data(path_main, file[:10], pat_type)

            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=False)
            ax1.plot(pat)
            ax1.set(xlabel='Samples (a.u.)', ylabel='PAT (ms)')
            ax2.plot(sbp)
            ax2.set(xlabel='Samples (a.u.)', ylabel='SBP (mmHg)')
            ax3.plot(dbp)
            ax3.set(xlabel='Samples (a.u.)', ylabel='DBP (mmHg)')
            fig.suptitle(f"PAT, SBP and DBP - subject {file[:10]}")
            plt.show()

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

        print(f"Subject-wise metrics of PAT({pat_type}) - Mean: {np.round(np.mean(mean), 2)} and SD: {np.round(np.mean(np.std(std)), 2)}")

def plot_BP_PAT_BoxPlot(path_main, sub_files, pat_type):
    """
    Plot the Pearson correlation between the PAT and BP as Boxplot across all PAT and BP types
    """
    r_spearman_sbp = {
        "r_pat_on": [],
        "r_pat_it": [],
        "r_pat_u": [],
        "r_pat_sp": [],
        "r_pat_dn": [],
        "r_pat_dp": [],
    }
    r_spearman_dbp = {
        "r_pat_on": [],
        "r_pat_it": [],
        "r_pat_u": [],
        "r_pat_sp": [],
        "r_pat_dn": [],
        "r_pat_dp": [],
    }
    for file in sub_files:
        for i in pat_type:
            sbp, dbp, pat = load_data(path_main, file[:10], i)
            r_spearman_sbp[f"r_pat_{i.lower()}"].append(scipy.stats.spearmanr(pat, sbp)[0])
            r_spearman_dbp[f"r_pat_{i.lower()}"].append(scipy.stats.spearmanr(pat, dbp)[0])

    df_dbp = pd.DataFrame([(k, v) for k, values in r_spearman_dbp.items() for v in values],
                      columns=['PAT Type', 'Spearman Correlation DBP'])
    df_sbp = pd.DataFrame([(k, v) for k, values in r_spearman_sbp.items() for v in values],
                          columns=['PAT Type', 'Spearman Correlation SBP'])

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    sns.boxplot(x="PAT Type", y="Spearman Correlation DBP", data=df_dbp, ax=axes[0])
    axes[0].set_title("Spearman Correlation between PAT types and DBP")
    sns.boxplot(x="PAT Type", y="Spearman Correlation SBP", data=df_sbp, ax=axes[1])
    axes[1].set_title("Spearman Correlation between PAT types and SBP")
    custom_labels = ["ON", "IT", "U", "SP", "DN", "DP"]  # Replace with your desired names
    axes[0].set_xticks(ticks=range(len(custom_labels)), labels=custom_labels, rotation=45)
    axes[1].set_xticks(ticks=range(len(custom_labels)), labels=custom_labels, rotation=45)
    plt.tight_layout()
    plt.show()

def calculate_corr_BP_PAT(path_main, sub_files, pat_type):
    for pat_type in pat_type:
        results = {
            "pearson_sbp": [],
            "pearson_dbp": [],
            "spearman_sbp": [],
            "spearman_dbp": []
        }
        corr_strength_dbp = {
            "very strong": [],
            "strong": [],
            "moderate": [],
            "weak": [],
            "very weak": []
        }
        corr_strength_sbp = {
            "very strong": [],
            "strong": [],
            "moderate": [],
            "weak": [],
            "very weak": []
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

            if np.round(np.abs(correlations["spearman_sbp"]), 2) >= 0.8:
                corr_strength_sbp["very strong"].append(file[:10])
            elif 0.6 <= np.round(np.abs(correlations["spearman_sbp"]), 2) < 0.79:
                corr_strength_sbp["strong"].append(file[:10])
            elif 0.4 <= np.round(np.abs(correlations["spearman_sbp"]), 2) < 0.59:
                corr_strength_sbp["moderate"].append(file[:10])
            elif 0.2 <= np.round(np.abs(correlations["spearman_sbp"]), 2) < 0.39:
                corr_strength_sbp["weak"].append(file[:10])
            elif 0.0 <= np.round(np.abs(correlations["spearman_sbp"]), 2) < 0.19:
                corr_strength_sbp["very weak"].append(file[:10])

            if np.round(np.abs(correlations["spearman_dbp"]), 2) >= 0.8:
                corr_strength_dbp["very strong"].append(file[:10])
            elif 0.6 <= np.round(np.abs(correlations["spearman_dbp"]), 2) < 0.79:
                corr_strength_dbp["strong"].append(file[:10])
            elif 0.4 <= np.round(np.abs(correlations["spearman_dbp"]), 2) < 0.59:
                corr_strength_dbp["moderate"].append(file[:10])
            elif 0.2 <= np.round(np.abs(correlations["spearman_dbp"]), 2) < 0.39:
                corr_strength_dbp["weak"].append(file[:10])
            elif 0.0 <= np.round(np.abs(correlations["spearman_dbp"]), 2) < 0.19:
                corr_strength_dbp["very weak"].append(file[:10])

        r_pearson_sbp = np.vstack(results["pearson_sbp"])
        r_pearson_dbp = np.vstack(results["pearson_dbp"])
        r_spearman_sbp = np.vstack(results["spearman_sbp"])
        r_spearman_dbp = np.vstack(results["spearman_dbp"])

        print(f"Pearson R - Overall mean of subject-wise R between PAT({pat_type}) and SBP - Mean: {np.round(np.mean(r_pearson_sbp), 2)} and SD: {np.round(np.std(r_pearson_sbp), 2)}")
        print(f"Pearson R - Overall mean of subject-wise R between PAT({pat_type}) and DBP - Mean: {np.round(np.mean(r_pearson_dbp), 2)} and SD: {np.round(np.std(r_pearson_dbp), 2)}")
        print(f"Spearman R - Overall mean of subject-wise R between PAT({pat_type}) and SBP - Mean: {np.round(np.mean(r_spearman_sbp), 2)} and SD: {np.round(np.std(r_spearman_sbp), 2)}")
        print(f"Spearman R - Overall mean of subject-wise R between PAT({pat_type}) and DBP - Mean: {np.round(np.mean(r_spearman_dbp), 2)} and SD: {np.round(np.std(r_spearman_dbp), 2)} \n")
