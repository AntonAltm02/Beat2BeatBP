import numpy as np
import scipy
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import seaborn as sns
import os
from src.load import get_data, get_data_frame


def load_data(path_main, file, pat_type):
    df = pd.read_csv(path_main + "DataFrame/" + file + ".csv")
    no_zeros = (df[f"PAT({pat_type})"] != 0)
    sbp = df["FinapresSBP"][no_zeros]
    dbp = df["FinapresDBP"][no_zeros]
    pat = df[f"PAT({pat_type})"][no_zeros]
    return sbp, dbp, pat

def plot_pat_bp(path_main, sub_files):
    """
    Plot the extracted PAT of one selected subject
    :param path_main: processed, extracted PAT
    :param sub_files: selected subject file
    :return:
    """
    for file in tqdm(sub_files, desc="Plotting PATs and Spearman Correlation with BP"):
        pat_list = {
            "on": [],
            "it": [],
            "u": [],
            "sp": [],
            "dn": [],
            "dp": [],
        }
        for key, _ in pat_list.items():
            sbp, dbp, pat = load_data(path_main, file[:10], key.upper())
            pat_list[key].append(pat)
            # Spearman Correlation SBP-PAT
            pat_list[key].append(np.round(scipy.stats.spearmanr(pat, sbp)[0], 2))
            # Spearman Correlation DBP-PAT
            pat_list[key].append(np.round(scipy.stats.spearmanr(pat, dbp)[0], 2))

        fig, axes = plt.subplots(4, 2, figsize=(15, 9), sharex=True, sharey=False)
        axes = axes.flatten()
        for (key, _), (_, ax) in zip(pat_list.items(), enumerate(axes)):
            ax.plot(pat_list[key][0])
            ax.set(xlabel='Samples (a.u.)', ylabel=f'PAT({key.upper()}) (ms)')
            ax.set_title(f"DBP_PAT({key.upper()}): ρ = {pat_list[key][2]}, SBP_PAT({key.upper()}): ρ = {pat_list[key][1]}",
                         fontsize=12)
        axes[6].plot(sbp)
        axes[6].set(xlabel='Samples (a.u.)', ylabel=f'SBP (mmHg)')
        axes[6].set_title(f"SBP", fontsize=12)
        axes[7].plot(dbp)
        axes[7].set(xlabel='Samples (a.u.)', ylabel=f'DBP (mmHg)')
        axes[7].set_title(f"DBP", fontsize=12)

        plt.tight_layout(pad=3.0)  # Increase padding between plots
        fig.suptitle(f"PAT - {file[:10]}")
        plt.savefig(path_main + f"../../reports/figures/PAT_BP/" + f'{file[:10]}.pdf', format='pdf')
        # plt.show()
        plt.close()

def calculate_mean_std_pat(path_main, sub_files, pat_type):
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

def plot_bp_pat_boxplot(path_main, sub_files, pat_type):
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
    for file in tqdm(sub_files, desc="Calculating the Spearman Correlation per subject"):
        for i in pat_type:
            sbp, dbp, pat = load_data(path_main, file[:10], i)
            r_spearman_sbp[f"r_pat_{i.lower()}"].append(scipy.stats.spearmanr(pat, sbp)[0])
            r_spearman_dbp[f"r_pat_{i.lower()}"].append(scipy.stats.spearmanr(pat, dbp)[0])

    # df_dbp = pd.DataFrame([(k, v) for k, values in r_spearman_dbp.items() for v in values],
    #                   columns=['PAT Type', 'Spearman Correlation DBP'])
    df_dbp = []
    for i, values in r_spearman_dbp.items():
        df_dbp.append(values)
    df_dbp = pd.DataFrame(df_dbp).transpose()
    df_sbp = []
    for i, values in r_spearman_sbp.items():
        df_sbp.append(values)
    df_sbp = pd.DataFrame(df_sbp).transpose()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    sns.boxplot(data=df_dbp, ax=axes[0])
    axes[0].set_title("Spearman Correlation between PAT types and DBP")
    sns.boxplot(data=df_sbp, ax=axes[1])
    axes[1].set_title("Spearman Correlation between PAT types and SBP")
    custom_labels = ["ON", "IT", "U", "SP", "DN", "DP"]  # Replace with your desired names
    axes[0].set_xticks(ticks=range(len(custom_labels)), labels=custom_labels, rotation=45)
    axes[1].set_xticks(ticks=range(len(custom_labels)), labels=custom_labels, rotation=45)
    plt.tight_layout()
    plt.savefig(path_main + "../../reports/figures/R_BoxPlot/" + f'Spearman_Correlation.pdf', format='pdf')
    plt.show()

def calculate_corr_bp_pat(path_main, sub_files, pat_type):
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

def descriptive_stats(files_train, files_test, bp_type):
    df_train = get_data(files_train)
    df_test = get_data(files_test)

    bp_train = df_train[f"Finapres{bp_type.upper()}"]
    bp_test = df_test[f"Finapres{bp_type.upper()}"]

    print("Training Labels Summary:")
    print(pd.Series(bp_train).describe())

    print("\nTesting Labels Summary:")
    print(pd.Series(bp_test).describe())

def t_test_pat(path_main, files_train, files_test, pat_type):
    """
    T-Test for checking the statistical difference between PAT values of the train and test files after split
    """
    pat_files_train = [path_main + f"ExtractedPAT/{pat_type}/{file}" for file in files_train]
    pat_files_test = [path_main + f"ExtractedPAT/{pat_type}/{file}" for file in files_test]

    pat_train = np.array([np.mean(np.load(file, allow_pickle=True)) for file in pat_files_train])
    pat_test = np.array([np.mean(np.load(file, allow_pickle=True)) for file in pat_files_test])

    t_stat_pat, p_val_pat = scipy.stats.ttest_ind(pat_train, pat_test)
    print(f"PAT{pat_type}-t-statistic:", t_stat_pat)
    print(f"PAT{pat_type}-p-value:", p_val_pat)
    alpha = 0.05
    if p_val_pat > alpha:
        print("Fail to reject null hypothesis: No significant difference between train and test datasets. \n")
    else:
        print("Reject null hypothesis: Significant difference between train and test datasets. \n")

def t_test_bp(files_train, files_test, bp_type, random_state):
    """
    T-Test for checking the statistical difference between PAT values of the train and test files after split
    """
    df_train = get_data_frame(files_train)
    df_test = get_data_frame(files_test)

    bp_train = df_train[f"Finapres{bp_type.upper()}"]
    bp_test = df_test[f"Finapres{bp_type.upper()}"]

    t_stat_bp, p_val_bp = scipy.stats.ttest_ind(bp_train, bp_test)
    print(f"t-statistic:", t_stat_bp)
    print(f"p-value:", p_val_bp)
    alpha = 0.05
    if p_val_bp < alpha:
        print(f"{random_state} - Significant difference detected between train and test labels!")
    else:
        print(f"{random_state} - Train and test labels have a similar distribution.")
        print("Training Labels Summary:")
        print(pd.Series(bp_train).describe())
        print("\nTesting Labels Summary:")
        print(pd.Series(bp_test).describe())

        return random_state

def ks_test_bp(files_train, files_test, bp_type, random_state):
    """
    Kolmogorov-Smirnov (KS) Test checks if the train and test labels come from the same distribution.
    :return:
    """
    df_train = get_data_frame(files_train)
    df_test = get_data_frame(files_test)

    bp_train = df_train[f"Finapres{bp_type.upper()}"]
    bp_test = df_test[f"Finapres{bp_type.upper()}"]

    ks_stat, p_value = scipy.stats.ks_2samp(bp_train, bp_test)
    print(f"KS Statistic: {ks_stat}, p-value: {p_value}")

    if p_value < 0.05:
        print("Significant difference detected between train and test labels!")
    else:
        print("Train and test labels have a similar distribution.")

        print("Training Labels Summary:")
        print(pd.Series(bp_train).describe())
        print("\nTesting Labels Summary:")
        print(pd.Series(bp_test).describe())

        return random_state