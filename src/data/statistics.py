import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')


def load_data(path_main, files, pat_type):
    for file in tqdm(files, desc=f"Loading data"):
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
    plt.show()
