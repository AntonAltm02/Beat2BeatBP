from tqdm import tqdm
import numpy as np
import os

script_dir = os.path.dirname(__file__)
data_path = os.path.join(script_dir + "/../data/processed/")


def get_data(files, pat_type, feature_type):
    features_list = []
    pat_list = []
    sbp_list = []
    dbp_list = []
    for file in tqdm(files):
        features_list.append(np.load(data_path + f"ExtractedFeatures/{feature_type}/{file[:10]}.npy", allow_pickle=True))
        pat_list.append(np.expand_dims(np.load(data_path + f"ExtractedPAT/{pat_type}/{file[:10]}.npy", allow_pickle=True),-1))
        sbp_list.append(np.expand_dims(np.load(data_path + f"ExtractedBP/SBP/{file[:10]}.npy", allow_pickle=True),-1))
        dbp_list.append(np.expand_dims(np.load(data_path + f"ExtractedBP/DBP/{file[:10]}.npy", allow_pickle=True),-1))

    features = np.concatenate(features_list, axis=0)
    pat = np.concatenate(pat_list, axis=0)
    sbp = np.concatenate(sbp_list, axis=0)
    dbp = np.concatenate(dbp_list, axis=0)

    non_zero_rows = ~np.all(pat == 0, axis=1)
    features = features[non_zero_rows]
    pat = pat[non_zero_rows]
    sbp = sbp[non_zero_rows]
    dbp = dbp[non_zero_rows]

    return features, pat, sbp, dbp