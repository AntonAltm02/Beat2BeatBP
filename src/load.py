from tqdm import tqdm
import numpy as np
import os
import pandas as pd

script_dir = os.path.dirname(__file__)
data_path = os.path.join(script_dir + "/../data/processed/")


def get_data(files, pat_type=None, feature_type=None):
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

def get_data_frame(files):
    result_df = pd.DataFrame()
    for file in files:
        df = pd.read_csv(data_path + f"DataFrame/{file[:10]}.csv")
        df = df.drop(columns=["Unnamed: 0"])
        result_df = pd.concat([result_df, df], ignore_index=True)
    return result_df

def process(files_train, files_val, files_test):
    df_train = get_data_frame(files_train)
    df_val = get_data_frame(files_val)
    df_test = get_data_frame(files_test)
    drop_columns = ["FinapresSBP", "FinapresDBP", "FinapresPP", "PPG", "SBP", "DBP", "PP", "MBP",
                    "PAT(ON)", "PAT(U)", "PAT(IT)", "PAT(SP)", "PAT(DN)", "PAT(DP)"]
    features_train = df_train.drop(columns=drop_columns)
    features_val = df_val.drop(columns=drop_columns)
    features_test = df_test.drop(columns=drop_columns)
    target_train = df_train[f"Finapres{bp_type.upper()}"]
    target_val = df_val[f"Finapres{bp_type.upper()}"]
    target_test = df_test[f"Finapres{bp_type.upper()}"]
    return features_train, features_val, features_test, target_train, target_val, target_test