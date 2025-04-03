from tqdm import tqdm
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

script_dir = os.path.dirname(__file__)
data_path = os.path.join(script_dir + "/../data/processed/")


def conduct_PCA(data):
    data.fillna(data.mean(), inplace=True)
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    pca = PCA(n_components=10)
    principal_components = pca.fit_transform(data_scaled)

    pca_df = pd.DataFrame(principal_components, columns=[f'PC_{i + 1}' for i in range(10)])
    explained_variance = pca.explained_variance_ratio_
    print("Explained Variance Ratio:", explained_variance)
    print("Cumulative Explained Variance:", np.cumsum(explained_variance))

    return pca_df

def moving_median_filter(signal):
    window_size = 10
    i = 0
    moving_median = []
    while i < len(signal) - window_size + 1:
        window = signal[i: i + window_size]
        window_median = np.median(window)
        moving_median.append(window_median)
        i += 1
    return np.array(moving_median)

def get_data_frame(files):
    result_df = pd.DataFrame()
    for file in files:
        df = pd.read_csv(data_path + f"DataFrame/{file[:10]}.csv")
        df = df.drop(columns=["Unnamed: 0"])
        result_df = pd.concat([result_df, df], ignore_index=True)
    return result_df

def process(files_train, files_val, files_test, bp_type):
    df_train = get_data_frame(files_train)
    df_val = get_data_frame(files_val)
    df_test = get_data_frame(files_test)

    drop_columns = ["FinapresSBP", "FinapresDBP", "FinapresPP", "PPG", "SBP", "DBP", "PP", "MBP"]
    features_train = df_train.drop(columns=drop_columns)
    features_val = df_val.drop(columns=drop_columns)
    features_test = df_test.drop(columns=drop_columns)

    target_train = df_train[f"Finapres{bp_type.upper()}"]
    target_val = df_val[f"Finapres{bp_type.upper()}"]
    target_test = df_test[f"Finapres{bp_type.upper()}"]

    return features_train, features_val, features_test, target_train, target_val, target_test