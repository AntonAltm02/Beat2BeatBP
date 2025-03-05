import os
import scipy
import matplotlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotmap import DotMap
import pyPPG.preproc as PP
from biosppy import signals
import pyPPG.fiducials as FP
from pyPPG import PPG, Fiducials
import pyPPG.ppg_sqi as SQI
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

def check_fiducial_plot(ecg, ppg, vpg, ref_pts, rPeak, key, sub_id):
    plt.figure()
    plt.plot(min_max_norm(ppg))
    plt.plot(min_max_norm(vpg))
    plt.plot(np.array(ref_pts).astype(int)[0:4],
             min_max_norm(ppg)[np.array(ref_pts).astype(int)[0:4]], "bo")
    plt.plot(np.array(ref_pts).astype(int)[4],
             min_max_norm(vpg)[np.array(ref_pts).astype(int)[4]], "go")
    plt.plot(min_max_norm(ecg))
    plt.plot(rPeak, min_max_norm(ecg)[rPeak], "ro")
    plt.xlim(rPeak - 100, np.array(ref_pts)[3].astype(int) + 100)
    plt.title(f"{sub_id} - PAT({key})")
    plt.show()


def pan_tompkins_algorithm(fs, ecg_signal):
    """
    Finding R-peaks in an ECG-signal using the Pan-Tompkins algorithm
    """
    diff_signal = np.diff(ecg_signal)
    squared_signal = diff_signal ** 2
    window_size = int(0.15 * fs)
    delay = (window_size - 1) // 2
    moving_avg_signal = np.convolve(squared_signal, np.ones(window_size) / window_size, mode="same")
    moving_avg_signal = np.concatenate((moving_avg_signal[delay:], np.zeros(delay)))
    locs, pks = scipy.signal.find_peaks(moving_avg_signal, height=0.6 * np.max(moving_avg_signal), distance=int(0.2 * fs))
    return locs

def get_r_peaks(fs, ecg_signal):
    try:
        r_idx = signals.ecg.ecg(ecg_signal, sampling_rate=fs, show=False)["rpeaks"]
    except:
        r_idx = pan_tompkins_algorithm(fs, ecg_signal)
    return r_idx

def min_max_norm(signal):
    out = (signal - np.min(signal))/(np.max(signal) - np.min(signal))
    return out

class Processor:
    def __init__(self, data_path, replace, config_filter):
        self.data_path = data_path + "/../data/raw/"
        self.target_path = data_path + "/../data/processed/"
        self.config_filter=config_filter
        self.fs = 100

        self.id = None
        self.subject_data = None
        self.subject_indices = None
        self.filtered_indices = None

        self.fiducials = None
        self.detection_points = None
        self.ppg_segment = None
        self.vpg_segment = None
        self.ecg_segment = None

        self.features_all = None
        self.features_reconstructed = None
        self.features_original = None
        self.features_kernel = None

        self.dbp = None
        self.sbp = None

        self.ids = None
        self.replace = replace
        self.keys = None
        self.data_error = False
        self._no_error = True

    def error_handling(self, sub_id):
        """

        :param sub_id:
        :return:
        """
        if self._no_error:
            with open(self.target_path + '_error.txt', 'w') as file:
                file.write('Following ID(s) have caused an error:\n')
                file.write(sub_id[:-4] + "\n")
        else:
            with open(self.target_path + '_error.txt', 'a') as file:
                file.write(sub_id[:-4] + "\n")
        print(sub_id, " creates an error!")
        self.data_error = True

    def load_mat_data(self):
        """

        :return:
        """
        try:
            data_dict = scipy.io.loadmat(self.data_path + "RawSignals/" + self.id[:10] + ".mat")
            self.ecg_segment = np.float32(np.swapaxes(data_dict["ecg"], 0, 1)).squeeze(-1)
            self.ppg_segment = np.float32(np.swapaxes(data_dict["fingerPPG"], 0, 1)).squeeze(-1)
        except:
            self.error_handling(self.id[:10])

    """
    Selection and cleaning of the raw features
    """
    def selection_and_cleaning(self):
        """

        :return:
        """
        self.ids = os.listdir(self.data_path + "RawFeatures/")
        if not self.replace:
            id_ready = os.listdir(self.target_path + "SelectedData/")
            self.ids = [x for x in self.ids if x[:10] + '.csv' not in id_ready]
        for self.id in tqdm(self.ids, desc="Selecting and cleaning the selected subject data"):
            # loading the raw data frame
            raw_data = pd.read_csv(self.data_path + "RawFeatures/" + self.id)
            # extracting the
            self.id = raw_data["ID"][0][:10]

            # loading the PPG beat time-series data
            ppg_beat = raw_data["PPG"].str.strip('[]').str.split(';').values
            ppg_beat = np.zeros((len(ppg_beat), 300))
            for i in range(len(ppg_beat)):
                ppg_beat[i] = np.array(ppg_beat[i])

            # removing data rows with PPG beats containing any NaN values
            delete_rows_nan = np.isnan(ppg_beat).any(axis=1)
            raw_data = raw_data[~delete_rows_nan]
            # removing data rows with SBP values higher than 250 and lower than 50 mmHg
            delete_rows_sbp = (raw_data["FinapresSBP"] > 250) | (raw_data["FinapresSBP"] < 50)
            raw_data = raw_data[~delete_rows_sbp]
            raw_data = raw_data.dropna(subset=["FinapresSBP"])
            # removing data rows with DBP values higher than 160 and lower than 30 mmHg
            delete_rows_dbp = (raw_data["FinapresDBP"] > 160) | (raw_data["FinapresDBP"] < 30)
            raw_data = raw_data[~delete_rows_dbp]
            raw_data = raw_data.dropna(subset=["FinapresDBP"])

            if not self.data_error:
                if len(raw_data) > 0:
                    raw_data.to_csv(self.target_path + "SelectedData/" + self.id + ".csv",  index=False)
                else:
                    self.error_handling(self.id)
                    self.data_error = False

    """
    Extraction of the fiducial points per beat of the PPG signal using the pyPPG library
    """
    def extract_fiducial_points(self):
        """

        :return:
        """
        signal = DotMap()
        signal.start = 0 # start sample of the signal
        signal.end = -1 # last sample of the signal
        signal.fs = self.fs
        signal.v = self.ppg_segment
        signal.filtering = True  # whether to filter the PPG signal
        signal.fL = 0.5  # Lower cutoff frequency (Hz)
        signal.fH = 12  # Upper cutoff frequency (Hz)
        signal.order = 4  # Filter order
        signal.sm_wins = {'ppg': 10, 'vpg': 10, 'apg': 10, 'jpg': 10}  # smoothing windows in millisecond for the PPG, PPG', PPG'' and PPG'''

        prep = PP.Preprocessing(signal, filtering=signal.filtering)
        signal.filt_ppg = prep[0]
        signal.filt_sig = prep[0]
        signal.filt_vpg = prep[1]
        signal.filt_d1 = prep[1]
        signal.filt_apg = prep[2]
        signal.filt_d2 = prep[2]
        signal.filt_jpg = prep[3]
        signal.filt_d3 = prep[3]

        np.save(self.data_path + "../processed/FilteredPPG/PPG/" + self.id[:10], signal.filt_sig)
        np.save(self.data_path + "../processed/FilteredPPG/VPG/" + self.id[:10], signal.filt_vpg)

        """
        Extracting the fiducial points
        """

        s = PPG(signal)
        # initializing the fiducials package of pyPPG
        fpex = FP.FpCollection(s=s)
        # extracting the fiducials
        fiducials = fpex.get_fiducials(s=s)

        fp = Fiducials(fp=fiducials)
        ppgSQI = round(np.mean(SQI.get_ppgSQI(ppg=signal.filt_ppg, fs=signal.fs, annotation=fp.sp)) * 100, 2)
        print(f"Mean PPG SQI - {self.id[:10]}: ", ppgSQI, '%')

        fiducials.to_csv(self.target_path + f"FiducialPoints/" + self.id, index=False)

    def fiducial_points_plotter(self):
        """

        :return:
        """
        self.ids = os.listdir(self.target_path + "FiducialPoints/")
        if not self.replace:
            id_ready = os.listdir(self.target_path + "../../reports/figures/pyPPG Fiducials/")
            self.ids = [x for x in self.ids if x[:10] + '.png' not in id_ready]
        for self.id in tqdm(self.ids, desc="Creating the plots of Fiducials of each subject"):
            fiducials = pd.read_csv(self.target_path + "FiducialPoints/" + self.id)
            self.load_mat_data()
            start_idx = 1
            end_idx = 100
            fiducials = fiducials.iloc[start_idx:end_idx]
            self.ppg_segment = self.ppg_segment[:int(fiducials["dp"].iloc[-1])+1]

            fig = plt.figure(figsize=(15, 8))
            ax1 = plt.subplot(211)
            ax1.set(xlabel='Samples (a.u.)', ylabel='Filtered PPG')
            plt.plot(self.ppg_segment, label=None)
            plt.plot(fiducials["on"], self.ppg_segment[fiducials["on"]], "o")
            ax2 = plt.subplot(212, sharex=ax1)
            ax2.set(xlabel='Samples (a.u.)', ylabel='Filtered VPG')
            plt.plot(self.ppg_segment, label=None)
            fig.subplots_adjust(hspace=0, wspace=0)
            plt.show()

    def fiducial_points_extraction(self):
        """

        :return:
        """
        self.ids = os.listdir(self.target_path + "SelectedData/")
        if not self.replace:
            id_ready = os.listdir(self.target_path + "FiducialPoints/")
            self.ids = [x for x in self.ids if x[:10] + '.csv' not in id_ready]
        # self.ids = ["subject014.csv"]
        for self.id in tqdm(self.ids, desc="Extracting the PPG fiducial points of each subject"):
            self.load_mat_data()
            if not self.data_error:
                self.extract_fiducial_points()
            else:
                self.error_handling(self.id)
                self.data_error = False

    """
    Extraction of the pulse arrival time (PAT) data per subject
    """
    def pat_extraction(self):
        """

        :return:
        """
        self.ids = os.listdir(self.target_path + "SelectedData/")
        if not self.replace:
            id_ready = os.listdir(self.target_path + "ExtractedPAT/ON/")
            self.ids = [x for x in self.ids if x[:10] + '.npy' not in id_ready]
        # self.ids = ["subject001.csv"]
        for self.id in tqdm(self.ids, desc="Extracting the PAT"):
            # loads the PPG and ECG segments
            self.load_mat_data()
            self.ppg_segment = np.load(self.data_path + "../processed/FilteredPPG/PPG/" + self.id[:10] + ".npy")
            self.vpg_segment = np.load(self.data_path + "../processed/FilteredPPG/VPG/" + self.id[:10] + ".npy")
            if not self.data_error:
                # loading the features and extracting the detection points per subject
                self.detection_points = (pd.read_csv(self.target_path + "SelectedData/" + self.id)["detectionPoint"])
                self.fiducials = pd.read_csv(self.target_path + "FiducialPoints/" + self.id)

                self.retrieve_PAT()
                np.save(self.target_path + "FilteredIndices/" + self.id[:10], self.filtered_indices)
            else:
                self.error_handling(self.id)
                self.data_error = False

    def get_filtered_detection_points(self):
        tmp = {
            "idx": [],
            "onset": []
        }
        for detection_point in self.detection_points:
            for idx, fiducial_onsets in enumerate(self.fiducials["on"]):
                if (fiducial_onsets >= detection_point - 10) and (fiducial_onsets <= detection_point + 10):
                    tmp["idx"].append(idx)
                    tmp["onset"].append(detection_point)
                    break
        return tmp["idx"], tmp["onset"]

    def retrieve_PAT(self):
        """
        Fiducial points data frame contains 15 columns with points per beat
        reference: https://pyppg.readthedocs.io/en/latest/tutorials/pyPPG_example.html
        :return:
        """

        indices, filtered_detection_points = self.get_filtered_detection_points()
        # the indices of the filtered detection points need to be matched to the vanilla indices of detection points
        self.filtered_indices = np.where(np.isin(filtered_detection_points, self.detection_points))[0]
        reference_points = self.fiducials.loc[indices]
        rPeaks = get_r_peaks(fs=self.fs, ecg_signal=self.ecg_segment)

        pat_values = {"on": [], "sp": [], "dn": [], "dp": [], "u": [], "it": []}
        for index, ref_pts in reference_points.iterrows():
            if not ref_pts["on"] < ref_pts["u"] < ref_pts["sp"]:
                ref_pts["u"] = np.argmax(self.vpg_segment[int(ref_pts["on"]):int(ref_pts["sp"])]) + int(ref_pts["on"])
            diff_onset_rPeak = ref_pts["on"] - rPeaks
            criteriaIdx = np.where((diff_onset_rPeak > 2) & (diff_onset_rPeak < 50))[0]
            if len(criteriaIdx) == 0:
                for key in pat_values:
                    if len(pat_values[key]) >= 3:
                        pat_values[key].append(np.mean(pat_values[key][-3:]))
                    else:
                        pat_values[key].append(pat_values[key][-1])
            else:
                selected_rPeak = rPeaks[criteriaIdx[np.argmin(diff_onset_rPeak[criteriaIdx])]]
                for key in pat_values:
                    if key == "it":
                        try:
                            tangent_slope_md = self.vpg_segment[int(ref_pts["u"])]
                            tangent_slope_v = self.vpg_segment[int(ref_pts["on"])]
                            # calculating the tangent intercept of md and v
                            tangent_intercept_md = self.ppg_segment[int(ref_pts["u"])] - self.vpg_segment[int(ref_pts["u"])] * ref_pts["u"]
                            tangent_intercept_v = self.ppg_segment[int(ref_pts["on"])] - self.vpg_segment[int(ref_pts["on"])] * ref_pts["on"]
                            # calculating the intersecting point of the tangents of v and md
                            intersecting_point = (tangent_intercept_v - tangent_intercept_md) / (tangent_slope_md - tangent_slope_v)
                            if ref_pts["on"] < intersecting_point < ref_pts["u"]:
                                pat_values[key].append(int(((intersecting_point - selected_rPeak) / self.fs) * 1000))
                            else:
                                check_fiducial_plot(ecg=self.ecg_segment, ppg=self.ppg_segment, vpg=self.vpg_segment,
                                                    ref_pts=ref_pts, key=key, rPeak=selected_rPeak, sub_id=self.id[:10])
                                pat_values[key].append(0)
                        except:
                            check_fiducial_plot(ecg=self.ecg_segment, ppg=self.ppg_segment, vpg=self.vpg_segment,
                                                ref_pts=ref_pts, key=key, rPeak=selected_rPeak, sub_id=self.id[:10])
                            pat_values[key].append(0)
                    else:
                        if not np.isnan(ref_pts[key]):
                            pat_values[key].append(int(((ref_pts[key] - selected_rPeak) / self.fs) * 1000))
                        else:
                            if len(pat_values[key]) >= 3:
                                pat_values[key].append(np.mean(pat_values[key][-3:]))
                            else:
                                pat_values[key].append(pat_values[key][-1])
        for key, values in pat_values.items():
            np.save(self.target_path + f"ExtractedPAT/{key.capitalize()}/" + self.id[:10], values)

    """
    Extraction of the different feature subsets
    """
    def feature_extraction(self):
        """
        Extraction of the different feature subsets
        - AF: all features
        - RF: features derived only from the reconstructed beat
        - OF: features derived only from the original beat
        - KF: features derived only from the kernels of the beat
        :return:
        """
        self.ids = os.listdir(self.target_path + "SelectedData/")
        if not self.replace:
            id_ready = os.listdir(self.target_path + "ExtractedFeatures/AF/")
            self.ids = [x for x in self.ids if x[:10] + '.npy' not in id_ready]
        for self.id in tqdm(self.ids, desc="Extracting the selected and cleaned subject data"):
            self.subject_data = pd.read_csv(self.target_path + "SelectedData/" + self.id)
            self.filtered_indices = np.load(self.target_path + "FilteredIndices/" + self.id[:10] + ".npy")
            self.subject_data = self.subject_data.iloc[self.filtered_indices]
            AF = ["b_a", "b_a_mod", "T_sys_dia", "T_sys_diaMode", "RI_area", "RI_peaks", "T1", "T2",
                            "P1", "P2", "kurt", "kurt_mod", "skew", "skew_mod", "SD", "SD_mod", "freq1",
                            "freq1_mod", "freq2", "freq2_mod", "freq3", "freq3_mod", "freq4", "freq4_mod",
                            "W1", "W2", "PulseWidth", "PulseWidth_mod", "PulseHeight", "PulseHeight_mod", "p",
                            "p_mod", "PWHA", "PWHA_mod"]
            RF = ["b_a_mod", "T_sys_diaMode", "T1", "T2", "P1", "P2", "kurt_mod", "skew_mod",
                                      "SD_mod", "freq1_mod", "freq2_mod", "freq3_mod", "freq4_mod",
                                      "W1", "W2", "PulseWidth_mod", "PulseHeight_mod", "p_mod", "PWHA_mod"]
            OF = ["b_a", "T_sys_dia", "RI_area", "RI_peaks", "kurt", "skew", "SD", "freq1",
                                 "freq2", "freq3", "freq4", "PulseWidth", "PulseHeight", "p", "PWHA"]
            KF = ["P1", "P2", "T1", "T2", "W1", "W2"]

            self.features_all = self.subject_data[AF]
            self.features_reconstructed = self.subject_data[RF]
            self.features_original = self.subject_data[OF]
            self.features_kernel = self.subject_data[KF]

            if not self.data_error:
                np.save(self.target_path + "ExtractedFeatures/AF/" + self.id[:10], self.features_all)
                np.save(self.target_path + "ExtractedFeatures/RF/" + self.id[:10], self.features_reconstructed)
                np.save(self.target_path + "ExtractedFeatures/OF/" + self.id[:10], self.features_original)
                np.save(self.target_path + "ExtractedFeatures/KF/" + self.id[:10], self.features_kernel)
            else:
                self.error_handling(self.id)
                self.data_error = False

    """
    Extraction of the blood pressure (BP) label/values per subject data
    """
    def bp_extraction(self):
        self.ids = os.listdir(self.target_path + "SelectedData/")
        if not self.replace:
            id_ready = os.listdir(self.target_path + "ExtractedBP/SBP/")
            self.ids = [x for x in self.ids if x[:10] + '.npy' not in id_ready]
        for self.id in tqdm(self.ids, desc="Extracting the blood pressure of each subject"):
            self.subject_data = pd.read_csv(self.target_path + "SelectedData/" + self.id)
            self.filtered_indices = np.load(self.target_path + "FilteredIndices/" + self.id[:10] + ".npy")
            self.subject_data = self.subject_data.iloc[self.filtered_indices]
            self.sbp = self.subject_data["FinapresSBP"]
            self.dbp = self.subject_data["FinapresDBP"]
            if not self.data_error:
                np.save(self.target_path + "ExtractedBP/SBP/" + self.id[:10], self.sbp)
                np.save(self.target_path + "ExtractedBP/DBP/" + self.id[:10], self.dbp)
            else:
                self.error_handling(self.id)
                self.data_error = False

    def create_data_frame(self):
        self.ids = os.listdir(self.target_path + "SelectedData/")
        if not self.replace:
            id_ready = os.listdir(self.target_path + "DataFrame/")
            self.ids = [x for x in self.ids if x[:10] + '.csv' not in id_ready]
        for self.id in tqdm(self.ids, desc="Extracting the blood pressure of each subject"):
            subject_data = pd.read_csv(self.target_path + "SelectedData/" + self.id)
            filtered_indices = np.load(self.target_path + "FilteredIndices/" + self.id[:10] + ".npy")
            subject_data = subject_data.iloc[filtered_indices]
            df = subject_data
            pat_list = {"on": [], "it": [], "u": [], "sp": [], "dn": [], "dp": []}
            for key, _ in pat_list.items():
                pat_list[key] = np.load(self.target_path + f"ExtractedPAT/{key.upper()}/" + self.id[:10] + ".npy", allow_pickle=True)
                assert (len(pat_list[key]) == len(subject_data))
                df[f"PAT({key.upper()})"] = pat_list[key]
            df = df.drop(columns=["ID", "Beat", "Age", "Gender", "detectionPoint", "corrABP", "corrPPG", "DC", "ABP"])
            df.to_csv(self.target_path + f"DataFrame/{self.id[:10]}.csv")

    def process(self):
        print("Starting the process of selecting and cleaning the raw feature data")
        self.replace = False
        self.selection_and_cleaning()
        print("Process complete \n")

        print("Starting the process of analysis and extraction of the fiducial points per beat")
        self.replace = False
        self.fiducial_points_extraction()
        print("Process complete \n")

        print("Starting the process of calculation and extraction of PAT")
        self.replace = True
        self.pat_extraction()
        print("Process complete \n")

        print("Starting the process of extracting the different feature sets")
        self.replace = False
        self.feature_extraction()
        print("Process complete \n")

        print("Starting the extraction of the blood pressure (BP) label/values per subject")
        self.replace = False
        self.bp_extraction()
        print("Process complete \n")

        print("Creating a Data Frame with Features, PATs and BPs")
        self.replace = False
        self.create_data_frame()
        print("Process complete \n")