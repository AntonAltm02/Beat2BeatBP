import os
from tqdm import tqdm
import numpy as np
import scipy
import pandas as pd
from biosppy import signals
from pyPPG import PPG
import pyPPG.preproc as PP
import pyPPG.fiducials as FP
from dotmap import DotMap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


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

        self.fiducial_points = None
        self.detection_points = None
        self.ppg_segment = None
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
            delete_rows_sbp = (raw_data["SBP"] > 250) | (raw_data["SBP"] < 50)
            raw_data = raw_data[~delete_rows_sbp]
            raw_data = raw_data.dropna(subset=["SBP"])
            # removing data rows with DBP values higher than 160 and lower than 30 mmHg
            delete_rows_dbp = (raw_data["DBP"] > 160) | (raw_data["DBP"] < 30)
            raw_data = raw_data[~delete_rows_dbp]
            raw_data = raw_data.dropna(subset=["DBP"])

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
        signal.start_sig = 0 # start sample of the signal
        signal.end_sig = -1 # last sample of the signal
        signal.fs = self.fs
        signal.v = self.ppg_segment
        signal.filtering = True  # whether to filter the PPG signal
        signal.fL = 0.5  # Lower cutoff frequency (Hz)
        signal.fH = 12  # Upper cutoff frequency (Hz)
        signal.order = 4  # Filter order
        signal.sm_wins = {'ppg': 50, 'vpg': 10, 'apg': 10, 'jpg': 10}  # smoothing windows in millisecond for the PPG, PPG', PPG'' and PPG'''

        prep = PP.Preprocessing(signal, filtering=True)
        signal.filt_ppg = prep[0]
        signal.filt_vpg = prep[1]
        signal.filt_apg = prep[2]
        signal.filt_jpg = prep[3]

        """
        Plot the raw and the derived signals before extracting the fiducial points
        """
        # setup figure
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=True, sharey=False)
        t = np.arange(0, len(signal.ppg)) / signal.fs
        # plot filtered PPG signal
        ax1.plot(t, signal.ppg)
        ax1.set(xlabel='', ylabel='Raw PPG')
        # plot filtered PPG signal
        ax2.plot(t, signal.filt_ppg)
        ax2.set(xlabel='', ylabel='PPG')
        # plot first derivative
        ax3.plot(t, signal.filt_vpg)
        ax3.set(xlabel='', ylabel='PPG\'')
        # plot second derivative
        ax4.plot(t, signal.filt_apg)
        ax4.set(xlabel='', ylabel='PPG\'\'')
        # plot third derivative
        ax5.plot(t, signal.filt_jpg)
        ax5.set(xlabel='Time (s)', ylabel='PPG\'\'\'')
        # show plot
        plt.show()

        """
        Extracting the fiducial points
        """
        s = PPG(signal)
        # initializing the fiducials package of pyPPG
        fpex = FP.FpCollection(s=s)
        # extracting the fiducials
        fiducials = fpex.get_fiducials(s=s)
        # Create a fiducials class
        fp = Fiducials(fp=fiducials)
        # Plot fiducial points
        plot_fiducials(s, fp, savingfolder, legend_fontsize=12)
        return fiducials

    def fiducial_points_extraction(self):
        self.ids = os.listdir(self.target_path + "SelectedData/")
        if not self.replace:
            id_ready = os.listdir(self.target_path + "FiducialPoints/")
            self.ids = [x for x in self.ids if x[:10] + '.csv' not in id_ready]
        for self.id in tqdm(self.ids, desc="Extracting the PPG fiducial points of each subject"):
            self.load_mat_data()
            if not self.data_error:
                self.fiducial_points = self.extract_fiducial_points()
                self.fiducial_points.to_csv(self.target_path + f"FiducialPoints/" + self.id, index=False)
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
        for self.id in tqdm(self.ids, desc="Extracting the PAT"):
            # loads the PPG and ECG segments
            self.load_mat_data()
            if not self.data_error:
                features = pd.read_csv(self.target_path + "SelectedData/" + self.id)
                self.subject_indices = np.array(features.index.array)
                self.detection_points = np.array(features["detectionPoint"])
                self.fiducial_points = pd.read_csv(self.target_path + "FiducialPoints/" + self.id)

                self.retrieve_PAT()
                np.save(self.target_path + "FilteredIndices/" + self.id[:10], self.filtered_indices)
            else:
                self.error_handling(self.id)
                self.data_error = False

    def retrieve_PAT(self):
        """
        Fiducial points data frame contains 15 columns with points per beat:
        - "on": the onset of the PPG beat
        - "sp": the systolic peak of the PPG beat
        - "dn": the dicrotic notch of the PPG beat
        - "dp": the diastolic peak of the PPG beat

        - "u": point in PPG' beat
        - "v": point in PPG' beat
        - "w": point in PPG' beat

        - "a": point in PPG'' beat
        - "b": point in PPG'' beat
        - "c": point in PPG'' beat
        - "d": point in PPG'' beat
        - "e": point in PPG'' beat
        - "f": point in PPG'' beat

        - "p1": point in PPG''' beat
        - "p2": point in PPG''' beat
        reference: https://pyppg.readthedocs.io/en/latest/tutorials/pyPPG_example.html
        :return:
        """

        def get_filtered_detection_points():
            tmp = []
            for detection_point in self.detection_points:
                for fiducial_onsets in self.fiducial_points["on"]:
                    if (fiducial_onsets >= detection_point - 10) and (fiducial_onsets >= detection_point + 10):
                        tmp.append(detection_point)
                        break
            return np.array(tmp)
        filtered_detection_points = get_filtered_detection_points()
        # the indices of the filtered detection points need to be matched to the vanilla indices of detection points
        self.filtered_indices = np.where(np.isin(filtered_detection_points, self.detection_points))[0]
        reference_points = self.fiducial_points.loc[self.filtered_indices]
        rPeaks = get_r_peaks(fs=self.fs, ecg_signal=self.ecg_segment)

        pat_values = {
            "on": [],
            "sp": [],
            "dn": [],
            "dp": [],
            "u": [],
            "it": []
        }
        vpg = np.gradient(self.ppg_segment)
        for _, ref_pts in reference_points.iterrows():
            diff_onset_rPeak = ref_pts["on"] - rPeaks
            criteriaIdx = np.where((diff_onset_rPeak > 7) & (diff_onset_rPeak < 35))[0]
            if len(criteriaIdx) == 0:
                for key in pat_values:
                    pat_values[key].append(0)
            else:
                selected_rPeak = rPeaks[np.max(criteriaIdx)]
                for key in pat_values:
                    if key == "it":
                        try:
                            tangent_slope_md = vpg[int(ref_pts["u"])]
                            tangent_slope_v = vpg[int(ref_pts["on"])]
                            # calculating the tangent intercept of md and v
                            tangent_intercept_md = self.ppg_segment[int(ref_pts["u"])] - vpg[int(ref_pts["u"])] * ref_pts["u"]
                            tangent_intercept_v = self.ppg_segment[int(ref_pts["on"])] - vpg[int(ref_pts["on"])] * ref_pts["on"]
                            # calculating the intersecting point of the tangents of v and md
                            intersecting_point = (tangent_intercept_v - tangent_intercept_md) / (
                                    tangent_slope_md - tangent_slope_v)
                            if ref_pts["on"] < intersecting_point < ref_pts["u"]:
                                pat_values[key].append(int(((intersecting_point - selected_rPeak) / self.fs) * 1000))
                            else:
                                pat_values[key].append(0)
                        except:
                            pat_values[key].append(0)
                    else:
                        if not np.isnan(ref_pts[key]):
                            pat_values[key].append(int(((ref_pts[key] - selected_rPeak) / self.fs) * 1000))
                        else:
                            pat_values[key].append(0)
        for key, values in pat_values.items():
            if key == "u" or key == "it":
                no_zeros = (pat_values[key] != 0)
                pat_values[key] = pat_values[key][no_zeros]
                tmp = np.where(pat_values[key] > (np.mean(pat_values[key]) + 0.25 * np.mean(pat_values[key])))[0]
                values = np.array(values)
                values[tmp] = 0
                pat_values[key] = values
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
            self.sbp = self.subject_data["SBP"]
            self.dbp = self.subject_data["DBP"]
            if not self.data_error:
                np.save(self.target_path + "ExtractedBP/SBP/" + self.id[:10], self.sbp)
                np.save(self.target_path + "ExtractedBP/DBP/" + self.id[:10], self.dbp)
            else:
                self.error_handling(self.id)
                self.data_error = False

    def process(self):
        print("Starting the process of selecting and cleaning the raw feature data")
        self.replace = False
        self.selection_and_cleaning()
        print("Process complete \n")

        print("Starting the process of analysis and extraction of the fiducial points per beat")
        self.replace = True
        self.fiducial_points_extraction()
        print("Process complete \n")

        print("Starting the process of calculation and extraction of PAT")
        self.replace = False
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