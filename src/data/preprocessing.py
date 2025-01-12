class Processor:
    def __init__(self, data_path, replace, config_filter):
        self.data_path = data_path + "../data/raw/"
        self.target_path = data_path + "../data/processed/"
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