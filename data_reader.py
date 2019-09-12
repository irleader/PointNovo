import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import csv
import re
import logging
from dataclasses import dataclass

import deepnovo_config
from deepnovo_cython_modules import process_spectrum, get_candidate_intensity

logger = logging.getLogger(__name__)


def parse_raw_sequence(raw_sequence: str):
    raw_sequence_len = len(raw_sequence)
    peptide = []
    index = 0
    while index < raw_sequence_len:
        if raw_sequence[index] == "(":
            if peptide[-1] == "C" and raw_sequence[index:index + 8] == "(+57.02)":
                peptide[-1] = "C(Carbamidomethylation)"
                index += 8
            elif peptide[-1] == 'M' and raw_sequence[index:index + 8] == "(+15.99)":
                peptide[-1] = 'M(Oxidation)'
                index += 8
            elif peptide[-1] == 'N' and raw_sequence[index:index + 6] == "(+.98)":
                peptide[-1] = 'N(Deamidation)'
                index += 6
            elif peptide[-1] == 'Q' and raw_sequence[index:index + 6] == "(+.98)":
                peptide[-1] = 'Q(Deamidation)'
                index += 6
            elif peptide[-1] == 'S' and raw_sequence[index:index + 8] == "(+79.97)":
                peptide[-1] = 'S(Phosphorylation)'
                index += 8
            elif peptide[-1] == 'T' and raw_sequence[index:index + 8] == "(+79.97)":
                peptide[-1] = 'T(Phosphorylation)'
                index += 8
            elif peptide[-1] == 'Y' and raw_sequence[index:index + 8] == "(+79.97)":
                peptide[-1] = 'Y(Phosphorylation)'
                index += 8
            else:  # unknown modification
                logger.warning(f"unknown modification in seq {raw_sequence}")
                return False, peptide
        else:
            peptide.append(raw_sequence[index])
            index += 1
    for aa in peptide:
        if aa not in deepnovo_config.vocab:
            logger.warning(f"unknown modification in seq {raw_sequence}")
            return False, peptide

    return True, peptide


def to_tensor(data_dict: dict) -> dict:
    temp = [(k, torch.from_numpy(v)) for k, v in data_dict.items()]
    return dict(temp)


def pad_to_length(input_data: list, pad_token, max_length: int) -> list:
    assert len(input_data) <= max_length
    result = input_data[:]
    for i in range(max_length - len(result)):
        result.append(pad_token)
    return result


@dataclass
class DDAFeature:
    feature_id: str
    mz: float
    z: float
    rt_mean: float
    peptide: list
    scan: str
    mass: float
    feature_area: str

@dataclass
class DenovoData:
    spectrum_holder: np.ndarray
    spectrum_original_forward: np.ndarray
    spectrum_original_backward: np.ndarray
    original_dda_feature: DDAFeature


@dataclass
class TrainData:
    spectrum_holder: np.ndarray
    forward_id_input: list
    forward_id_target: list
    backward_id_input: list
    backward_id_target: list
    forward_candidate_intensity: list
    backward_candidate_intensity: list


class DeepNovoTrainDataset(Dataset):
    def __init__(self, feature_filename, spectrum_filename, transform=None):
        """
        read all feature information and store in memory,
        :param feature_filename:
        :param spectrum_filename:
        """
        logger.info(f"input spectrum file: {spectrum_filename}")
        logger.info(f"input feature file: {feature_filename}")
        self.spectrum_filename = spectrum_filename
        self.input_spectrum_handle = None
        self.feature_list = []
        self.spectrum_location_dict = {}
        self.transform = transform
        # read spectrum location file
        spectrum_location_file = spectrum_filename + '.location.pytorch.pkl'
        if os.path.exists(spectrum_location_file):
            logger.info(f"read cached spectrum locations")
            with open(spectrum_location_file, 'rb') as fr:
                self.spectrum_location_dict = pickle.load(fr)
        else:
            logger.info("build spectrum location from scratch")
            spectrum_location_dict = {}
            line = True
            with open(spectrum_filename, 'r') as f:
                while line:
                    current_location = f.tell()
                    line = f.readline()
                    if "BEGIN IONS" in line:
                        spectrum_location = current_location
                    elif "SCANS=" in line:
                        scan = re.split('[=\r\n]', line)[1]
                        spectrum_location_dict[scan] = spectrum_location
            self.spectrum_location_dict = spectrum_location_dict
            with open(spectrum_location_file, 'wb') as fw:
                pickle.dump(self.spectrum_location_dict, fw)

        # read feature file
        skipped_by_mass = 0
        skipped_by_ptm = 0
        skipped_by_length = 0
        with open(feature_filename, 'r') as fr:
            reader = csv.reader(fr, delimiter=',')
            header = next(reader)
            feature_id_index = header.index(deepnovo_config.col_feature_id)
            mz_index = header.index(deepnovo_config.col_precursor_mz)
            z_index = header.index(deepnovo_config.col_precursor_charge)
            rt_mean_index = header.index(deepnovo_config.col_rt_mean)
            seq_index = header.index(deepnovo_config.col_raw_sequence)
            scan_index = header.index(deepnovo_config.col_scan_list)
            feature_area_index = header.index(deepnovo_config.col_feature_area)
            for line in reader:
                mass = (float(line[mz_index]) - deepnovo_config.mass_H) * float(line[z_index])
                ok, peptide = parse_raw_sequence(line[seq_index])
                if not ok:
                    skipped_by_ptm += 1
                    logger.debug(f"{line[seq_index]} skipped by ptm")
                    continue
                if mass > deepnovo_config.MZ_MAX:
                    skipped_by_mass += 1
                    logger.debug(f"{line[seq_index]} skipped by mass")
                    continue
                if len(peptide) >= deepnovo_config.MAX_LEN:
                    skipped_by_length += 1
                    logger.debug(f"{line[seq_index]} skipped by length")
                    continue
                new_feature = DDAFeature(feature_id=line[feature_id_index],
                                         mz=float(line[mz_index]),
                                         z=float(line[z_index]),
                                         rt_mean=float(line[rt_mean_index]),
                                         peptide=peptide,
                                         scan=line[scan_index],
                                         mass=mass,
                                         feature_area=line[feature_area_index])
                self.feature_list.append(new_feature)
        logger.info(f"read {len(self.feature_list)} features, {skipped_by_mass} skipped by mass, "
                    f"{skipped_by_ptm} skipped by unknown modification, {skipped_by_length} skipped by length")

    def __len__(self):
        return len(self.feature_list)

    def close(self):
        self.input_spectrum_handle.close()

    def _parse_spectrum_ion(self):
        mz_list = []
        intensity_list = []
        line = self.input_spectrum_handle.readline()
        while not "END IONS" in line:
            mz, intensity = re.split(' |\r|\n', line)[:2]
            mz_float = float(mz)
            intensity_float = float(intensity)
            # skip an ion if its mass > MZ_MAX
            if mz_float > deepnovo_config.MZ_MAX:
                line = self.input_spectrum_handle.readline()
                continue
            mz_list.append(mz_float)
            intensity_list.append(intensity_float)
            line = self.input_spectrum_handle.readline()
        return mz_list, intensity_list

    def _get_feature(self, feature: DDAFeature) -> TrainData:
        spectrum_location = self.spectrum_location_dict[feature.scan]
        self.input_spectrum_handle.seek(spectrum_location)
        # parse header lines
        line = self.input_spectrum_handle.readline()
        assert "BEGIN IONS" in line, "Error: wrong input BEGIN IONS"
        line = self.input_spectrum_handle.readline()
        assert "TITLE=" in line, "Error: wrong input TITLE="
        line = self.input_spectrum_handle.readline()
        assert "PEPMASS=" in line, "Error: wrong input PEPMASS="
        line = self.input_spectrum_handle.readline()
        assert "CHARGE=" in line, "Error: wrong input CHARGE="
        line = self.input_spectrum_handle.readline()
        assert "SCANS=" in line, "Error: wrong input SCANS="
        line = self.input_spectrum_handle.readline()
        assert "RTINSECONDS=" in line, "Error: wrong input RTINSECONDS="
        mz_list, intensity_list = self._parse_spectrum_ion()
        spectrum_holder, \
        spectrum_original_forward, \
        spectrum_original_backward = process_spectrum(mz_list, intensity_list, feature.mass)

        assert np.max(spectrum_holder) < 1.0 + 1e-5

        peptide_id_list = [deepnovo_config.vocab[x] for x in feature.peptide]
        forward_id_input = [deepnovo_config.GO_ID] + peptide_id_list
        forward_id_target = peptide_id_list + [deepnovo_config.EOS_ID]
        candidate_intensity_forward = []
        prefix_mass = 0.
        for i, id in enumerate(forward_id_input):
            prefix_mass += deepnovo_config.mass_ID[id]
            candidate_intensity = get_candidate_intensity(spectrum_original_forward, feature.mass, prefix_mass, 0)
            candidate_intensity_forward.append(candidate_intensity)

        backward_id_input = [deepnovo_config.EOS_ID] + peptide_id_list[::-1]
        backward_id_target = peptide_id_list[::-1] + [deepnovo_config.GO_ID]
        candidate_intensity_backward = []
        suffix_mass = 0
        for i, id in enumerate(backward_id_input):
            suffix_mass += deepnovo_config.mass_ID[id]
            candidate_intensity = get_candidate_intensity(spectrum_original_backward, feature.mass, suffix_mass, 1)
            candidate_intensity_backward.append(candidate_intensity)
        assert len(candidate_intensity_backward) == len(candidate_intensity_forward) == len(forward_id_target) == len(backward_id_target), \
            f"{len(candidate_intensity_backward)} {len(candidate_intensity_forward)} {len(forward_id_target)} {len(backward_id_target)}"
        return TrainData(spectrum_holder=spectrum_holder,
                         forward_id_input=forward_id_input,
                         forward_id_target=forward_id_target,
                         backward_id_input=backward_id_input,
                         backward_id_target=backward_id_target,
                         forward_candidate_intensity=candidate_intensity_forward,
                         backward_candidate_intensity=candidate_intensity_backward)

    def __getitem__(self, idx):
        if self.input_spectrum_handle is None:
            self.input_spectrum_handle = open(self.spectrum_filename, 'r')
        feature = self.feature_list[idx]
        return self._get_feature(feature)


def collate_func(train_data_list):
    """

    :param train_data_list: list of TrainData
    :return:
    """
    # sort data by seq length (decreasing order)
    train_data_list.sort(key=lambda x: len(x.forward_id_input), reverse=True)
    batch_max_seq_len = len(train_data_list[0].forward_id_input)
    intensity_shape = train_data_list[0].forward_candidate_intensity[0].shape
    spectrum_holder = [x.spectrum_holder for x in train_data_list]
    spectrum_holder = np.stack(spectrum_holder) # [batch_size, mz_size]
    spectrum_holder = torch.from_numpy(spectrum_holder)

    batch_forward_intensity = []
    batch_forward_id_input = []
    batch_forward_id_target = []
    for data in train_data_list:
        f_intensity = np.zeros((batch_max_seq_len, intensity_shape[0], intensity_shape[1], intensity_shape[2]),
                               np.float32)
        forward_intensity = np.stack(data.forward_candidate_intensity)
        f_intensity[:forward_intensity.shape[0], :, :, :] = forward_intensity
        batch_forward_intensity.append(f_intensity)

        f_input = np.zeros((batch_max_seq_len,), np.int64)
        f_target = np.zeros((batch_max_seq_len,), np.int64)
        forward_input = np.array(data.forward_id_input, np.int64)
        f_input[:forward_input.shape[0]] = forward_input
        forward_target = np.array(data.forward_id_target, np.int64)
        f_target[:forward_target.shape[0]] = forward_target
        batch_forward_id_input.append(f_input)
        batch_forward_id_target.append(f_target)

    batch_forward_intensity = torch.from_numpy(np.stack(batch_forward_intensity))  # [batch_size, batch_max_seq_len, 26, 8, 10]
    batch_forward_id_input = torch.from_numpy(np.stack(batch_forward_id_input))  # [batch_size, batch_max_seq_len]
    batch_forward_id_target = torch.from_numpy(np.stack(batch_forward_id_target))  # [batch_size, batch_max_seq_len]

    batch_backward_intensity = []
    batch_backward_id_input = []
    batch_backward_id_target = []
    for data in train_data_list:
        b_intensity = np.zeros((batch_max_seq_len, intensity_shape[0], intensity_shape[1], intensity_shape[2]),
                               np.float32)
        backward_intensity = np.stack(data.backward_candidate_intensity)
        b_intensity[:backward_intensity.shape[0], :, :, :] = backward_intensity
        batch_backward_intensity.append(b_intensity)

        b_input = np.zeros((batch_max_seq_len,), np.int64)
        b_target = np.zeros((batch_max_seq_len,), np.int64)
        backward_input = np.array(data.backward_id_input, np.int64)
        b_input[:backward_input.shape[0]] = backward_input
        backward_target = np.array(data.backward_id_target, np.int64)
        b_target[:backward_target.shape[0]] = backward_target
        batch_backward_id_input.append(b_input)
        batch_backward_id_target.append(b_target)

    batch_backward_intensity = torch.from_numpy(
        np.stack(batch_backward_intensity))  # [batch_size, batch_max_seq_len, 26, 8, 10]
    batch_backward_id_input = torch.from_numpy(np.stack(batch_backward_id_input))  # [batch_size, batch_max_seq_len]
    batch_backward_id_target = torch.from_numpy(np.stack(batch_backward_id_target))  # [batch_size, batch_max_seq_len]

    return (spectrum_holder,
            batch_forward_intensity,
            batch_forward_id_input,
            batch_forward_id_target,
            batch_backward_intensity,
            batch_backward_id_input,
            batch_backward_id_target)


# helper functions
def chunks(l, n: int):
    for i in range(0, len(l), n):
        yield l[i:i + n]


class DeepNovoDenovoDataset(DeepNovoTrainDataset):
    # override _get_feature method
    def _get_feature(self, feature: DDAFeature) -> DenovoData:
        spectrum_location = self.spectrum_location_dict[feature.scan]
        self.input_spectrum_handle.seek(spectrum_location)
        # parse header lines
        line = self.input_spectrum_handle.readline()
        assert "BEGIN IONS" in line, "Error: wrong input BEGIN IONS"
        line = self.input_spectrum_handle.readline()
        assert "TITLE=" in line, "Error: wrong input TITLE="
        line = self.input_spectrum_handle.readline()
        assert "PEPMASS=" in line, "Error: wrong input PEPMASS="
        line = self.input_spectrum_handle.readline()
        assert "CHARGE=" in line, "Error: wrong input CHARGE="
        line = self.input_spectrum_handle.readline()
        assert "SCANS=" in line, "Error: wrong input SCANS="
        line = self.input_spectrum_handle.readline()
        assert "RTINSECONDS=" in line, "Error: wrong input RTINSECONDS="
        mz_list, intensity_list = self._parse_spectrum_ion()
        spectrum_holder, \
        spectrum_original_forward, \
        spectrum_original_backward = process_spectrum(mz_list, intensity_list, feature.mass)

        return DenovoData(spectrum_holder=spectrum_holder,
                          spectrum_original_forward=spectrum_original_forward,
                          spectrum_original_backward=spectrum_original_backward,
                          original_dda_feature=feature)
