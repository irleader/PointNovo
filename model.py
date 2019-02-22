import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import deepnovo_config
from enum import Enum


activation_func = F.relu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_units = deepnovo_config.num_units


class TNet(nn.Module):
    """
    the T-net structure in the Point Net paper
    """
    def __init__(self):
        super(TNet, self).__init__()
        self.conv1 = nn.Conv1d(deepnovo_config.vocab_size * deepnovo_config.num_ion + 1, num_units, 1)
        self.conv2 = nn.Conv1d(num_units, 2*num_units, 1)
        self.conv3 = nn.Conv1d(2*num_units, 4*num_units, 1)
        self.fc1 = nn.Linear(4*num_units, 2*num_units)
        self.fc2 = nn.Linear(2*num_units, num_units)
        self.fc3 = nn.Linear(num_units, deepnovo_config.vocab_size)
        self.relu = nn.ReLU()

        self.input_batch_norm = nn.BatchNorm1d(26*8 + 1)
        self.bn1 = nn.BatchNorm1d(num_units)
        self.bn2 = nn.BatchNorm1d(2*num_units)
        self.bn3 = nn.BatchNorm1d(4*num_units)
        self.bn4 = nn.BatchNorm1d(2*num_units)
        self.bn5 = nn.BatchNorm1d(num_units)

    def forward(self, x):
        """

        :param x: [batch * T, 26*8+1, N]
        :return:
            logit: [batch * T, 26]
        """
        x = self.input_batch_norm(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x, _ = torch.max(x, dim=2)  # global max pooling
        assert x.size(1) == 4*num_units

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)  # [batch * T, 26]
        return x


class DeepNovoPointNet(nn.Module):
    def __init__(self):
        super(DeepNovoPointNet, self).__init__()
        self.t_net = TNet()

    def forward(self, location_index, peaks_location, peaks_intensity):
        """

        :param location_index: [batch, T, 26, 8] long
        :param peaks_location: [batch, N] N stands for MAX_NUM_PEAK, long
        :param peaks_intensity: [batch, N], float32
        :return:
            logits: [batch, T, 26]
        """

        N = peaks_location.size(1)
        assert N == peaks_intensity.size(1)
        batch_size, T, vocab_size, num_ion = location_index.size()

        peaks_location = peaks_location.view(batch_size, 1, N, 1)
        peaks_intensity = peaks_intensity.view(batch_size, 1, N, 1)
        peaks_location = peaks_location.repeat(1, T, 1, 1)  # [batch, T, N, 1]
        peaks_location_mask = (peaks_location > 1e-5).float()
        peaks_intensity = peaks_intensity.repeat(1, T, 1, 1)  # [batch, T, N, 1]

        location_index = location_index.view(batch_size, T, 1, vocab_size*num_ion)
        location_index_mask = (location_index > 1e-5).float()

        location_exp_minus_abs_diff = torch.exp(-torch.abs(peaks_location - location_index))  # [batch, T, N, 26*8]

        location_exp_minus_abs_diff = location_exp_minus_abs_diff * peaks_location_mask * location_index_mask

        input_feature = torch.cat((location_exp_minus_abs_diff, peaks_intensity), dim=3)
        input_feature = input_feature.view(batch_size*T, N, vocab_size *num_ion + 1)
        input_feature = input_feature.transpose(1, 2)

        result = self.t_net(input_feature).view(batch_size, T, vocab_size)
        return result


DeepNovoModel = DeepNovoPointNet


class Direction(Enum):
    forward = 1
    backward = 2


class InferenceModelWrapper(object):
    """
    a wrapper class so that the beam search part of code is the same for both with lstm and without lstm model.
    TODO(Rui): support no lstm branch here
    """
    def __init__(self, forward_model: DeepNovoModel, backward_model: DeepNovoModel, spectrum_cnn=None):
        if spectrum_cnn is None:
            assert deepnovo_config.use_lstm == False
        self.forward_model = forward_model
        self.backward_model = backward_model
        self.spectrum_cnn = spectrum_cnn
        # make sure all models are in eval mode
        self.forward_model.eval()
        self.backward_model.eval()
        if deepnovo_config.use_lstm:
            self.spectrum_cnn.eval()
        else:
            assert spectrum_cnn is None

    def initial_hidden_state(self, spectrum_holder_list: list):
        """
        get initial hidden state
        :param spectrum_holder: list of np.ndarray
        :return: (h0, c0), each is [num_layer, batch, num_units] tensor
        """
        if not deepnovo_config.use_lstm:
            return None
        else:
            temp = np.array(spectrum_holder_list)
            with torch.no_grad():
                spectrum_holder = torch.from_numpy(temp).to(device)
                return self.spectrum_cnn(spectrum_holder)

    def step(self, candidate_location, peaks_location, peaks_intensity, direction):
        """
        :param candidate_location: [batch, 1, 26, 8]
        :param peaks_location: [batch, N]
        :param peaks_intensity: [batch, N]
        :param direction: enum class, whether forward or backward
        :return: (log_prob, new_hidden_state)
        log_prob: the pred log prob of shape [batch, 26]
        """
        if direction == Direction.forward:
            model = self.forward_model
        else:
            model = self.backward_model

        with torch.no_grad():
            logit = model(candidate_location, peaks_location, peaks_intensity)
            logit = torch.squeeze(logit, dim=1)
            log_prob = F.log_softmax(logit)
        return log_prob


