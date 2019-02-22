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
        self.conv1 = nn.Conv1d(deepnovo_config.vocab_size * deepnovo_config.num_ion, num_units, 1)
        self.conv2 = nn.Conv1d(num_units, 2*num_units, 1)
        self.conv3 = nn.Conv1d(2*num_units, 4*num_units, 1)
        self.fc1 = nn.Linear(4*num_units, 2*num_units)
        self.fc2 = nn.Linear(2*num_units, num_units)
        self.fc3 = nn.Linear(num_units, deepnovo_config.vocab_size)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(num_units)
        self.bn2 = nn.BatchNorm1d(2*num_units)
        self.bn3 = nn.BatchNorm1d(4*num_units)
        self.bn4 = nn.BatchNorm1d(2*num_units)
        self.bn5 = nn.BatchNorm1d(num_units)

    def forward(self, x):
        """

        :param x: [batch * T, 26*8, N]
        :return:
            logit: [batch * T, 26]
        """
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
        self.spectrum_embedding_matrix = nn.Embedding(num_embeddings=deepnovo_config.MZ_SIZE+1,
                                                      embedding_dim=deepnovo_config.embedding_size,
                                                      padding_idx=0,
                                                      sparse=True,  # TODO(Rui) try sparse with false
                                                      )
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
        peaks_location = torch.unsqueeze(peaks_location, dim=1)
        peaks_embedded = self.spectrum_embedding_matrix(peaks_location)  # [batch, 1, N, embed_size]
        peaks_embedded = peaks_embedded * peaks_intensity.view(batch_size, 1, N, 1)  # multiply embedding by intensity
        ion_embedded = self.spectrum_embedding_matrix(location_index)
        peaks_embedded = peaks_embedded.repeat(1, T, 1, 1).view(batch_size*T, N, deepnovo_config.embedding_size)
            # [batch * T, N, embed_size]
        ion_embedded = ion_embedded.view(batch_size * T, vocab_size*num_ion, deepnovo_config.embedding_size)
            # [batch * T, 26*8, embed_size]
        score_matrix = torch.bmm(ion_embedded, peaks_embedded.transpose(1, 2))  # [batch * T, 26 * 8， N]
        result = self.t_net(score_matrix).view(batch_size, T, vocab_size)
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

    def step(self, candidate_intensity, aa_input, prev_hidden_state_tuple, direction):
        """
        :param candidate_intensity: [batch, 1, 26, 8, 10]
        :param aa_input: [batch, 1]
        :param prev_hidden_state_tuple: (h, c), each is [batch, 1, num_units]
        :param direction: enum class, whether forward or backward
        :return: (log_prob, new_hidden_state)
        log_prob: the pred log prob of shape [batch, 26]
        new_hidden_state: new hidden state for next step
        """
        if direction == Direction.forward:
            model = self.forward_model
        else:
            model = self.backward_model
        if deepnovo_config.use_lstm:
            assert candidate_intensity.size(1) == aa_input.size(1) == 1
        with torch.no_grad():
            logit, new_hidden_state = model(candidate_intensity, aa_input, prev_hidden_state_tuple)
            logit = torch.squeeze(logit, dim=1)
            log_prob = F.log_softmax(logit)
        return log_prob, new_hidden_state


