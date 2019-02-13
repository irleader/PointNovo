import torch
import torch.nn as nn
import torch.nn.functional as F
import deepnovo_config


activation_func = F.relu


class SpectrumCNN(nn.Module):
    def __init__(self, dropout_p=0.25):
        super(SpectrumCNN, self).__init__()
        kernel_size = deepnovo_config.SPECTRUM_RESOLUTION
        reduced_size = int(deepnovo_config.MZ_SIZE / kernel_size)
        self.max_pool_1 = nn.MaxPool1d(kernel_size=kernel_size, stride=kernel_size)
        self.conv_1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=5, stride=1, padding=2)  # SAME padding
        self.conv_2 = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=5, stride=1, padding=2)  # SAME padding
        self.max_pool_2 = nn.MaxPool1d(kernel_size=6, stride=4, padding=1)  # SAME padding
        self.dropout = nn.Dropout(dropout_p)
        in_feature_size = reduced_size
        self.linear = nn.Linear(in_feature_size, deepnovo_config.num_units)

    def forward(self, spectrum_holder):
        batch_size, mz_size = spectrum_holder.szie()
        spectrum_holder = spectrum_holder.view(batch_size, 1, mz_size)
        net = self.max_pool_1(spectrum_holder)
        net = activation_func(self.conv_1(net))
        net = activation_func(self.conv_2(net))
        net = self.max_pool_2(net)
        net = self.dropout(net)
        net = net.view(batch_size, -1)
        net = activation_func(self.linear(net))
        return net


class IonCNN(nn.Module):
    def __init__(self, dropout_p=0.25):
        super(IonCNN, self).__init__()
        self.conv_1 = nn.Conv2d(deepnovo_config.vocab_size, 64, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv_2 = nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv_3 = nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.maxpool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.dropout = nn.Dropout(dropout_p)
        in_feature_size = int(deepnovo_config.num_ion * deepnovo_config.WINDOW_SIZE // 2 * 64)
        self.linear_1 = nn.Linear(in_feature_size, deepnovo_config.num_units)

    def forward(self, candidate_intensity):
        """

        :param candidate_intensity: [batch_size, T, 26, 8, 10]
        :return:
        ion cnn feature of size [batch_size, T, num_units]
        """
        batch_size, T, vocab_szie, num_ion, window_size = candidate_intensity.szie()
        candidate_intensity = candidate_intensity.view(batch_size*T, vocab_szie, num_ion, window_size)
        net = activation_func(self.conv_1(candidate_intensity))
        net = activation_func(self.conv_2(net))
        net = activation_func(self.conv_3(net))
        net = self.maxpool(net)  #[batch_size*T, 64, 8, 5]
        net = self.dropout(net)
        net = activation_func(self.linear_1(net))
        net = net.view(batch_size, T, deepnovo_config.num_units)
        return net


class DeepNovoLSTM(nn.Module):
    def __init__(self, vocab_size, num_ion, num_maps, num_units, dropout_rate=0.25):
        super(DeepNovoLSTM, self).__init__()
        self.num_units = num_units
        self.num_ion = num_ion
        self.vocab_size = vocab_size
