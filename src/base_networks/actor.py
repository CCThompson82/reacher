from collections import OrderedDict

import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, nb_features, nb_actions, params, seed):
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        print('Cuda is available: {}'.format(torch.cuda.is_available()))
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.network = torch.nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(in_features=nb_features,
                                  out_features=params['network']['fc1'],
                                  bias=True)),
                ('relu1', nn.ReLU()),
                ('fc2', nn.Linear(in_features=params['network']['fc1'],
                                  out_features=params['network']['fc2'],
                                  bias=True)),
                ('relu2', nn.ReLU()),
                ('fc3', nn.Linear(in_features=params['network']['fc2'],
                                  out_features=nb_actions,
                                  bias=True)),
                ('tanh_out', nn.Tanh())]))
        self.network.to(self.device)

    def forward(self, state):
        """

        Args:
            state: torch.nn.Tensor object [batch_size, feature_size]

        Returns:
            target q_values for each action available
        """
        return self.network.forward(state.to(self.device))
