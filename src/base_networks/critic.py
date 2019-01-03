from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as fn
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Network(nn.Module):
    def __init__(self, nb_features, nb_actions, params, seed):
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        print('Cuda is available: {}'.format(torch.cuda.is_available()))
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.bn0 = nn.BatchNorm1d(nb_features)
        self.fc1 = nn.Linear(in_features=nb_features,
                             out_features=params['network']['fc1'],
                             bias=True)
        self.fc2_with_concat = nn.Linear(
                    in_features=params['network']['fc1'] + nb_actions,
                    out_features=params['network']['fc2'],
                    bias=True)
        self.fc3 = nn.Linear(in_features=params['network']['fc2'],
                             out_features=1,
                             bias=True)
        # self.reset_parameters()
        #
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2_with_concat.weight.data.uniform_(*hidden_init(self.fc2_with_concat))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states, actions):
        """

        Args:
            states: torch.nn.Tensor object [batch_size, feature_size]
            actions: torch.nn.Tensor object [batch_size, action_size]

        Returns:
            target q_values for each action available
        """
        x = self.bn0(states.to(self.device))
        x = self.fc1(x)
        x = fn.relu(x)
        x = self.fc2_with_concat(torch.cat((x, actions.to(self.device)), dim=1))
        x = fn.relu(x)
        x = self.fc3(x)
        return x

    def init_uniform(self, layer):
        if type(layer) == nn.Linear:
            nn.init.uniform(layer, -1e-3, 1e-3)
