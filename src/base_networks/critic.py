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

    def forward(self, states, actions):
        """

        Args:
            states: torch.nn.Tensor object [batch_size, feature_size]
            actions:

        Returns:
            target q_values for each action available
        """
        x = self.fc1(states)
        x = nn.ReLU(x)
        x = self.fc2(torch.cat(x, actions, dim=1))
        x = nn.ReLU(x)
        return self.fc3(x)
