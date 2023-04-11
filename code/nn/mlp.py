#  For licensing see accompanying LICENSE file.
#  Copyright (C) 2023 Apple Inc. All Rights Reserved.

import torch.nn as nn
from code.nn.utils import ReactionDiffusionParametersScaler

act_dict = {"ReLU": nn.ReLU(),
            "Softplus": nn.Softplus(),
            "SELU": nn.SELU(),
            "ReactionDiffusionParametersScaler": ReactionDiffusionParametersScaler(),
            None: nn.Identity()}


class AddModule(nn.Module):
    def __init__(self, m1, m2):
        super(AddModule, self).__init__()
        self.m1 = m1
        self.m2 = m2

    def forward(self, x):
        return self.m1(x) + self.m2(x)


class MLP(nn.Module):
    def __init__(self, layers=None, linear_dim=0, hidden_act="ReLU", final_act=None):
        super(MLP, self).__init__()
        if layers is not None and len(layers) > 0:
            nn_lay = []
            for l1, l2 in zip(layers[:-1], layers[1:]):
                nn_lay += [nn.Linear(l1, l2), act_dict[hidden_act]]
            nn_lay.pop()
            nn_lay.append(act_dict[final_act])
            self.nn = nn.Sequential(*nn_lay)
        else:
            self.nn = lambda x: 0.
        if linear_dim > 0:
            self.linear = nn.Linear(linear_dim, linear_dim)
            self.net = AddModule(self.nn, self.linear)
        else:
            self.linear = lambda x: 0.
            self.net = self.nn


    def forward(self, x):
        return self.net(x)