#  For licensing see accompanying LICENSE file.
#  Copyright (C) 2023 Apple Inc. All Rights Reserved.

import torch
import torch.nn as nn


class Permute(nn.Module):
    def __init__(self, order):
        super().__init__()
        self.order = order

    def forward(self, x):
        return x.permute(self.order)


class ReactionDiffusionParametersScaler(nn.Module):
    def __init__(self, scale=.01):
        super(ReactionDiffusionParametersScaler, self).__init__()
        self.scale = scale

    def forward(self, x):
        return torch.sigmoid(x) * self.scale


def kl_gaussians(mu_1, sigma_1, mu_2, sigma_2):
    p = torch.distributions.Normal(mu_1, sigma_1)
    q = torch.distributions.Normal(mu_2, sigma_2)
    kl = torch.distributions.kl_divergence(p, q)
    return kl.sum(1)