#  For licensing see accompanying LICENSE file.
#  Copyright (C) 2023 Apple Inc. All Rights Reserved.

import torch
import torch.nn as nn


class PhysicalModel(nn.Module):
    def __init__(self, param_values, trainable_param):
        super(PhysicalModel, self).__init__()
        self._nb_parameters = len(trainable_param)
        self._X_dim = -1
        self.incomplete_param_dim_textual = []
        self.full_param_dim_textual = []
        self.missing_param_dim_textual = []
        self.trainable_param = trainable_param
        for p in param_values.keys():
            if p in trainable_param:
                self.incomplete_param_dim_textual.append(p)
            else:
                self.missing_param_dim_textual.append(p)
            self.full_param_dim_textual.append(p)

    def _nb_parameters(self):
        return self._nb_parameters

    def _X_dim(self):
        return self._nb_parameters

    def forward(self, t, x):
        pass

    def parameterized_forward(self, t, x, **parameters):
        if len(set(parameters.keys()) - set(self.trainable_param)) != 0:
            raise Exception("Parameterized forward physical arguments does not match the simulator specification. "
                            "Simulator: {} - kwargs: {}".format(self.trainable_param, parameters.keys()))
        pass

    def get_x_labels(self):
        return ["$x_%d$" for i in range(self._X_dim)]

    def get_name(self):
        return "Generic Simulator"


class GenericSimulator:
    def __init__(self):
        pass

    def sample_theta(self, n=1) -> torch.tensor:
        pass

    def sample_sequence(self, theta=None) -> tuple[torch.tensor, torch.tensor]:
        pass

    def sample_split_sequence(self, theta=None) -> tuple[torch.tensor, tuple[torch.tensor, torch.tensor]]:
        pass

    def get_uncomplete_forward_model(self) -> nn.Module:
        pass
