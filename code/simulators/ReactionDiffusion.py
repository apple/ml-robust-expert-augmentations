#  For licensing see accompanying LICENSE file.
#  Copyright (C) 2023 Apple Inc. All Rights Reserved.

import torch
import torch.nn as nn
from torchdyn.core import NeuralODE
from torchdyn import *
from code.simulators.GenericSimulator import PhysicalModel
import math
import torch.nn.functional as F

# This code is strongly inspired by https://github.com/yuan-yin/APHYNITY


class ReactionDiffusionPDE(PhysicalModel):
    def __init__(self, param_values, trainable_param):
        super(ReactionDiffusionPDE, self).__init__(param_values, trainable_param)
        self._X_dim = (2, 32, 32)
        self.a = nn.Parameter(torch.tensor(param_values["a"])) if "a" in trainable_param else param_values["a"]
        self.b = nn.Parameter(torch.tensor(param_values["b"])) if "b" in trainable_param else param_values["b"]
        self.k = nn.Parameter(torch.tensor(param_values["k"])) if "k" in trainable_param else param_values["k"]

        self._dx = 2./32.
        self.register_buffer('_laplacian', torch.tensor(
            [
                [0, 1, 0],
                [1, -4, 1],
                [0, 1, 0],
            ],
        ).float().view(1, 1, 3, 3) / (self._dx * self._dx))
        '''
        self.params_org = nn.ParameterDict({
            'a_org': nn.Parameter(torch.tensor(1e-3)),
            'b_org': nn.Parameter(torch.tensor(5e-3)),
            'k_org': nn.Parameter(torch.tensor(5e-3)),
        })
        '''

    def forward(self, t, x):
        return self.parameterized_forward(t, x)

    def parameterized_forward(self, t, x, **parameters):
        super(ReactionDiffusionPDE, self).parameterized_forward(t, x, **parameters)
        a = self.a if "a" not in parameters else parameters["a"].unsqueeze(2).unsqueeze(3)
        b = self.b if "b" not in parameters else parameters["b"].unsqueeze(2).unsqueeze(3)
        k = self.k if "k" not in parameters else parameters["k"].unsqueeze(2).unsqueeze(3)

        U = x[:, [0]]
        V = x[:, [1]]

        # if self.real_params is None:
        #    self.params['a'] = torch.sigmoid(self.params_org['a_org']) * 1e-2
        #    self.params['b'] = torch.sigmoid(self.params_org['b_org']) * 1e-2

        U_ = F.pad(U, pad=(1, 1, 1, 1), mode='circular')
        Delta_u = F.conv2d(U_, self._laplacian)

        V_ = F.pad(V, pad=(1, 1, 1, 1), mode='circular')
        Delta_v = F.conv2d(V_, self._laplacian)

        dUdt = a * Delta_u + U - U.pow(3) - V - k
        dVdt = b * Delta_v + U - V

        return torch.cat([dUdt, dVdt], dim=1)  # .reshape(-1, 2*32*32)

    def get_x_labels(self):
        return ["$U$", "$V$"]

    def get_name(self):
        return "Reaction Diffusion" + str(self.trainable_param)


class ReactionDiffusion:
    def __init__(self, init_param=None, true_param=None, T0=0., T1=5, n_timesteps=40, partial_model_param=None,
                 name="RLCCircuit", **kwargs):
        if partial_model_param is None:
            partial_model_param = ["a", "b"]
        self.full_param_dim_textual = ["a", "b", "k"]
        self.incomplete_param_dim_textual = partial_model_param
        self.init_param = self.prior_full_parameters() if init_param is None else init_param
        self.T0 = float(T0)
        self.T1 = float(T1)
        self.n_timesteps = int(n_timesteps)
        self.name = "ReactionDiffusion"
        self._X_dim = (2, 32, 32)
        self.true_param = {"a": 1e-3, "b": 5e-3, "k": 5e-3} if true_param is None else true_param

    def prior_incomplete_parameters(self):
        return {"a": .1, "b": .1}

    def prior_full_parameters(self):
        return {"a": .1, "b": .1, "k": .1}

    def sample_init_state(self, n=1):
        return torch.rand(n, 2, 32, 32)

    def sample_sequences(self, parameters=None, n=1, x0=None):
        if parameters is None:
            parameters = self.true_param
        x0 = self.sample_init_state(n) if x0 is None else x0
        t_span = torch.linspace(self.T0, self.T1, self.n_timesteps + 1)
        f = self.get_full_physical_model(parameters)
        model = NeuralODE(f, sensitivity='adjoint', solver='dopri5')
        with torch.no_grad():
            t_eval, y_hat = model(x0, t_span)
            return t_eval, y_hat# + torch.randn_like(y_hat) * .01

    def get_incomplete_physical_model(self, parameters, trainable=True) -> nn.Module:
        if trainable:
            for p in self.full_param_dim_textual:
                if p not in self.incomplete_param_dim_textual and p not in parameters.keys():
                    parameters[p] = 0.
            return ReactionDiffusionPDE(param_values=parameters, trainable_param=self.incomplete_param_dim_textual)
        else:
            return ReactionDiffusionPDE(param_values=parameters, trainable_param=[])

    def get_full_physical_model(self, parameters, trainable=False) -> nn.Module:
        if trainable:
            return ReactionDiffusionPDE(param_values=parameters, trainable_param=self.full_param_dim_textual)
        else:
            return ReactionDiffusionPDE(param_values=parameters, trainable_param=[])

