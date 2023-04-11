#  For licensing see accompanying LICENSE file.
#  Copyright (C) 2023 Apple Inc. All Rights Reserved.

import torch
import torch.nn as nn
from torchdyn.core import NeuralODE
from torchdyn import *
from code.simulators.GenericSimulator import PhysicalModel
import math


class DoublePendulumODE(PhysicalModel):
    def __init__(self, param_values=None, trainable_param=None):
        super(DoublePendulumODE, self).__init__({}, [])
        # Defining constants that we could potentially set as parameters
        self.g = 9.81
        self.m1 = 1.
        self.m2 = 1.
        self.l1 = 0.091
        self.l2 = 0.070
        self._X_dim = 4

    def forward(self, t, x):
        return self.parameterized_forward(t, x)

    def parameterized_forward(self, t, x, **parameters):
        #super(DoublePendulumODE, self).parameterized_forward(None, x, **parameters)
        g = self.g
        m1 = self.m1
        m2 = self.m2
        l1 = self.l1
        l2 = self.l2

        theta_1, theta_2, d_theta_1, d_theta_2 = torch.chunk(x, 4, 1)
        dd_theta_1 = (-g * (2 * m1 + m2) * torch.sin(theta_1) -
                      m2 * g * torch.sin(theta_1 - 2 * theta_2) -
                      2 * torch.sin(theta_1 - theta_2) *
                      m2 * (d_theta_2 ** 2 * l2 + d_theta_1 ** 2 * l1 * torch.cos(theta_1 - theta_2))) / \
                     (l1 * (2 * m1 + m2 - m2 * torch.cos(2 * theta_1 - 2 * theta_2)))
        dd_theta_2 = (2 * torch.sin(theta_1 - theta_2) * (d_theta_1 ** 2 * l1 * (m1 + m2) +
                                                          g * (m1 + m2) * torch.cos(theta_1) +
                                                          d_theta_2 ** 2 * l2 * m2 * torch.cos(theta_1 - theta_2))) / \
                    (l2 * (2 * m1 + m2 - m2 * torch.cos(2 * theta_1 - 2 * theta_2)))

        return torch.cat((d_theta_1, d_theta_2, dd_theta_1, dd_theta_2), 1)

    def get_x_labels(self):
        return ["$\\theta_0$", "$\\theta_1$", "$\\dot \\theta_0$", "$\\dot \\theta_1$"]

    def get_name(self):
        return "Double Pendulum"


class DoublePendulum:
    def __init__(self, init_param=None, true_param=None, T0=0., T1=20, n_timesteps=100, partial_model_param=None,
                 name="DoublePendulum",
                 **kwargs):
        self.full_param_dim_textual = []
        self.incomplete_param_dim_textual = partial_model_param
        self.init_param = self.prior_full_parameters() if init_param is None else init_param
        self.T0 = float(T0)
        self.T1 = float(T1)
        self.n_timesteps = int(n_timesteps)
        self.name = name
        self.true_param = {} if true_param is None else true_param

    def prior_incomplete_parameters(self):
        return {}

    def prior_full_parameters(self):
        return {}

    def sample_init_state(self, n=1):
        theta = torch.rand([n, 2]) * 2.0 * math.pi - math.pi
        return torch.cat([theta, torch.zeros_like(theta)], 1)

    def sample_sequences(self, parameters=None, n=1, x0=None):
        if parameters is None:
            parameters = self.true_param
        x0 = self.sample_init_state(n) if x0 is None else x0
        t_span = torch.linspace(self.T0, self.T1, self.n_timesteps + 1)
        f = self.get_full_physical_model(parameters)
        model = NeuralODE(f, sensitivity='adjoint', solver='dopri5', rtol=1e-7, atol=1e-7)
        with torch.no_grad():
            t_eval, y_hat = model(x0, t_span)
            return t_eval, y_hat + torch.randn_like(y_hat) * .01

    def get_incomplete_physical_model(self, parameters, trainable=True) -> nn.Module:
        if trainable:
            for p in self.full_param_dim_textual:
                if p not in self.incomplete_param_dim_textual:
                    parameters[p] = 0.
            return DoublePendulumODE(param_values=parameters, trainable_param=self.incomplete_param_dim_textual)
        else:
            return DoublePendulumODE(param_values=parameters, trainable_param=[])

    def get_full_physical_model(self, parameters, trainable=False) -> nn.Module:
        if trainable:
            return DoublePendulumODE(param_values=parameters, trainable_param=self.full_param_dim_textual)
        else:
            return DoublePendulumODE(param_values=parameters, trainable_param=[])
