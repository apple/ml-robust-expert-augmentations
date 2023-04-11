#  For licensing see accompanying LICENSE file.
#  Copyright (C) 2023 Apple Inc. All Rights Reserved.

import torch
import torch.nn as nn
from torchdyn.core import NeuralODE
from torchdyn import *
from code.simulators.GenericSimulator import PhysicalModel
import math


class DampedPendulumODE(PhysicalModel):
    def __init__(self, param_values, trainable_param):
        super(DampedPendulumODE, self).__init__(param_values, trainable_param)
        self._X_dim = 2
        if "omega_0" in trainable_param:
            self.omega_0 = nn.Parameter(torch.tensor(param_values["omega_0"], requires_grad=True))
        else:
            self.register_buffer("omega_0", torch.tensor(param_values["omega_0"], requires_grad=False))

        if "alpha" in trainable_param:
            self.alpha = nn.Parameter(torch.tensor(param_values["alpha"], requires_grad=True))
        else:
            self.register_buffer("alpha", torch.tensor(param_values["alpha"], requires_grad=False))

        if "A" in trainable_param:
            self.A = nn.Parameter(torch.tensor(param_values["A"], requires_grad=True))
        else:
            self.register_buffer("A", torch.tensor(param_values["A"], requires_grad=False))

        if "phi" in trainable_param:
            self.phi = nn.Parameter(torch.tensor(param_values["phi"], requires_grad=True))
        else:
            self.register_buffer("phi", torch.tensor(param_values["phi"], requires_grad=False))

    def forward(self, t, x):
        return torch.cat((x[:, [1]], -self.omega_0 ** 2 * torch.sin(x[:, [0]]) - self.alpha * x[:, [1]]), 1)

    def u(self, t, A=None, phi=None, omega_0=None):
        return A*(omega_0**2)*torch.cos(2*math.pi*phi)

    def parameterized_forward(self, t, x, **parameters):
        super(DampedPendulumODE, self).parameterized_forward(None, x, **parameters)
        omega_0 = self.omega_0 if "omega_0" not in parameters else parameters["omega_0"]
        alpha = self.alpha if "alpha" not in parameters else parameters["alpha"]
        #A = self.A if "A" not in parameters else parameters["A"]
        phi = self.phi if "phi" not in parameters else parameters["phi"]
        return torch.cat((x[:, [1]], -omega_0 ** 2 * torch.sin(x[:, [0]]) - alpha * x[:, [1]]), 1)
        #+ self.u(t, A, phi, omega_0)

    def get_x_labels(self):
        return ["$\\theta$", "$\\dot \\theta$"]

    def get_name(self):
        return "Damped Pendulum"

    def to(self, device):
        super(DampedPendulumODE, self).to(device)
        self.omega_0 = self.omega_0.to(device)
        self.alpha = self.alpha.to(device)
        self.A = self.A.to(device)
        self.phi = self.phi.to(device)
        return self


class DampedPendulum:
    def __init__(self, init_param=None, true_param=None, T0=0., T1=20, n_timesteps=40, partial_model_param=None, name="DampedPendulum",
                 **kwargs):
        if partial_model_param is None:
            partial_model_param = ["omega_0"]
        self.full_param_dim_textual = ["omega_0", "alpha", "A", "phi"]
        self.incomplete_param_dim_textual = partial_model_param
        self.init_param = self.prior_full_parameters() if init_param is None else init_param
        self.T0 = float(T0)
        self.T1 = float(T1)
        self.n_timesteps = int(n_timesteps)
        self.name = name
        self.true_param = {"omega_0": 2 * math.pi / 6, "alpha": 0.1, "A": 0., "phi": 1} if true_param is None else true_param

    def prior_incomplete_parameters(self):
        T0 = torch.rand(1) * 7 + 3
        return {"omega_0": 2 * math.pi / T0}

    def prior_full_parameters(self):
        alpha = torch.rand(1) * .5
        omega_0 = self.prior_incomplete_parameters()["omega_0"]
        A = torch.rand(1) * 40.
        phi = torch.rand(1) + 1.
        return {"omega_0": omega_0, "alpha": alpha}

    def init_state_APHYNITY_CODE(self, n):
        y0 = torch.rand([n, 2]) * 2.0 - 1
        radius = (torch.rand([n, 1]) + 1.3).expand(-1, 2)
        y0 = y0 / torch.sqrt((y0 ** 2).sum(1).unsqueeze(1).expand(-1, 2)) * radius
        return y0

    def sample_init_state(self, n=1):
        #return self.init_state_APHYNITY_CODE(n)
        min_theta, max_theta = -math.pi / 2, math.pi / 2
        return torch.cat((torch.rand([n, 1]) * (max_theta - min_theta) + min_theta, torch.zeros([n, 1])), 1)

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
            return DampedPendulumODE(param_values=parameters, trainable_param=self.incomplete_param_dim_textual)
        else:
            return DampedPendulumODE(param_values=parameters, trainable_param=[])

    def get_full_physical_model(self, parameters, trainable=False) -> nn.Module:
        if trainable:
            return DampedPendulumODE(param_values=parameters, trainable_param=self.full_param_dim_textual)
        else:
            return DampedPendulumODE(param_values=parameters, trainable_param=[])

