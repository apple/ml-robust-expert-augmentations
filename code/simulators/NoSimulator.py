#  For licensing see accompanying LICENSE file.
#  Copyright (C) 2023 Apple Inc. All Rights Reserved.

import torch
import torch.nn as nn
from torchdyn.core import NeuralODE
from torchdyn import *
from code.simulators.GenericSimulator import PhysicalModel
import math


class NoSimulator(PhysicalModel):
    def __init__(self, param_values=None, trainable_param=None):
        super(NoSimulator, self).__init__({}, [])

    def forward(self, t, x):
        return 0 * x

    def parameterized_forward(self, t, x, **parameters):
        return 0 * x

    def get_x_labels(self):
        return []

    def get_name(self):
        return "No Fp"