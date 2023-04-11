#  For licensing see accompanying LICENSE file.
#  Copyright (C) 2023 Apple Inc. All Rights Reserved.

from abc import abstractmethod
import torch.nn as nn
import torch


class HybridAutoencoder(nn.Module):
    def __init__(self):
        super(HybridAutoencoder, self).__init__()

    @abstractmethod
    def forward(self, t_span, x) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        pass

    @abstractmethod
    def augmented_data(self, t_span, x) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        pass

    @abstractmethod
    def predicted_parameters(self, t_span, x, zero_param=False) -> torch.FloatTensor:
        pass

    @abstractmethod
    def predicted_parameters_as_dict(self, t_span, x, zero_param=False) -> dict:
        pass

    @abstractmethod
    def loss(self, t_span, x) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        pass
