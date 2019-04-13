from overrides import overrides
from typing import Dict, Tuple
import torch
import torch.nn as nn


class Flow(nn.Module):
    """
    Normalizing Flow base class
    """

    def __init__(self):
        super(Flow, self).__init__()

    @overrides
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor
                The random variable before flow
        Returns: y: Tensor, logdet: Tensor
            y, the random variable after flow
            logdet, the log determinant of :math:`\partial x / \partial y`
            Then the density :math:`\log(p(y)) = \log(p(x)) + logdet`
        """
        raise NotImplementedError

    def backward(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            y: Tensor
                The random variable after flow
        Returns: x: Tensor, logdet: Tensor
            x, the random variable before flow
            logdet, the log determinant of :math:`\partial x / \partial y`
            Then the density :math:`\log(p(y)) = \log(p(x)) + logdet`
        """
        raise NotImplementedError