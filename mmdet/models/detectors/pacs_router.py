# Copyright (c) FusionKD. All rights reserved.
import torch
import torch.nn as nn


class PacsRouter(nn.Module):
    """PaCS spatial routing network for T2 pseudo branch.

    The module is intentionally lightweight and shared across FPN levels.
    The last projection is zero-initialized so the router starts from a
    temporal-prior-only state.
    """

    def __init__(self,
                 in_channels: int = 256,
                 hidden_channels: int = 64,
                 num_branches: int = 1,
                 detach_input: bool = True) -> None:
        super().__init__()
        self.detach_input = detach_input
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.norm = nn.GroupNorm(8, hidden_channels)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_channels, num_branches, kernel_size=1)

        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='linear')
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = feat.detach() if self.detach_input else feat
        x = self.conv1(x)
        x = self.norm(x)
        x = self.act(x)
        return self.conv2(x)

    @staticmethod
    def compute_lb_loss(w_avg: torch.Tensor,
                        target_mean: torch.Tensor) -> torch.Tensor:
        return (w_avg - target_mean).pow(2)
