import torch.nn as nn
from asteroid.losses.sdr import PairwiseNegSDR
from asteroid.losses.pit_wrapper import PITLossWrapper
import torch
import typing


class SISDR_with_PIT_Loss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sisdrloss = PITLossWrapper(PairwiseNegSDR("sisdr"))
    
    def forward(self, est: typing.List[torch.Tensor], gt: typing.List[torch.Tensor]):
        """
        est: (B, S, T)
        gt: (B, S, T)

        returns (B, S, T)
        """

        B, S, T = est.shape

        sisdrloss, reordered = self.sisdrloss(est, gt, return_est=True)
        
        return sisdrloss, (reordered[:, 0:1], reordered[:, 1:2])
