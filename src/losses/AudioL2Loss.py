import torch
import torch.nn as nn


class AudioL2Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.mse_loss = nn.MSELoss()
        


    def forward(self, est: torch.Tensor, gt: torch.Tensor, **kwargs):
        """
        est: (B, C, T)
        gt: (B, C, T)
        """
        B, C, T = est.shape

        #est = est.reshape(B*C, T)
        #gt = gt.reshape(B*C, T)
        
        loss1 = self.mse_loss(est, gt)
         
        return loss1
