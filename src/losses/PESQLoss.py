import torch
import torch.nn as nn
from src.losses.SNRLosses import SNRLosses
from torch_pesq import PesqLoss


class PESQLoss(nn.Module):
    def __init__(self, pesq_factor=0.5, sr=16000) -> None:
        super().__init__()
        self.pesq_loss = PesqLoss(pesq_factor, sample_rate=sr)
        
        self.device = None

    def forward(self, est: torch.Tensor, gt: torch.Tensor, *args, **kwargs):
        """
        est, gt: [B, C, t]
        """
        B, C, t = gt.shape

        if self.device != gt.device:
            # Move to CUDA
            self.pesq_loss.to(gt.device)
            self.device = gt.device
        
        pesq_loss = self.pesq_loss(gt.flatten(0,1), est.flatten(0,1))
        
        return pesq_loss.reshape(B, C)
    
if __name__ == "__main__":
    loss = PESQLoss()

    x = torch.randn(2, 3, 10000)
    y = torch.randn(2, 3, 10000)

    z = loss(x, y)

    print(z)