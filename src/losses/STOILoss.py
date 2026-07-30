import torch
import torch.nn as nn
from torch_stoi import NegSTOILoss


class STOILoss(nn.Module):
    def __init__(self, sr=16000) -> None:
        super().__init__()
        self.stoi_loss = NegSTOILoss(sample_rate=sr)
        
        self.device = None

    def forward(self, est: torch.Tensor, gt: torch.Tensor, *args, **kwargs):
        """
        est, gt: [B, C, t]
        """
        B, C, t = gt.shape

        if self.device != gt.device:
            # Move to CUDA
            self.stoi_loss.to(gt.device)
            self.device = gt.device
        
        stoi_loss = self.stoi_loss(est.flatten(0,1), gt.flatten(0,1))
        
        return stoi_loss.reshape(B, C)
    
if __name__ == "__main__":
    loss = STOILoss()

    x = torch.randn(2, 3, 10000)
    y = torch.randn(2, 3, 10000)

    z = loss(x, y)

    print(z)