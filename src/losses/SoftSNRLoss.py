import torch
import torch.nn as nn


class SoftSNRLoss(nn.Module):
    def __init__(self, sisdr=False, SNRmax=20, **kwargs) -> None:
        super().__init__()
        self.EPS = 1e-9
        self.tau = 10 ** (-SNRmax / 10)
        self.sisdr = sisdr

    def forward(self, est: torch.Tensor, gt: torch.Tensor, **kwargs):
        """
        est: (B, C, T)
        gt: (B, C, T)
        """
        B, C, T = est.shape

        assert (torch.isnan(est).max() == 0), "Output tensor has nan!"
        assert (torch.isnan(gt).max() == 0), "GT tensor has nan!"

        if self.sisdr:
            alpha = torch.sum(est * gt, axis=-1) / torch.sum(gt ** 2, axis=-1)
            gt = gt * alpha

        sig_pwr = torch.sum(gt ** 2, axis=-1)
        noise_pwr = torch.sum((gt - est) ** 2, axis=-1)
        
        snr_loss = 10 * torch.log10 ( (noise_pwr + self.EPS) / (sig_pwr + self.EPS) + self.tau)
        snr_loss = snr_loss.mean(axis=1) # (B,)
        
        return snr_loss#  Negative SNR

def test():
    x = torch.ones(2, 3, 10)
    y = x + torch.zeros(2, 3, 10)

    loss = SoftSNRLoss()
    print(loss(y, x))

if __name__ == "__main__":
    test()
