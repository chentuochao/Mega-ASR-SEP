import torch
import torch.nn as nn
import torchaudio.functional as AF

from asteroid.losses.sdr import SingleSrcNegSDR






class HybridLoss_Paper(nn.Module):
    def __init__(self, 
                n_fft=4096,
                hop_length=1024,
                alpha=0.01,
                beta=0.3,
                eps=1e-8) -> None:
        super().__init__()

        self.l1 = nn.L1Loss()
        
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

        self.n_fft = n_fft
        self.hop_length = hop_length

        self.mse = nn.MSELoss()
        #self.loss_fn = SingleSrcNegSDR("sisdr")
        
    def compute_stft(self, audio):
        """
        audio: Tensor [C, T]
        returns: filtered Tensor [C, T]
        """

        B, C, T = audio.shape
        audio = audio.reshape(B*C, T)
        device = audio.device
        window = torch.hann_window(self.n_fft, device=device)
        # -------------------------
        # STFT
        # -------------------------
        S = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
        )  # shape: [C, F, frames]


        return S

    def si_snr(self, est, gt):
        """
        est, gt: [B, C, T]
        returns scalar
        """

        B, C, T = est.shape

        est = est.reshape(B * C, T)
        gt = gt.reshape(B * C, T)

        # zero-mean
        est = est - est.mean(dim=1, keepdim=True)
        gt = gt - gt.mean(dim=1, keepdim=True)

        # projection
        dot = torch.sum(est * gt, dim=1, keepdim=True)
        gt_energy = torch.sum(gt ** 2, dim=1, keepdim=True) + self.eps

        proj = dot * gt / gt_energy

        noise = est - proj

        ratio = (
            torch.sum(proj ** 2, dim=1)
            / (torch.sum(noise ** 2, dim=1) + self.eps)
        )

        sisnr = -10 * torch.log10(ratio + self.eps)

        return sisnr.mean()

    def forward(self, est: torch.Tensor, gt: torch.Tensor, **kwargs):
        """
        est: (B, C=1, T)
        gt: (B, C=1, T)
        """
        B, C, T = est.shape

        # ---------- SI-SNR ----------
        #loss_sisnr = self.loss_fn(est, gt)
        loss_sisnr = self.si_snr(est, gt)

        # ---------- STFT ----------
        S_est = self.compute_stft(est)
        S_gt = self.compute_stft(gt)

        mag_est = torch.abs(S_est) + self.eps
        mag_gt = torch.abs(S_gt) + self.eps


        # ---------- magnitude loss ----------
        loss_mag = self.mse(
            mag_est ** 0.3,
            mag_gt ** 0.3,
        )

        # ---------- real/imag compressed ----------
        real_est = S_est.real / (mag_est ** 0.7)
        real_gt = S_gt.real / (mag_gt ** 0.7)

        imag_est = S_est.imag / (mag_est ** 0.7)
        imag_gt = S_gt.imag / (mag_gt ** 0.7)

        loss_real = self.mse(real_est, real_gt)
        loss_imag = self.mse(imag_est, imag_gt)

        # ---------- total ----------
        total_loss = (
            self.alpha * loss_sisnr
            + (1 - self.beta) * loss_mag
            + self.beta * (loss_real + loss_imag)
        )

        return total_loss
         
        # return loss1
