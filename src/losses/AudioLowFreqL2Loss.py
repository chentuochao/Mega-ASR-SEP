import torch
import torch.nn as nn
import torchaudio.functional as AF


def brickwall_bandpass_stft(
    audio,
    sr=16000,
    cutoff=800,
    low_cutoff=0,
    n_fft=4096,
    hop_length=None,
    window=None,
):
    """
    audio: Tensor [C, T]
    returns: filtered Tensor [C, T]
    """

    B, C, T = audio.shape
    audio = audio.reshape(B*C, T)
    device = audio.device

    if hop_length is None:
        hop_length = n_fft // 4  # good reconstruction default

    if window is None:
        window = torch.hann_window(n_fft, device=device)

    # -------------------------
    # STFT
    # -------------------------
    S = torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    )  # shape: [C, F, frames]

    # -------------------------
    # Compute cutoff bin
    # -------------------------
    freqs = torch.linspace(0, sr / 2, n_fft // 2 + 1, device=device)
    cut_bin = torch.searchsorted(freqs, cutoff)
    low_cut_bin = torch.searchsorted(freqs, low_cutoff)

    # -------------------------
    # Brick-wall mask
    # -------------------------
    mask = torch.zeros_like(S)
    mask[:, low_cut_bin:cut_bin, :] = 1.0

    S_filtered = S * mask


    return S_filtered


class AudioLowFreqL2Loss(nn.Module):
    def __init__(self, sr, cut_off, low_cut_off=0) -> None:
        super().__init__()

        self.mse_loss = nn.MSELoss()
        self.cut_off = cut_off # upper limit on band
        self.low_cut_off = low_cut_off # upper limit on band
        self.sr = sr
        


    def forward(self, est: torch.Tensor, gt: torch.Tensor, **kwargs):
        """
        est: (B, C=1, T)
        gt: (B, C=1, T)
        """
        B, C, T = est.shape

        est = brickwall_bandpass_stft(est, sr=self.sr, cutoff=self.cut_off, low_cutoff = self.low_cut_off, n_fft=4096)
        gt = brickwall_bandpass_stft(gt, sr=self.sr, cutoff=self.cut_off, low_cutoff = self.low_cut_off, n_fft=4096)

        #est = est.reshape(B*C, T)
        #gt = gt.reshape(B*C, T)
        
        loss1 = self.mse_loss(est.real, gt.real) + self.mse_loss(est.imag, gt.imag)
         
        return loss1
