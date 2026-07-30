import torch
import torch.nn as nn
import torch.nn.functional as F

def filter_brickwall(audio, sr, high_cutoff, low_cutoff, n_fft=4096, hop_length=None, window=None):
    """
    audio: Tensor [B, C, T]
    returns: filtered Tensor [B, C, T]
    """

    B, C, T = audio.shape
    audio_reshaped = audio.reshape(B * C, T)
    device = audio.device

    if hop_length is None:
        hop_length = n_fft // 4

    if window is None:
        window = torch.hann_window(n_fft, device=device)

    # -------------------------
    # STFT
    # -------------------------
    S = torch.stft(
        audio_reshaped,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    )  # [B*C, F, frames]

    # -------------------------
    # Frequency bins
    # -------------------------
    freqs = torch.linspace(0, sr / 2, n_fft // 2 + 1, device=device)
    high_cut_bin = torch.searchsorted(freqs, high_cutoff)
    low_cut_bin = torch.searchsorted(freqs, low_cutoff)

    # -------------------------
    # Brick-wall mask
    # -------------------------
    mask = torch.zeros_like(S)
    mask[:, low_cut_bin:high_cut_bin, :] = 1.0

    S_filtered = S * mask

    # -------------------------
    # iSTFT (CRITICAL PART)
    # -------------------------
    audio_filtered = torch.istft(
        S_filtered,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        length=T,  # ensures same length output
    )  # [B*C, T]

    # reshape back
    audio_filtered = audio_filtered.reshape(B, C, T)

    return audio_filtered

def design_bandpass_fir(
    sr,
    low_cutoff,
    high_cutoff,
    num_taps=513,
    device="cpu",
    out_band_gain = None
):
    """
    Returns FIR filter [1, 1, K]
    """

    # time index centered at zero
    t = torch.arange(num_taps, device=device) - (num_taps - 1) / 2

    # normalized frequencies
    low = low_cutoff / sr
    high = high_cutoff / sr

    # sinc bandpass = highpass - lowpass
    h_high = 2 * high * torch.sinc(2 * high * t)
    h_low  = 2 * low  * torch.sinc(2 * low  * t)

    h = h_high - h_low

    # window (Hann)
    window = torch.hann_window(num_taps, device=device)
    h = h * window

    # normalize gain
    # h = h / h.sum()
    # Normalize in frequency domain
    H = torch.fft.rfft(h, n=8192)
    freqs = torch.fft.rfftfreq(8192, d=1/sr).to(device)

    center_freq = (low_cutoff + high_cutoff) / 2
    idx = torch.argmin(torch.abs(freqs - center_freq))

    h = h / H.abs()[idx]
    
    if out_band_gain:
        # ---------------------------------------------------
        # 🔥 KEY FIX: add spectral floor (prevents zero stopband)
        # ---------------------------------------------------

        h_id = torch.zeros_like(h)
        h_id[num_taps // 2] = 1.0  # identity (flat gain = 1)

        h = out_band_gain * h_id + (1 - out_band_gain) * h

    return h.view(1, 1, -1)

def filter_smooth_cutoff(audio, sr, low_cutoff, high_cutoff, num_taps = 513, out_band_gain=None):
    """
    audio: [B, C, T]
    h: [1, 1, K]
    """
    h = design_bandpass_fir(
        sr=sr,
        low_cutoff=low_cutoff,
        high_cutoff=high_cutoff,
        num_taps=num_taps,
        device=audio.device,
        out_band_gain = out_band_gain
    )


    B, C, T = audio.shape
    K = h.shape[-1]

    # expand filter per channel
    h = h.repeat(C, 1, 1)  # [C, 1, K]

    # padding to preserve length
    pad = K // 2
    audio_padded = F.pad(audio, (pad, pad))

    # grouped conv (per channel filtering)
    y = F.conv1d(audio_padded, h, groups=C)

    return y

class AudioL2Loss_with_filter_variations(nn.Module):
    def __init__(self, filter_name = None, filter_params = None, apply_on_gt = False, apply_on_est=False) -> None:
        super().__init__()

        self.mse_loss = nn.MSELoss()
        
        self.apply_on_gt = apply_on_gt
        self.apply_on_est = apply_on_est
        self.filter_name = filter_name
        self.filter_params = filter_params

        
    def apply_filters(self, sig):
        """
        sig: (B, C, T)
        """
        if self.filter_name == "brickwall":
            sig = filter_brickwall(sig, **self.filter_params)

        if self.filter_name == "smooth_cutoff":
            sig = filter_smooth_cutoff(sig, **self.filter_params)
        return sig

    def forward(self, est: torch.Tensor, gt: torch.Tensor, **kwargs):
        """
        est: (B, C, T)
        gt: (B, C, T)
        """
        B, C, T = est.shape

        #est = est.reshape(B*C, T)
        #gt = gt.reshape(B*C, T)

        if self.apply_on_gt:
            gt = self.apply_filters(gt)
        if self.apply_on_est:
            est = self.apply_filters(est)
        
        
        loss1 = self.mse_loss(est, gt)
         
        return loss1
