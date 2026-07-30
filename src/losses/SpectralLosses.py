import torch
import torch.nn as nn
import scipy
import numpy as np


def fft2gammatonemx(
    sr=20000, n_fft=2048, n_bins=64, width=1.0, fmin=0.0, fmax=11025, maxlen=1024
):
    """
    # Ellis' description in MATLAB:
    # [wts,cfreqa] = fft2gammatonemx(nfft, sr, nfilts, width, minfreq, maxfreq, maxlen)
    #      Generate a matrix of weights to combine FFT bins into
    #      Gammatone bins.  nfft defines the source FFT size at
    #      sampling rate sr.  Optional nfilts specifies the number of
    #      output bands required (default 64), and width is the
    #      constant width of each band in Bark (default 1).
    #      minfreq, maxfreq specify range covered in Hz (100, sr/2).
    #      While wts has nfft columns, the second half are all zero.
    #      Hence, aud spectrum is
    #      fft2gammatonemx(nfft,sr)*abs(fft(xincols,nfft));
    #      maxlen truncates the rows to this many bins.
    #      cfreqs returns the actual center frequencies of each
    #      gammatone band in Hz.
    #
    # 2009/02/22 02:29:25 Dan Ellis dpwe@ee.columbia.edu  based on rastamat/audspec.m
    # Sat May 27 15:37:50 2017 Maddie Cusimano, mcusi@mit.edu 27 May 2017: convert to python
    """

    wts = np.zeros([n_bins, n_fft], dtype=np.float32)

    # after Slaney's MakeERBFilters
    EarQ = 9.26449
    minBW = 24.7
    order = 1

    nFr = np.array(range(n_bins)) + 1
    em = EarQ * minBW
    cfreqs = (fmax + em) * np.exp(
        nFr * (-np.log(fmax + em) + np.log(fmin + em)) / n_bins
    ) - em
    cfreqs = cfreqs[::-1]

    GTord = 4
    ucircArray = np.array(range(int(n_fft / 2 + 1)))
    ucirc = np.exp(1j * 2 * np.pi * ucircArray / n_fft)
    # justpoles = 0 :taking out the 'if' corresponding to this.

    ERB = width * np.power(
        np.power(cfreqs / EarQ, order) + np.power(minBW, order), 1 / order
    )
    B = 1.019 * 2 * np.pi * ERB
    r = np.exp(-B / sr)
    theta = 2 * np.pi * cfreqs / sr
    pole = r * np.exp(1j * theta)
    T = 1 / sr
    ebt = np.exp(B * T)
    cpt = 2 * cfreqs * np.pi * T
    ccpt = 2 * T * np.cos(cpt)
    scpt = 2 * T * np.sin(cpt)
    A11 = -np.divide(
        np.divide(ccpt, ebt) + np.divide(np.sqrt(3 + 2 ** 1.5) * scpt, ebt), 2
    )
    A12 = -np.divide(
        np.divide(ccpt, ebt) - np.divide(np.sqrt(3 + 2 ** 1.5) * scpt, ebt), 2
    )
    A13 = -np.divide(
        np.divide(ccpt, ebt) + np.divide(np.sqrt(3 - 2 ** 1.5) * scpt, ebt), 2
    )
    A14 = -np.divide(
        np.divide(ccpt, ebt) - np.divide(np.sqrt(3 - 2 ** 1.5) * scpt, ebt), 2
    )
    zros = -np.array([A11, A12, A13, A14]) / T
    wIdx = range(int(n_fft / 2 + 1))
    gain = np.abs(
        (
            -2 * np.exp(4 * 1j * cfreqs * np.pi * T) * T
            + 2
            * np.exp(-(B * T) + 2 * 1j * cfreqs * np.pi * T)
            * T
            * (
                np.cos(2 * cfreqs * np.pi * T)
                - np.sqrt(3 - 2 ** (3 / 2)) * np.sin(2 * cfreqs * np.pi * T)
            )
        )
        * (
            -2 * np.exp(4 * 1j * cfreqs * np.pi * T) * T
            + 2
            * np.exp(-(B * T) + 2 * 1j * cfreqs * np.pi * T)
            * T
            * (
                np.cos(2 * cfreqs * np.pi * T)
                + np.sqrt(3 - 2 ** (3 / 2)) * np.sin(2 * cfreqs * np.pi * T)
            )
        )
        * (
            -2 * np.exp(4 * 1j * cfreqs * np.pi * T) * T
            + 2
            * np.exp(-(B * T) + 2 * 1j * cfreqs * np.pi * T)
            * T
            * (
                np.cos(2 * cfreqs * np.pi * T)
                - np.sqrt(3 + 2 ** (3 / 2)) * np.sin(2 * cfreqs * np.pi * T)
            )
        )
        * (
            -2 * np.exp(4 * 1j * cfreqs * np.pi * T) * T
            + 2
            * np.exp(-(B * T) + 2 * 1j * cfreqs * np.pi * T)
            * T
            * (
                np.cos(2 * cfreqs * np.pi * T)
                + np.sqrt(3 + 2 ** (3 / 2)) * np.sin(2 * cfreqs * np.pi * T)
            )
        )
        / (
            -2 / np.exp(2 * B * T)
            - 2 * np.exp(4 * 1j * cfreqs * np.pi * T)
            + 2 * (1 + np.exp(4 * 1j * cfreqs * np.pi * T)) / np.exp(B * T)
        )
        ** 4
    )
    # in MATLAB, there used to be 64 where here it says n_bins:
    wts[:, wIdx] = (
        ((T ** 4) / np.reshape(gain, (n_bins, 1)))
        * np.abs(ucirc - np.reshape(zros[0], (n_bins, 1)))
        * np.abs(ucirc - np.reshape(zros[1], (n_bins, 1)))
        * np.abs(ucirc - np.reshape(zros[2], (n_bins, 1)))
        * np.abs(ucirc - np.reshape(zros[3], (n_bins, 1)))
        * (
            np.abs(
                np.power(
                    np.multiply(
                        np.reshape(pole, (n_bins, 1)) - ucirc,
                        np.conj(np.reshape(pole, (n_bins, 1))) - ucirc,
                    ),
                    -GTord,
                )
            )
        )
    )
    wts = wts[:, range(maxlen)] * (1 / n_fft)

    return wts, cfreqs

def get_window(win_type: str, win_length: int):
    """Return a window function.

    Args:
        win_type (str): Window type. Can either be one of the window function provided in PyTorch
            ['hann_window', 'bartlett_window', 'blackman_window', 'hamming_window', 'kaiser_window']
            or any of the windows provided by [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.get_window.html).
        win_length (int): Window length

    Returns:
        win: The window as a 1D torch tensor
    """

    try:
        win = getattr(torch, win_type)(win_length)
    except:
        win = torch.from_numpy(scipy.signal.windows.get_window(win_type, win_length)).float()

    return win

class GammaToneLoss(nn.Module):
    def __init__(self, fs, nfft, fmin, fmax, nbins):
        super().__init__()
        self.gm_matrix, _ = fft2gammatonemx(sr=fs, n_fft=nfft, n_bins=nbins, fmin=fmin, fmax=fmax, maxlen=nfft//2 + 1)
        self.gm_matrix = torch.from_numpy(self.gm_matrix).T
        self.device = None

    def forward(self, est, gt):
        est_gm = torch.einsum('abc, bd -> adc', torch.abs(est), self.gm_matrix)
        gt_gm = torch.einsum('abc, bd -> adc', torch.abs(gt), self.gm_matrix)

        # print(est_gm.shape)
        # print(est, est_gm)

        return torch.abs(est_gm - gt_gm).mean()

class MRSpectralLosses(nn.Module):
    def __init__(self, sample_rate=16000, fft_sizes=[1024, 2048, 8192], hop_sizes=[256, 512, 2048], win_lengths=[1024, 2048, 8192], win_type='hann',
                 compression_factor = 1, w_cplx=1.0, w_mag=0, w_gamma=0, eps=1e-6) -> None:
        super().__init__()
        self.compression_factor = compression_factor
        
        self.w_cplx = w_cplx
        self.w_mag = w_mag
        self.w_gamma = w_gamma

        self.fft = fft_sizes
        self.hop = hop_sizes
        self.win = win_lengths
        self.win_type = win_type

        self.windows = [get_window(self.win_type, w) for w in self.win]
        self.device = None

        self.gamma_losses = nn.ModuleList()
        for f in self.fft:
            self.gamma_losses.append(GammaToneLoss(fs=sample_rate, nfft=f, fmin=20, fmax = sample_rate // 2, nbins=64))

        self.eps = eps

    def compress(self, x):
        if self.compression_factor != 1:
            mag = torch.abs(x) + self.eps
            normalized = x / mag
            compressed_mag = torch.pow(mag, self.compression_factor)
            return normalized * compressed_mag

        return x

    def mag_loss(self, e, g):
        absdiff = torch.abs(torch.abs(e) - torch.abs(g))
        return absdiff.mean()

    def cplx_loss(self, e, g):
        absdiff = torch.abs(e - g)
        return absdiff.mean()

    def forward(self, est: torch.Tensor, gt: torch.Tensor):
        """
        est, gt: [B, C, t]
        """

        est = est.flatten(0,1)
        gt = gt.flatten(0,1)

        if self.device != gt.device:
            # Move to CUDA
            for i in range(len(self.windows)):
                self.windows[i] = self.windows[i].to(gt.device)
                self.gamma_losses[i].gm_matrix = self.gamma_losses[i].gm_matrix.to(gt.device)
            # self.to(gt.device)
            self.device = gt.device

        n_resolutions = len(self.fft)
        loss = 0
        for r, (f, h, w, wd) in enumerate(zip(self.fft, self.hop, self.win, self.windows)):
            E = torch.stft(est, n_fft=f, hop_length=h, win_length=w, window=wd, return_complex=True)
            G = torch.stft(gt, n_fft=f, hop_length=h, win_length=w, window=wd, return_complex=True)

            E = self.compress(E)
            G = self.compress(G)

            if self.w_mag > 0:
                mag_loss = self.mag_loss(E, G)
                loss += self.w_mag * mag_loss
            
            if self.w_cplx > 0:
                cplx_loss = self.cplx_loss(E, G)
                loss += self.w_cplx * cplx_loss

            if self.w_gamma > 0:
                gamma_loss = self.gamma_losses[r](E, G)
                loss += self.w_gamma * gamma_loss
        
        return loss / n_resolutions
    
if __name__ == "__main__":
    loss = MRSpectralLosses(w_gamma = 1)
    e = torch.randn(2, 2, 16000)
    g = torch.randn(2, 2, 16000)
    x = loss(e, g)

    print(x)