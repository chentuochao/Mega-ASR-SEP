import torch
import numpy as np


class FrequencyMaskingAugmentation_only_mixture:
    def __init__(self, min_freq_masks, max_freq_masks, fs, nfft=4096, reference_channels = [0, 1]):
        self.min_freq_masks = min_freq_masks
        self.max_freq_masks = max_freq_masks

        self.fs = fs
        self.nfft = nfft
        
        self.reference_channels = reference_channels

    def __call__(self, audio_data, gt_audio, rng: np.random.RandomState):
        C = audio_data.shape[0]
        T = audio_data.shape[-1]
        N = self.nfft //2 + 1

        cutoff = rng.randint(self.min_freq_masks, self.max_freq_masks + 1)
        cut_bin = int(cutoff * self.nfft / self.fs)

        self.window = torch.hamming_window(self.nfft, device=audio_data.device)

        
        
        augmented_audio_data = audio_data
        augmented_gt_audio = gt_audio
        
        for i in range(C):
            
            S = torch.stft(augmented_audio_data[i], n_fft=self.nfft, return_complex=True, window=self.window)
            S[cut_bin:, :] = 0
            augmented_audio_data[i] = torch.istft(S, n_fft=self.nfft, length=T, window=self.window)

            
        
        return augmented_audio_data, augmented_gt_audio
    
if __name__ == "__main__":
    augmentation_v1 = FrequencyMaskingAugmentation_only_mixture(min_freq_masks=1500, max_freq_masks=4000, fs=16000)

    x = torch.randn(4,16000)
    rng = np.random.RandomState(42)
    augmented_audio_data, augmented_gt_audio = augmentation_v1(x, x, rng)