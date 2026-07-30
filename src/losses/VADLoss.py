import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.customized_silero_vad.silero_vad_model import SileroVAD


class VADLoss(nn.Module):
    """
    Frame-level VAD loss (binary cross-entropy).

    Inputs
    ------
    pred   : FloatTensor [B, T_frames] — sigmoid output of VAD model
    target : FloatTensor [B, T_frames] — soft or hard labels in [0, 1]

    Parameters
    ----------
    pos_weight : float
        Weight for positive (speech) frames. Use > 1 to penalise missed speech,
        < 1 to penalise false alarms.  Passed to F.binary_cross_entropy.
    label_smoothing : float
        Mixes hard labels toward 0.5: label = label * (1 - eps) + 0.5 * eps.
        Helps when the GT silero labels have small systematic biases.
    """

    def __init__(self, pos_weight: float = 1.0, label_smoothing: float = 0.0):
        super().__init__()
        self.pos_weight     = pos_weight
        self.label_smoothing = label_smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred   : [B, T]
        target : [B, T]
        """
        pred = pred.clamp(1e-7, 1 - 1e-7)

        if self.label_smoothing > 0:
            target = target * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        if self.pos_weight != 1.0:
            # Manually apply per-frame weight
            weight = torch.where(target >= 0.5,
                                 torch.full_like(target, self.pos_weight),
                                 torch.ones_like(target))
            loss = F.binary_cross_entropy(pred, target, weight=weight)
        else:
            loss = F.binary_cross_entropy(pred, target)

        return loss


class SileroVADLoss(nn.Module):
    """
    Auxiliary loss that encourages the separation model's output to have
    the same VAD profile as the ground-truth target.

    The frozen Silero VAD scores both `est` and `gt`; a VADLoss (BCE)
    is then computed between them.  Gradients flow through the VAD forward
    pass back into `est` but never update the VAD weights.

    Follows the same device-migration pattern as Wav2vecLoss.

    Parameters
    ----------
    vad_model_path : str
        Path to the official Silero JIT checkpoint (.jit file).
    sample_rate : int
        Must be 8000 or 16000 (must match the training sr).
    channel : int
        Which channel of the [B, C, T] tensor to feed to the VAD (default 0).
    pos_weight : float
        Weight on speech frames passed to VADLoss.
    label_smoothing : float
        Label smoothing passed to VADLoss.
    """

    def __init__(
        self,
        vad_model_path: str,
        sample_rate: int = 16000,
        channel: int = 0,
        pos_weight: float = 1.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.vad = SileroVAD.from_jit(vad_model_path, sample_rate=sample_rate)
        for param in self.vad.parameters():
            param.requires_grad_(False)
        # cuDNN's LSTM kernel requires train mode for backward(), so we cannot
        # call self.vad.eval().  Instead, disable only the Dropout layers so
        # VAD outputs are deterministic while the LSTM stays differentiable.
        self._disable_dropout()

        self.sr      = sample_rate
        self.bce     = VADLoss(pos_weight=pos_weight, label_smoothing=label_smoothing)
        self.device  = None
        self.channel = channel

    def _disable_dropout(self):
        for m in self.vad.modules():
            if isinstance(m, nn.Dropout):
                m.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # Re-disable Dropout after any train/eval switch; LSTM must stay in
        # train mode so cuDNN allows backward passes on CUDA.
        self._disable_dropout()
        return self

    def forward(self, est: torch.Tensor, gt: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        est : [B, C, T]  — separated audio  (gradients flow through this)
        gt  : [B, C, T]  — ground-truth target

        Returns a scalar loss.
        """
        if self.device != est.device:
            self.vad.to(est.device)
            self.device = est.device

        est_mono = est[:, self.channel, :]   # [B, T]
        gt_mono  = gt[:,  self.channel, :]   # [B, T]

        # GT soft labels — no gradients needed
        with torch.no_grad():
            gt_vad = self.vad(gt_mono, sr=self.sr)    # [B, n_chunks]
            gt_vad = gt_vad.detach()

        # Prediction — keep graph so gradients flow back to est
        est_vad = self.vad(est_mono, sr=self.sr)       # [B, n_chunks]

        return self.bce(est_vad, gt_vad)
