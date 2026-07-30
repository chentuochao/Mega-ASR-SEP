"""
Pure-PyTorch MarbleNet frame-VAD model.

Exactly mirrors NeMo's EncDecFrameClassificationModel used by
nvidia/frame_vad_multilingual_marblenet_v2.0 so that:
  - NeMo pretrained weights can be ported with a 1-to-1 state_dict copy, and
  - the model can be trained from scratch without NeMo.

Forward signature is identical to the NeMo model:
    logits = model(input_signal=audio_tensor, input_signal_length=length)
    # logits: [B, T_frames, num_classes]

Architecture
------------
  Preprocessor : STFT → 80-mel → log → per-feature normalize
                 (n_fft=512, win=400=25ms, hop=160=10ms → 10ms mel frames)
  Encoder (6 JasperBlocks):
    Block 0 – Prologue  : SepConv(80→128, K=11, stride=2)  ← stride=2 gives 20ms output
    Block 1 – B1 (×2)   : SepConv(128→64,  K=13, residual)
    Block 2 – B2 (×2)   : SepConv(64→64,   K=15, residual)
    Block 3 – B3 (×2)   : SepConv(64→64,   K=17, residual)
    Block 4 – Epilogue-1: SepConv(64→128,  K=29, D=2)
    Block 5 – Epilogue-2: Conv(128→128,    K=1)
  Decoder : Linear(128→num_classes) applied per frame → [B, T, num_classes]

Key naming mirrors NeMo so that weight porting is a 1-to-1 copy:
  encoder.encoder.{0-5}.mconv.{j}.*      (encoder conv layers)
  encoder.encoder.{1-3}.res.0.{0,1}.*   (residual projections)
  decoder.layer0.weight / bias           (frame classifier)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MarbleNetConfig:
    """
    Hyper-parameters for MarbleNet-3x2x64.
    Defaults reproduce nvidia/frame_vad_multilingual_marblenet_v2.0.
    """
    sample_rate: int = 16000
    n_mels:      int = 80       # mel filterbanks  (NeMo uses 80)
    n_fft:       int = 512
    win_length:  int = 400      # 25 ms at 16 kHz
    hop_length:  int = 160      # 10 ms at 16 kHz
    num_classes: int = 2        # ['background', 'speech']
    dither:      float = 1e-5
    dropout:     float = 0.1    # NeMo default
    bn_eps:      float = 1e-3   # NeMo BatchNorm1d uses eps=0.001


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

class AudioPreprocessor(nn.Module):
    """
    Raw 16 kHz waveform [B, T_samples]
        → log-mel [B, 80, T_mel]  at 10 ms / frame
        → per-feature normalised (NeMo normalize='per_feature')

    The encoder's prologue then halves T_mel via stride=2, giving 20 ms output frames.
    """

    def __init__(self, cfg: MarbleNetConfig):
        super().__init__()
        self.hop_length = cfg.hop_length
        self.dither     = cfg.dither
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            win_length=cfg.win_length,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            window_fn=torch.hann_window,
            power=2.0,
            normalized=False,
        )
        self.register_buffer('_device_anchor', torch.zeros(1))

    def forward(
        self,
        waveform: torch.Tensor,                   # [B, T_samples]
        lengths:  Optional[torch.Tensor] = None,  # [B] sample counts
    ):
        if self.dither > 0 and self.training:
            waveform = waveform + self.dither * torch.randn_like(waveform)

        self.mel = self.mel.to(waveform.device)
        mel      = self.mel(waveform)                     # [B, n_mels, T]
        log_mel  = torch.log(mel.clamp(min=2 ** -24))

        # per-feature (per-bin) normalisation over time axis – matches NeMo
        mean    = log_mel.mean(dim=-1, keepdim=True)
        std     = log_mel.std(dim=-1, keepdim=True).clamp(min=1e-5)
        log_mel = (log_mel - mean) / std

        if lengths is not None:
            feat_lengths = (lengths // self.hop_length).long()
        else:
            feat_lengths = torch.full(
                (waveform.shape[0],), log_mel.shape[-1],
                dtype=torch.long, device=waveform.device,
            )

        return log_mel, feat_lengths


# ---------------------------------------------------------------------------
# Building blocks – naming mirrors NeMo for 1-to-1 weight porting
# ---------------------------------------------------------------------------

class MaskedConv1d(nn.Module):
    """
    Thin wrapper so the inner Conv1d is accessible as .conv — this makes
    the state dict key  mconv.N.conv.weight  match NeMo's layout exactly.
    """
    def __init__(self, conv: nn.Conv1d):
        super().__init__()
        self.conv = conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class JasperBlock(nn.Module):
    """
    One NeMo JasperBlock with optional repeat and residual.

    Sub-block structure (separable conv, repeat=2, residual example = NeMo Block 1):
        mconv[0]: MaskedConv1d(dw, K, in_ch→in_ch)
        mconv[1]: MaskedConv1d(pw, 1, in_ch→out_ch)
        mconv[2]: BatchNorm1d(out_ch)
        mconv[3]: ReLU          ← non-last sub-blocks only
        mconv[4]: Dropout       ← non-last sub-blocks only
        mconv[5]: MaskedConv1d(dw, K, out_ch→out_ch)
        mconv[6]: MaskedConv1d(pw, 1, out_ch→out_ch)
        mconv[7]: BatchNorm1d(out_ch)
        res[0][0]: MaskedConv1d(1x1, in_ch→out_ch)
        res[0][1]: BatchNorm1d(out_ch)
        mout:     Sequential(ReLU, Dropout)

    The module indices (especially the jump from 2→5) are caused by ReLU and
    Dropout having no parameters and thus not appearing in the state_dict.
    """

    def __init__(
        self,
        in_ch:     int,
        out_ch:    int,
        kernel:    int,
        dilation:  int  = 1,
        separable: bool = True,
        residual:  bool = False,
        repeat:    int  = 1,
        dropout:   float = 0.1,
        stride:    int  = 1,
        bn_eps:    float = 1e-3,
    ):
        super().__init__()

        layers = []
        ch = in_ch
        pad = (kernel - 1) * dilation // 2

        for i in range(repeat):
            is_last = (i == repeat - 1)
            effective_stride = stride if i == 0 else 1

            if separable:
                layers.append(MaskedConv1d(nn.Conv1d(
                    ch, ch, kernel,
                    stride=effective_stride, padding=pad, dilation=dilation,
                    groups=ch, bias=False,
                )))
                layers.append(MaskedConv1d(nn.Conv1d(ch, out_ch, 1, bias=False)))
            else:
                layers.append(MaskedConv1d(nn.Conv1d(
                    ch, out_ch, kernel,
                    padding=pad, dilation=dilation, bias=False,
                )))

            layers.append(nn.BatchNorm1d(out_ch, eps=bn_eps))

            if not is_last:
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(p=dropout))

            ch = out_ch

        self.mconv = nn.ModuleList(layers)

        if residual:
            self.res = nn.ModuleList([
                nn.ModuleList([
                    MaskedConv1d(nn.Conv1d(in_ch, out_ch, 1, bias=False)),
                    nn.BatchNorm1d(out_ch, eps=bn_eps),
                ])
            ])
        else:
            self.res = nn.ModuleList()

        self.mout = nn.Sequential(nn.ReLU(inplace=True), nn.Dropout(p=dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        for layer in self.mconv:
            x = layer(x)
        if self.res:
            skip = inp
            for layer in self.res[0]:
                skip = layer(skip)
            x = x + skip
        return self.mout(x)


# ---------------------------------------------------------------------------
# Encoder  (wraps JasperBlocks in a Sequential named .encoder)
# ---------------------------------------------------------------------------

class ConvASREncoder(nn.Module):
    """
    MarbleNet-3x2x64 encoder.  The inner Sequential is named .encoder so that
    state_dict keys are  encoder.encoder.{0-5}.*  matching NeMo.
    """
    def __init__(self, cfg: MarbleNetConfig):
        super().__init__()
        D = cfg.dropout
        E = cfg.bn_eps

        self.encoder = nn.Sequential(
            # Block 0 – Prologue: sep, K=11, stride=2, no residual
            JasperBlock(cfg.n_mels, 128, kernel=11, separable=True,
                        residual=False, repeat=1, stride=2, dropout=D, bn_eps=E),
            # Block 1 – B1: sep, K=13, repeat=2, residual (128→64)
            JasperBlock(128, 64, kernel=13, separable=True,
                        residual=True,  repeat=2, dropout=D, bn_eps=E),
            # Block 2 – B2: sep, K=15, repeat=2, residual (64→64)
            JasperBlock(64,  64, kernel=15, separable=True,
                        residual=True,  repeat=2, dropout=D, bn_eps=E),
            # Block 3 – B3: sep, K=17, repeat=2, residual (64→64)
            JasperBlock(64,  64, kernel=17, separable=True,
                        residual=True,  repeat=2, dropout=D, bn_eps=E),
            # Block 4 – Epilogue-1: sep, K=29, D=2, no residual (64→128)
            JasperBlock(64,  128, kernel=29, dilation=2, separable=True,
                        residual=False, repeat=1, dropout=D, bn_eps=E),
            # Block 5 – Epilogue-2: plain K=1, no residual (128→128)
            JasperBlock(128, 128, kernel=1,  separable=False,
                        residual=False, repeat=1, dropout=D, bn_eps=E),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, n_mels, T] → [B, 128, T//2]"""
        return self.encoder(x)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class FrameClassificationDecoder(nn.Module):
    """
    Per-frame linear classifier.

    Named .layer0 to match NeMo's MultiLayerPerceptron decoder:
      decoder.layer0.weight  [num_classes, 128]
      decoder.layer0.bias    [num_classes]
    """
    def __init__(self, cfg: MarbleNetConfig):
        super().__init__()
        self.layer0 = nn.Linear(128, cfg.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 128, T] → [B, T, num_classes]"""
        return self.layer0(x.transpose(1, 2))


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class MarbleNetVAD(nn.Module):
    """
    End-to-end MarbleNet frame VAD.

    Drop-in replacement for NeMo's EncDecFrameClassificationModel:
        logits = model(input_signal=waveform, input_signal_length=lengths)
        # logits : [B, T_frames, num_classes]   T_frames at 20 ms resolution
    """

    def __init__(self, cfg: Optional[MarbleNetConfig] = None):
        super().__init__()
        if cfg is None:
            cfg = MarbleNetConfig()
        self.cfg         = cfg
        self.preprocessor = AudioPreprocessor(cfg)
        self.encoder      = ConvASREncoder(cfg)
        self.decoder      = FrameClassificationDecoder(cfg)

    def forward(
        self,
        input_signal:        torch.Tensor,
        input_signal_length: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        features, _ = self.preprocessor(input_signal, input_signal_length)
        encoded     = self.encoder(features)
        return self.decoder(encoded)

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str):
        import dataclasses
        torch.save({
            'state_dict': self.state_dict(),
            'cfg':        dataclasses.asdict(self.cfg),
        }, path)

    @classmethod
    def from_checkpoint(cls, path: str, map_location='cpu') -> 'MarbleNetVAD':
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        cfg_dict = ckpt.get('cfg', {})
        cfg      = MarbleNetConfig(**cfg_dict) if cfg_dict else MarbleNetConfig()
        model    = cls(cfg=cfg)
        model.load_state_dict(ckpt['state_dict'])
        return model

    # ------------------------------------------------------------------
    # NeMo weight porting
    # ------------------------------------------------------------------

    def port_nemo_weights(self, nemo_state: dict, strict: bool = True) -> list:
        """
        Copy encoder + decoder weights from a NeMo state_dict into this model.

        NeMo keys  encoder.encoder.{i}.mconv.{j}.conv.weight  and
                   decoder.layer0.weight  map 1-to-1 to our key names.
        The preprocessor buffers (featurizer.window / fb) and loss / augmentation
        weights in the NeMo checkpoint are intentionally skipped.

        Parameters
        ----------
        nemo_state : dict
            Output of nemo_model.state_dict().
        strict : bool
            If True (default), raise if any custom-model key is missing from
            the NeMo state dict.  Set False to allow partial loading.

        Returns
        -------
        missing_keys : list[str]
            Keys present in our model but absent in the NeMo state dict.
        """
        our_state   = self.state_dict()
        new_state   = {}
        missing     = []

        for key, param in our_state.items():
            if key in nemo_state:
                nemo_val = nemo_state[key]
                if nemo_val.shape != param.shape:
                    raise ValueError(
                        f"Shape mismatch for {key}: "
                        f"NeMo={list(nemo_val.shape)}  custom={list(param.shape)}"
                    )
                new_state[key] = nemo_val
            else:
                missing.append(key)
                new_state[key] = param   # keep random init

        if strict and missing:
            raise RuntimeError(
                f"Missing {len(missing)} keys in NeMo state dict:\n" +
                "\n".join(f"  {k}" for k in missing)
            )

        self.load_state_dict(new_state)
        return missing

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
