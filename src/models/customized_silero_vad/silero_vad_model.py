"""
Trainable PyTorch reimplementation of Silero VAD.

Architecture (verified against official JIT state dict):
    audio → STFT (Conv1d filter bank) → 4×Conv1d Encoder → LSTMCell Decoder → P(speech)

JIT state-dict key structure (v5 / v6):
    _model.*      — 16 kHz sub-model (STFT: 258ch × 256-kernel, stride 128)
    _model_8k.*   — 8 kHz sub-model  (STFT: 130ch × 128-kernel, stride 64)

This file exposes:
    SileroVAD        — single-rate model (16 kHz or 8 kHz), fully trainable
    SileroVADWrapper — houses both sub-models like the original JIT, drop-in replacement

Quick-start:
    model = SileroVAD.from_jit("silero_vad.jit")       # load 16 kHz weights
    model.eval(); model.reset_states()
    prob, state = model(chunk_512, sr=16000)            # [B,1,1], [2,B,128]

References:
    Official:  https://github.com/snakers4/silero-vad
    Community RE: https://github.com/lovemefan/Silero-vad-pytorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class STFT(nn.Module):
    """
    Learnable STFT approximation via a single Conv1d.

    The 2*(n_fft//2+1) output channels represent real + imaginary components
    of an n_fft-point DFT; magnitude = sqrt(real² + imag²).

    Default (16 kHz):  n_fft=256, stride=128 → Conv1d(1, 258, 256, stride=128)
    8 kHz variant:     n_fft=128, stride=64  → Conv1d(1, 130, 128, stride=64)

    Input : [B, T]          — waveform, T = context_size + chunk_size
    Output: [B, n_bins, F]  — magnitude spectrogram, F ≤ 6 frames
    """

    def __init__(self, n_fft: int = 256, hop_length: Optional[int] = None):
        super().__init__()
        self.n_fft     = n_fft
        self.hop_length = hop_length or n_fft // 2
        self.n_bins    = n_fft // 2 + 1          # 129 (16k) or 65 (8k)

        # Right-pad so the strided conv sees a full frame at the chunk boundary.
        # JIT uses ReflectionPad1d((0, n_fft//4)).
        self.padding = nn.ReflectionPad1d((0, n_fft // 4))

        # Weight stored as 'forward_basis_buffer' (no .weight suffix) in the JIT.
        # nn.Conv1d auto-registers it as forward_basis_buffer.weight — the
        # from_jit loader handles this remapping.
        self.forward_basis_buffer = nn.Conv1d(
            in_channels  = 1,
            out_channels = self.n_bins * 2,   # 258 or 130
            kernel_size  = n_fft,
            stride       = self.hop_length,
            padding      = 0,
            bias         = False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T]
        x   = self.padding(x).unsqueeze(1)         # [B, 1, T + pad]
        out = self.forward_basis_buffer(x)          # [B, n_bins*2, frames]
        real = out[:, :self.n_bins, :6]             # [B, n_bins, ≤6]
        imag = out[:, self.n_bins:, :6]
        return torch.sqrt(real ** 2 + imag ** 2 + 1e-9)   # magnitude [B, n_bins, F]


class EncoderBlock(nn.Module):
    """Conv1d + ReLU.  Named 'reparam_conv' to match the JIT weight keys."""

    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.reparam_conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.reparam_conv(x))


class Decoder(nn.Module):
    """
    Stateful single-step LSTM decoder.

    Input : encoder output [B, 128, T'] — typically T'=1 after full stride
    State : [2, B, 128] tensor (stack of h, c); pass torch.zeros(0) for fresh state
    Output: speech prob [B, 1, 1], updated state [2, B, 128]

    Uses nn.LSTM (not LSTMCell) so that forward_sequence() can call
    self.rnn(full_seq, (h0, c0)) in one shot without torch._VF tricks.
    For single-step streaming, seq_len=1 is equivalent to LSTMCell.
    """

    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(128, 128, batch_first=False)
        self.decoder = nn.Sequential(
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Conv1d(128, 1, kernel_size=1),   # index 2 in Sequential → decoder.2.*
            nn.Sigmoid(),
        )

    def forward(
        self,
        x:     torch.Tensor,       # [B, 128, T']
        state: torch.Tensor,       # [2, B, 128] or empty
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # nn.LSTM expects [seq_len, B, input]; use seq_len=1 for single-chunk streaming.
        x = x.squeeze(-1).unsqueeze(0)          # [1, B, 128]
        if state.numel():
            # state: [2, B, 128] → split into (h0, c0) each [1, B, 128]
            _, (h, c) = self.rnn(x, (state[0].unsqueeze(0), state[1].unsqueeze(0)))
        else:
            _, (h, c) = self.rnn(x)
        h     = h.squeeze(0)                    # [B, 128]
        c     = c.squeeze(0)                    # [B, 128]
        state = torch.stack([h, c])             # [2, B, 128]
        out   = self.decoder(h.unsqueeze(-1))   # [B, 1, 1]
        return out, state


# ---------------------------------------------------------------------------
# JIT → nn.LSTM key remapping
# ---------------------------------------------------------------------------

def _remap_jit_keys(sd: dict) -> dict:
    """
    Normalise a state-dict extracted from the official Silero JIT so it can be
    loaded into SileroVAD which uses nn.LSTM instead of nn.LSTMCell.

    Two remappings are applied:
        1. LSTMCell key names → nn.LSTM key names  (add '_l0' suffix)
        2. Bare STFT tensor   → Conv1d weight key  (add '.weight' suffix)
    """
    # LSTMCell stores weights without a layer index; nn.LSTM appends '_l0'.
    lstm_remap = {
        "decoder.rnn.weight_ih": "decoder.rnn.weight_ih_l0",
        "decoder.rnn.weight_hh": "decoder.rnn.weight_hh_l0",
        "decoder.rnn.bias_ih":   "decoder.rnn.bias_ih_l0",
        "decoder.rnn.bias_hh":   "decoder.rnn.bias_hh_l0",
    }
    sd = {lstm_remap.get(k, k): v for k, v in sd.items()}

    # STFT conv is stored as a plain tensor in the JIT (no '.weight' suffix).
    bare = "stft.forward_basis_buffer"
    if bare in sd:
        sd[bare + ".weight"] = sd.pop(bare)

    return sd


# ---------------------------------------------------------------------------
# Single-rate model  (matches one of _model / _model_8k inside the JIT)
# ---------------------------------------------------------------------------

class SileroVAD(nn.Module):
    """
    Fully trainable Silero VAD for one sample rate (16 kHz or 8 kHz).

    State-dict keys exactly match the JIT's inner sub-model after stripping
    the '_model.' (or '_model_8k.') prefix:

        stft.forward_basis_buffer.weight   [258, 1, 256]
        encoder.0.reparam_conv.{weight,bias}
        encoder.1.reparam_conv.{weight,bias}
        encoder.2.reparam_conv.{weight,bias}
        encoder.3.reparam_conv.{weight,bias}
        decoder.rnn.{weight_ih_l0,weight_hh_l0,bias_ih_l0,bias_hh_l0}
        decoder.decoder.2.{weight,bias}

    Streaming inference (internal state):
        model.reset_states()
        for chunk in stream:
            prob, _ = model(chunk)      # updates _state/_context internally

    Training (explicit state — enables BPTT):
        state = torch.zeros(0)
        for chunk, label in sequence:
            prob, state = model(chunk, state=state)
            loss = F.binary_cross_entropy(prob.squeeze(), label)
            loss.backward()
    """

    _CHUNK_SIZE   = {16000: 512, 8000: 256}
    _CONTEXT_SIZE = {16000:  64, 8000:  32}
    _N_FFT        = {16000: 256, 8000: 128}
    _HOP          = {16000: 128, 8000:  64}
    _N_BINS       = {16000: 129, 8000:  65}

    def __init__(self, sample_rate: int = 16000):
        super().__init__()
        assert sample_rate in (8000, 16000), "Only 8000 and 16000 Hz are supported"
        self.sample_rate  = sample_rate
        self.chunk_size   = self._CHUNK_SIZE[sample_rate]
        self.context_size = self._CONTEXT_SIZE[sample_rate]
        n_bins            = self._N_BINS[sample_rate]

        self.stft = STFT(
            n_fft      = self._N_FFT[sample_rate],
            hop_length = self._HOP[sample_rate],
        )
        # nn.Sequential with integer keys matches JIT: encoder.0, encoder.1, ...
        self.encoder = nn.Sequential(
            EncoderBlock(n_bins, 128, kernel_size=3, stride=1, padding=1),
            EncoderBlock(128,   64,  kernel_size=3, stride=2, padding=1),
            EncoderBlock(64,    64,  kernel_size=3, stride=2, padding=1),
            EncoderBlock(64,   128,  kernel_size=3, stride=1, padding=1),
        )
        self.decoder = Decoder()

        # Internal streaming state (detached between calls)
        self._state   = torch.zeros(0)
        self._context = torch.zeros(0)
        self._last_sr = 0
        self._last_batch_size = 0

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset_states(self, batch_size: int = 1):
        self._state   = torch.zeros(0)
        self._context = torch.zeros(0)
        self._last_sr = 0
        self._last_batch_size = 0

    def _check_reset(self, batch_size: int, sr: int):
        if (not self._last_batch_size
                or self._last_sr != sr
                or self._last_batch_size != batch_size):
            self.reset_states(batch_size)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward_chunk(
        self,
        data:  torch.Tensor,
        sr:    int                     = 16000,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            data  : audio chunk [B, chunk_size]
            sr    : sample rate (8000 or 16000)
            state : LSTM state [2, B, 128].
                    • None  → streaming mode (uses/updates internal _state)
                    • tensor → training/explicit mode (caller owns state)
                    • torch.zeros(0) → fresh stateless run

        Returns:
            prob  : speech probability [B, 1, 1]
            state : updated LSTM state [2, B, 128]
        """
        chunk_size   = self._CHUNK_SIZE[sr]
        context_size = self._CONTEXT_SIZE[sr]

        if data.shape[-1] != chunk_size:
            raise ValueError(
                f"Expected {chunk_size} samples at {sr} Hz, got {data.shape[-1]}"
            )

        batch_size = data.shape[0]
        streaming  = state is None

        if streaming:
            self._check_reset(batch_size, sr)
            if not self._context.numel():
                self._context = torch.zeros(
                    batch_size, context_size, device=data.device, dtype=data.dtype
                )
            ctx   = self._context
            state = self._state
        else:
            if not self._context.numel():
                ctx = torch.zeros(
                    batch_size, context_size, device=data.device, dtype=data.dtype
                )
            else:
                ctx = self._context

        x = torch.cat([ctx, data], dim=1)   # [B, context + chunk]
        x = self.stft(x)                    # [B, n_bins, frames]
        x = self.encoder(x)                 # [B, 128, T']
        prob, new_state = self.decoder(x, state)

        if streaming:
            self._context = data[..., -context_size:].detach()
            self._state   = new_state.detach()
            self._last_sr = sr
            self._last_batch_size = batch_size
        else:
            # Keep context for explicit-state mode too (caller may reuse model)
            self._context = data[..., -context_size:].detach()

        return prob, new_state

    def forward(
        self,
        audio: torch.Tensor,
        sr:    int = 16000,
    ) -> torch.Tensor:
        """
        Fully vectorised forward over a full audio sequence, designed for training.

        All three stages run without any Python-level loop:

            1. Frame construction  — torch.unfold  (zero Python iterations)
            2. STFT + 4×Conv1d     — one batched kernel over B × n_chunks frames
            3. LSTM decoder        — one cuDNN kernel via torch._VF.lstm
                                     (LSTMCell and LSTM share identical weight layout)
            4. Decoder head        — Conv1d(kernel_size=1) applied to all steps at once

        Does not touch or update the internal streaming state (_state, _context).

        Args:
            audio : [B, T]  full waveform; T should be a multiple of chunk_size
                            (trailing samples that don't fill a chunk are dropped)
            sr    : 8000 or 16000

        Returns:
            probs : [B, n_chunks]  speech probability for every chunk
        """
        chunk_size   = self._CHUNK_SIZE[sr]
        context_size = self._CONTEXT_SIZE[sr]
        B, T         = audio.shape
        n_chunks     = T // chunk_size
        dev, dtype   = audio.device, audio.dtype

        # ── 1. Build all [context | chunk] frames with a sliding window ──
        # Zero-pad CTX samples at the front so chunk-0 gets a silent context;
        # the unfold window naturally carries the last CTX samples of chunk i
        # as the context for chunk i+1.
        ctx_pad = torch.zeros(B, context_size, device=dev, dtype=dtype)
        padded  = torch.cat([ctx_pad, audio[:, :n_chunks * chunk_size]], dim=1)
        # [B, n_chunks, context_size + chunk_size]
        frames  = padded.unfold(dimension=1,
                                size=context_size + chunk_size,
                                step=chunk_size)

        # ── 2. STFT + Encoder — one batched forward over B*n_chunks ─────
        # .contiguous() is required: unfold() returns a non-contiguous view and
        # passing it directly to Conv1d can trigger a different SIMD kernel path,
        # producing O(1e-4) float32 rounding differences vs the sequential forward.
        frames_flat = frames.reshape(B * n_chunks, context_size + chunk_size).contiguous()
        spec        = self.stft(frames_flat)          # [B*n_chunks, n_bins, F]
        enc_out     = self.encoder(spec)              # [B*n_chunks, 128, 1]
        # frames_flat is laid out as (B, n_chunks) in memory — reshape accordingly,
        # then permute to (n_chunks, B) for the LSTM input convention.
        # Doing reshape(n_chunks, B, ...) directly would silently mix batch items.
        enc_out     = (enc_out
                       .reshape(B, n_chunks, 128, -1)   # [B, n_chunks, 128, 1]
                       .permute(1, 0, 2, 3)              # [n_chunks, B, 128, 1]
                       .contiguous())

        # ── 3. LSTM — one cuDNN kernel, zero Python loop ─────────────────
        enc_seq = enc_out.squeeze(-1)                  # [n_chunks, B, 128]
        h0 = torch.zeros(1, B, 128, device=dev, dtype=dtype)
        c0 = torch.zeros(1, B, 128, device=dev, dtype=dtype)
        lstm_out, _ = self.decoder.rnn(enc_seq, (h0, c0))  # [n_chunks, B, 128]

        # ── 4. Decoder head — all steps in one pass ───────────────────────
        # decoder = Sequential(Dropout(0.1), ReLU, Conv1d(128→1, k=1), Sigmoid)
        # Conv1d(kernel_size=1) is position-wise → works on any sequence length.
        h_all = lstm_out.permute(1, 2, 0)             # [B, 128, n_chunks]
        probs = self.decoder.decoder(h_all)            # [B, 1, n_chunks]
        return probs.squeeze(1)                        # [B, n_chunks]

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    @classmethod
    def from_jit(cls, jit_path: str, sample_rate: int = 16000) -> "SileroVAD":
        """
        Load weights from the official Silero VAD JIT checkpoint.

        The JIT wraps two sub-models:
            _model.*     — 16 kHz weights
            _model_8k.*  — 8 kHz weights

        This method extracts the appropriate sub-model weights and remaps two
        naming differences between the JIT and this implementation:

            JIT (LSTMCell naming)          → nn.LSTM naming
            ───────────────────────────────────────────────────
            decoder.rnn.weight_ih          → decoder.rnn.weight_ih_l0
            decoder.rnn.weight_hh          → decoder.rnn.weight_hh_l0
            decoder.rnn.bias_ih            → decoder.rnn.bias_ih_l0
            decoder.rnn.bias_hh            → decoder.rnn.bias_hh_l0
            stft.forward_basis_buffer      → stft.forward_basis_buffer.weight
        """
        prefix = "_model." if sample_rate == 16000 else "_model_8k."
        jit_sd = torch.jit.load(jit_path, map_location="cpu").state_dict()

        sd = {}
        for k, v in jit_sd.items():
            if k.startswith(prefix):
                sd[k[len(prefix):]] = v

        sd = _remap_jit_keys(sd)

        model = cls(sample_rate=sample_rate)
        missing, unexpected = model.load_state_dict(sd, strict=True)
        if missing:
            print(f"[SileroVAD] Missing   : {missing}")
        if unexpected:
            print(f"[SileroVAD] Unexpected: {unexpected}")
        return model
    
    @classmethod
    def from_hub(cls, sample_rate: int = 16000,
                 force_reload: bool = False) -> "SileroVAD":
        """Download the official Silero VAD from PyTorch Hub and load its weights."""
        hub_model, _ = torch.hub.load(
            "snakers4/silero-vad", "silero_vad",
            force_reload=force_reload, trust_repo=True,
        )
        prefix = "_model." if sample_rate == 16000 else "_model_8k."
        jit_sd = hub_model.state_dict()
        sd = {k[len(prefix):]: v for k, v in jit_sd.items() if k.startswith(prefix)}
        sd = _remap_jit_keys(sd)
        model = cls(sample_rate=sample_rate)
        model.load_state_dict(sd, strict=True)
        return model


# ---------------------------------------------------------------------------
# Dual-rate wrapper  (mirrors the top-level JIT — drop-in replacement)
# ---------------------------------------------------------------------------

class SileroVADWrapper(nn.Module):
    """
    Houses both the 16 kHz and 8 kHz sub-models exactly like the official JIT,
    so its state-dict is a drop-in replacement for the checkpoint.

    State-dict keys:
        _model.*      — 16 kHz SileroVAD
        _model_8k.*   — 8 kHz  SileroVAD

    Usage:
        wrapper = SileroVADWrapper.from_jit("silero_vad.jit")
        prob = wrapper(chunk_512, 16000)   # scalar, matches JIT output
    """

    def __init__(self):
        super().__init__()
        self._model    = SileroVAD(sample_rate=16000)
        self._model_8k = SileroVAD(sample_rate=8000)

    def forward(self, x: torch.Tensor, sr: int = 16000) -> torch.Tensor:
        model = self._model if sr == 16000 else self._model_8k
        prob, _ = model(x, sr=sr)
        return prob.squeeze(-1).mean()

    def reset_states(self, batch_size: int = 1):
        self._model.reset_states(batch_size)
        self._model_8k.reset_states(batch_size)

    @classmethod
    def from_jit(cls, jit_path: str) -> "SileroVADWrapper":
        jit_sd = torch.jit.load(jit_path, map_location="cpu").state_dict()
        sd = dict(jit_sd)
        for prefix in ("_model.", "_model_8k."):
            bare = f"{prefix}stft.forward_basis_buffer"
            if bare in sd:
                sd[bare + ".weight"] = sd.pop(bare)
        wrapper = cls()
        wrapper.load_state_dict(sd, strict=True)
        return wrapper


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=== Shape check (random weights) ===")
    model = SileroVAD(sample_rate=16000)
    model.eval()
    model.reset_states()
    dummy = torch.randn(1, 512)
    with torch.no_grad():
        prob, state = model(dummy, sr=16000)
    print(f"prob  : {prob.shape}   (expected [1,1,1])")
    print(f"state : {state.shape}  (expected [2,1,128])")
    total = sum(p.numel() for p in model.parameters())
    print(f"params: {total:,}")

    if len(sys.argv) > 1:
        jit_path = sys.argv[1]
        print(f"\n=== Loading JIT weights from {jit_path} ===")

        our_model = SileroVAD.from_jit(jit_path, sample_rate=16000)
        our_model.eval()
        our_model.reset_states()

        jit_model = torch.jit.load(jit_path, map_location="cpu")
        jit_model.eval()

        wrapper = SileroVADWrapper.from_jit(jit_path)
        wrapper.eval()
        wrapper.reset_states()

        print("\n=== Numerical comparison (5 sequential chunks) ===")
        torch.manual_seed(42)
        all_match = True
        for i in range(5):
            chunk = torch.randn(1, 512)
            with torch.no_grad():
                our_prob, _ = our_model(chunk, sr=16000)
                jit_prob    = jit_model(chunk, 16000)
                wrap_prob   = wrapper(chunk, 16000)
            ov, jv, wv = our_prob.item(), jit_prob.item(), wrap_prob.item()
            m1 = abs(ov - jv) < 1e-5
            m2 = abs(wv - jv) < 1e-5
            all_match = all_match and m1 and m2
            print(f"  [{i+1}] SileroVAD={ov:.6f}  Wrapper={wv:.6f}  JIT={jv:.6f}  "
                  f"match={'✓' if m1 and m2 else '✗'}")
        print(f"\nAll match: {all_match}")
