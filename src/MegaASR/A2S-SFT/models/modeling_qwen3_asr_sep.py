# coding=utf-8
# Mega-ASR-SEP: mixture-conditioned fusion on top of Qwen3-ASR.
#
# This module DOES NOT edit the upstream model. It vendors the pristine upstream
# code in `_modeling_qwen3_asr_base.py` (verbatim diff base, used for regression
# checks) and adds thin `*Sep` subclasses here. All fusion-specific logic lives
# in this file so the diff against upstream is isolated and reviewable.
#
# Four fusion strategies, one class each (see each class's docstring below for
# its own fusion math and init-as-no-op story):
#   late_gate   -- Qwen3ASRAudioEncoderLateGate   (2 tower passes, gate on output)
#   early_conv  -- Qwen3ASRAudioEncoderEarlyConv  (1 tower pass, parallel conv on input)
#   fddt        -- Qwen3ASRAudioEncoderFDDT       (1 tower pass, per-layer FiLM conditioning)
#   cross_attn  -- Qwen3ASRAudioEncoderCrossAttn  (2 tower passes, local cross-attention on output)
#
# Each concrete class independently owns its __init__ (its own parameters),
# reset_fusion_params (its own no-op init), and fuse (its own forward path).
# They share only Qwen3ASRAudioEncoderSepBase's `use_fusion` flag and the
# `_align_mix_len` helper -- deliberately NOT a shared chunking/conv helper,
# even though early_conv/fddt's chunking arithmetic looks similar, so that
# editing one fusion type's internals can never change another's behaviour.
#
# Everything downstream of the audio tower (placeholder scatter, LLM decoder,
# LM head, loss) is untouched: one audio placeholder span, same length, same
# position, for every fusion type. The LLM's input contract is unchanged.
from typing import Optional, Union

import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import BaseModelOutput
from transformers.utils import logging

from ._modeling_qwen3_asr_base import (
    Qwen3ASRAudioEncoder,
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRPreTrainedModel,
    Qwen3ASRThinkerCausalLMOutputWithPast,
    Qwen3ASRThinkerForConditionalGeneration,
    _get_feat_extract_output_lengths,
)

# Supported fusion strategies (see the module docstring / PLAN.md).
FUSION_LATE_GATE = "late_gate"    # gated residual on the tower OUTPUT (2 tower passes)
FUSION_EARLY_CONV = "early_conv"  # parallel conv on the mixture mel INPUT (1 tower pass)
FUSION_FDDT = "fddt"              # per-layer FiLM conditioning, mix primary / sep condition (1 tower pass)
FUSION_CROSS_ATTN = "cross_attn"  # local windowed cross-attention on the tower OUTPUT (2 tower passes)
FUSION_TYPES = (FUSION_LATE_GATE, FUSION_EARLY_CONV, FUSION_FDDT, FUSION_CROSS_ATTN)

logger = logging.get_logger(__name__)


class Qwen3ASRAudioEncoderSepBase(Qwen3ASRAudioEncoder):
    """Common base for every mixture-conditioned fusion strategy.

    Holds ONLY what every strategy needs -- the `use_fusion` flag and the
    mel-length-alignment helper -- no fusion-specific parameters or forward
    logic. Concrete strategies below subclass this and independently
    implement `__init__` (their own parameters), `reset_fusion_params` (their
    own no-op init), and `fuse` (their own forward path), so a change to one
    fusion type's subclass cannot affect another's behaviour.

    The base `forward()` (single-stream, inherited from `Qwen3ASRAudioEncoder`
    unchanged) is what every strategy's `fuse()` calls into for its "run the
    pretrained tower on one stream" step, so a plain (non-fusion) run is
    numerically identical to upstream regardless of which subclass is used.
    """

    fusion_type: str = None  # set by each concrete subclass

    def __init__(self, config):
        super().__init__(config)
        self.use_fusion = getattr(config, "use_fusion", False)

    def reset_fusion_params(self):
        """(Re)initialise this strategy's fusion parameters to a (near-)no-op.

        No-op by default; override in subclasses that add fusion parameters.
        MUST be re-applied AFTER `from_pretrained`: fusion params are absent
        from pretrained checkpoints, so HF flags them as missing keys and
        re-runs `_init_weights` on them, overwriting whatever this method set.
        The model loader (`modeling.py`'s `load_qwen3_asr`) calls this again
        post-load for exactly that reason."""

    def fuse(self, sep_input_features, mix_input_features, sep_feature_lens, mix_feature_lens):
        raise NotImplementedError(f"{type(self).__name__} must implement fuse()")

    @staticmethod
    def _align_mix_len(sep_input_features, mix_input_features, where):
        """Match the mixture mel's time length to the separated mel's, so both
        streams chunk/encode identically. Drift should be fixed upstream (PLAN
        Phase 4); here we degrade gracefully with a warning."""
        t_sep = sep_input_features.shape[-1]
        t_mix = mix_input_features.shape[-1]
        if t_mix == t_sep:
            return mix_input_features
        logger.warning_once(
            f"{where}: sep/mix mel length mismatch ({t_sep} vs {t_mix}); "
            "aligning mixture to the separated length. Investigate upstream."
        )
        if t_mix > t_sep:
            return mix_input_features[:, :t_sep]
        return F.pad(mix_input_features, (0, t_sep - t_mix))

    @staticmethod
    def _warn_token_count_mismatch(where, sep_feat, mix_feat):
        """Both post-tower streams should yield the same number of audio
        tokens (same waveform length -> same mel length -> same tokens). A
        mismatch means upstream alignment drift; degrade gracefully by
        truncating to the shorter length so training does not crash."""
        if sep_feat.shape[0] == mix_feat.shape[0]:
            return sep_feat, mix_feat
        n = min(sep_feat.shape[0], mix_feat.shape[0])
        logger.warning_once(
            f"{where}: sep/mix audio-token count mismatch "
            f"({sep_feat.shape[0]} vs {mix_feat.shape[0]}); truncating to {n}. "
            "Investigate separated/mixture alignment upstream."
        )
        return sep_feat[:n], mix_feat[:n]


class Qwen3ASRAudioEncoderLateGate(Qwen3ASRAudioEncoderSepBase):
    """late_gate: run the pretrained tower on BOTH streams and merge their
    outputs with a per-channel gate,
        fused = sep + sigmoid(fusion_gate(sep) + GATE_INIT_SHIFT) * mix
    Two tower passes. Near-no-op at init (gate = sigmoid(-5) ~= 0.0067).

    GATE_INIT_SHIFT (a fixed constant, NOT a Parameter) is what makes the gate
    start near-zero -- `fusion_gate` itself is zero-initialized (both weight
    AND bias), so this constant is the only thing keeping the initial
    behavior identical to the old "bias = -5.0" approach. That used to be a
    trainable bias sitting at -5.0, which is fatal under native-bf16 training
    (no fp32 master weights, see modeling.py's load_qwen3_asr): bf16's
    representable precision near magnitude 5 is only ~0.03 (the gap between
    -5.0 and the next representable value), while an Adam step at lr=1e-5 is
    ~3 orders of magnitude smaller -- every single optimizer step was a
    complete no-op once applied to that bf16 tensor, so the bias (and
    empirically, the whole gate) never moved off init across a full training
    run, on 1 GPU or many, regardless of how long training ran. bf16 near 0.0
    has no such problem (subnormal range gives far finer precision there),
    so keeping every trainable tensor at/near zero and moving the large
    constant OUTSIDE the optimizer's reach fixes it with no change to the
    initial numerical behavior.

    Note the gate is a function of `sep` alone -- it never looks at `mix` to
    decide how much of it to pull in, and even at gate ~= 1 it only ever ADDS
    mix on top of sep, never replaces a corrupted sep. See
    Qwen3ASRAudioEncoderCrossAttn for a strategy that removes both
    limitations."""

    fusion_type = FUSION_LATE_GATE
    GATE_INIT_SHIFT = -5.0  # sigmoid(-5) ~= 0.0067; see class docstring

    def __init__(self, config):
        super().__init__(config)
        # Per-channel gate on the tower output (output_dim wide).
        self.fusion_gate = nn.Linear(config.output_dim, config.output_dim)
        self.reset_fusion_params()

    def reset_fusion_params(self):
        # Both weight AND bias zeroed -- GATE_INIT_SHIFT (not a Parameter)
        # supplies the near-zero-at-init behavior instead. See class docstring.
        nn.init.zeros_(self.fusion_gate.weight)
        nn.init.zeros_(self.fusion_gate.bias)

    def fuse(self, sep_input_features, mix_input_features, sep_feature_lens, mix_feature_lens):
        """Args match the per-audio slices used in `get_audio_features`: each
        `*_input_features` is a single audio's mel tensor `(num_mel_bins, T)`
        and each `*_feature_lens` is a length-1 LongTensor."""
        sep_out = super().forward(sep_input_features, feature_lens=sep_feature_lens)
        mix_out = super().forward(mix_input_features, feature_lens=mix_feature_lens)
        sep_feat = sep_out.last_hidden_state  # (num_tokens_sep, output_dim)
        mix_feat = mix_out.last_hidden_state  # (num_tokens_mix, output_dim)
        sep_feat, mix_feat = self._warn_token_count_mismatch(
            "Qwen3ASRAudioEncoderLateGate.fuse", sep_feat, mix_feat
        )

        gate = torch.sigmoid(self.fusion_gate(sep_feat) + self.GATE_INIT_SHIFT)
        fused = sep_feat + gate * mix_feat
        return BaseModelOutput(last_hidden_state=fused)


class Qwen3ASRAudioEncoderEarlyConv(Qwen3ASRAudioEncoderSepBase):
    """early_conv: add a parallel first conv on the mixture mel, summed into
    conv2d1's output BEFORE the encoder,
        embed = gelu(conv2d1(sep) + conv2d1_mix(mix))
    Equivalent to widening conv2d1 to 2 input channels, but leaves the
    pretrained conv2d1 untouched. One tower pass. TRUE bit-exact no-op at
    init (conv2d1_mix weight = 0).

    Note this has NO gating at all -- both conv outputs are always summed
    unconditionally, so a badly corrupted stream floods the shared embedding
    at that time step regardless of how clean the other stream is."""

    fusion_type = FUSION_EARLY_CONV

    def __init__(self, config):
        super().__init__(config)
        # Parallel first conv on the mixture stream; mirrors conv2d1 but with
        # a single input channel and no bias (conv2d1 supplies the bias).
        self.conv2d1_mix = nn.Conv2d(
            1, config.downsample_hidden_size, 3, 2, padding=1, bias=False
        )
        self.reset_fusion_params()

    def reset_fusion_params(self):
        # Zero weight => conv2d1_mix contributes nothing at init => this
        # fusion path reproduces the single-stream output bit-for-bit.
        nn.init.zeros_(self.conv2d1_mix.weight)

    def fuse(self, sep_input_features, mix_input_features, sep_feature_lens, mix_feature_lens):
        """Stack the mixture mel as a second conv input channel, then run the
        rest of the tower exactly as upstream. Mirrors the base `forward()`
        chunking. Args match the per-audio slices used in `get_audio_features`."""
        mix_input_features = self._align_mix_len(
            sep_input_features, mix_input_features, "Qwen3ASRAudioEncoderEarlyConv.fuse"
        )
        feature_lens = sep_feature_lens

        # --- identical chunking to the base forward (shared by both streams) ---
        aftercnn_lens = _get_feat_extract_output_lengths(feature_lens)
        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()
        chunk_lengths = torch.tensor(
            [self.n_window * 2] * chunk_num.sum(),
            dtype=torch.long,
            device=feature_lens.device,
        )
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths[chunk_lengths == 0] = self.n_window * 2

        def _to_conv_input(feats):
            chunk_list = feats.T.split(chunk_lengths.tolist(), dim=0)
            padded = nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
            return padded.unsqueeze(1)  # (num_chunks, 1, num_mel_bins, max_chunk_len)

        sep_feature = _to_conv_input(sep_input_features)
        mix_feature = _to_conv_input(mix_input_features)

        feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths)
        padded_mask_after_cnn = nn.utils.rnn.pad_sequence(
            [
                torch.ones(length, dtype=torch.bool, device=sep_feature.device)
                for length in feature_lens_after_cnn
            ],
            batch_first=True,
        )

        # --- the only fusion point: two-channel first conv ---
        padded_embeds = []
        for sep_chunk, mix_chunk in zip(
            sep_feature.split(self.conv_chunksize, dim=0),
            mix_feature.split(self.conv_chunksize, dim=0),
        ):
            padded_embed = F.gelu(self.conv2d1(sep_chunk) + self.conv2d1_mix(mix_chunk))
            padded_embed = F.gelu(self.conv2d2(padded_embed))
            padded_embed = F.gelu(self.conv2d3(padded_embed))
            padded_embeds.append(padded_embed)
        padded_embed = torch.cat(padded_embeds, dim=0)
        b, c, f, t = padded_embed.size()
        padded_embed = self.conv_out(padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))

        positional_embedding = (
            self.positional_embedding.positional_embedding[: padded_embed.shape[1], :]
            .unsqueeze(0)
            .to(padded_embed.dtype)
        )
        padded_embed = padded_embed + positional_embedding
        hidden_states = padded_embed[padded_mask_after_cnn]
        cu_chunk_lens = [0]
        window_aftercnn = padded_mask_after_cnn.shape[-1] * (self.n_window_infer // (self.n_window * 2))
        for cnn_len in aftercnn_lens:
            cu_chunk_lens += [window_aftercnn] * (cnn_len // window_aftercnn)
            remainder = cnn_len % window_aftercnn
            if remainder != 0:
                cu_chunk_lens += [remainder]
        cu_seqlens = torch.tensor(cu_chunk_lens, device=aftercnn_lens.device).cumsum(-1, dtype=torch.int32)

        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(hidden_states, cu_seqlens)
            hidden_states = layer_outputs[0]

        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.proj2(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states)


class _FDDTLayerGates(nn.Module):
    """Container for FDDT's per-layer gates (one small Linear per transformer
    layer). This is a real nn.Module wrapping an nn.ModuleList, NOT a bare
    nn.ModuleList as the class attribute itself -- PEFT's `modules_to_save`
    explicitly refuses to wrap ModuleList/ModuleDict (see
    `peft.utils.other.AuxiliaryTrainingWrapper.check_module`), and its
    wrapper only proxies `forward(x, *args, **kwargs)`, not `__getitem__` --
    so layer selection has to happen INSIDE this module's own forward rather
    than via external indexing, or a LoRA run with fusion_type=fddt would
    fail as soon as PEFT wraps this attribute.

    GATE_INIT_SHIFT is a fixed constant (not a Parameter), added to each
    gate's raw logit here so every gate starts near-closed (sigmoid(-5) ~=
    0.0067) while weight AND bias both stay zero-initialized. A trainable
    bias sitting at a large magnitude like -5.0 is invisible to bf16
    optimizer updates (bf16's precision near magnitude 5 is ~0.03, far
    coarser than a typical Adam step) -- see
    Qwen3ASRAudioEncoderLateGate's docstring, which hit exactly this bug:
    that gate never moved off its -5.0-initialized bias across a full
    training run. These per-layer gates were initialized the same way and
    would have the identical failure mode, just less visibly since
    FDDT is a TRUE no-op at init regardless of gate value (film_proj is
    zeroed) -- but frozen-forever gates would just as thoroughly break
    training here too."""

    GATE_INIT_SHIFT = -5.0  # sigmoid(-5) ~= 0.0067

    def __init__(self, d_model, num_layers):
        super().__init__()
        self.gates = nn.ModuleList(nn.Linear(2 * d_model, 1) for _ in range(num_layers))

    def forward(self, hidden_states, cond, layer_idx):
        return self.gates[layer_idx](torch.cat([hidden_states, cond], dim=-1)) + self.GATE_INIT_SHIFT

    def reset(self):
        for gate in self.gates:
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)


class Qwen3ASRAudioEncoderFDDT(Qwen3ASRAudioEncoderSepBase):
    """fddt: frame-level diarization-dependent-transform-style conditioning,
    inspired by DiCoW (Polok et al., "DiCoW: Diarization-Conditioned Whisper
    for Target Speaker ASR"). Unlike late_gate/early_conv, which run BOTH
    streams as symmetric peers and merge them at one point (the very start or
    the very end), fddt treats `mix` as the PRIMARY signal (always complete,
    not dependent on separation quality) and `sep` as a CONDITION that steers
    how `mix` gets encoded, injected at EVERY transformer layer rather than
    once:

        cond        = conv_stem(sep)                        # cheap: stem only, no transformer layers
        modulation  = film_proj(cond_proj(cond))             # shared across layers, computed once
        for each layer l:
            gate_t^l = sigmoid(layer_gate_l([h_t^l ; cond_t]))   # content-dependent, own gate per layer
            h_t^l    = h_t^l + gate_t^l * modulation_t
            h^l      = TransformerLayer_l(h^l)                   # unmodified pretrained layer

    `sep` only ever passes through the (shared, pretrained) conv stem -- never
    through the 24-layer transformer stack -- so this costs one full tower
    pass (on `mix`) plus a cheap conv-only side pass (on `sep`), similar to
    early_conv's cost, while giving the model per-layer (not single-point)
    opportunities to pull in `sep` information, similar to late_gate's
    per-token gate but repeated at depth.

    TRUE bit-exact no-op at init: `film_proj` is zeroed, so `modulation` is
    identically zero regardless of `cond` or the (also near-zero-biased)
    gates -- the primary `mix` path is untouched at init no matter what."""

    fusion_type = FUSION_FDDT

    def __init__(self, config):
        super().__init__(config)
        self.cond_proj = nn.Linear(config.d_model, config.d_model)
        self.film_proj = nn.Linear(config.d_model, config.d_model)
        self.layer_gates = _FDDTLayerGates(config.d_model, config.encoder_layers)
        self.reset_fusion_params()

    def reset_fusion_params(self):
        # film_proj zeroed => modulation is exactly zero at init => this
        # fusion path reproduces the single-stream (mix-only) output
        # bit-for-bit, regardless of gate values.
        nn.init.zeros_(self.film_proj.weight)
        nn.init.zeros_(self.film_proj.bias)
        # Belt-and-suspenders: also start every per-layer gate closed
        # (sigmoid(-5) ~= 0.0067), so training starts from "barely open" even
        # if film_proj ever moves before the gates learn to use it.
        self.layer_gates.reset()

    def _conv_stem(self, feats, chunk_lengths):
        """Run only the shared, pretrained conv stem (conv2d1/2/3 + conv_out)
        on one stream's mel features -- no transformer layers. Used for BOTH
        the primary (`mix`) and condition (`sep`) streams so their per-frame
        stem outputs live in the same space and stay time-aligned."""
        chunk_list = feats.T.split(chunk_lengths.tolist(), dim=0)
        padded = nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
        conv_input = padded.unsqueeze(1)  # (num_chunks, 1, num_mel_bins, max_chunk_len)

        embeds = []
        for chunk in conv_input.split(self.conv_chunksize, dim=0):
            e = F.gelu(self.conv2d1(chunk))
            e = F.gelu(self.conv2d2(e))
            e = F.gelu(self.conv2d3(e))
            embeds.append(e)
        embed = torch.cat(embeds, dim=0)
        b, c, f, t = embed.size()
        return self.conv_out(embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))

    def fuse(self, sep_input_features, mix_input_features, sep_feature_lens, mix_feature_lens):
        """`mix` is the primary path (runs the full transformer stack, FiLM-
        modulated per layer by `sep`); `sep` only runs the conv stem. Args
        match the per-audio slices used in `get_audio_features`."""
        mix_input_features = self._align_mix_len(
            sep_input_features, mix_input_features, "Qwen3ASRAudioEncoderFDDT.fuse"
        )
        feature_lens = sep_feature_lens

        # --- identical chunking to the base forward (shared by both streams) ---
        aftercnn_lens = _get_feat_extract_output_lengths(feature_lens)
        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()
        chunk_lengths = torch.tensor(
            [self.n_window * 2] * chunk_num.sum(),
            dtype=torch.long,
            device=feature_lens.device,
        )
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths[chunk_lengths == 0] = self.n_window * 2

        mix_embed = self._conv_stem(mix_input_features, chunk_lengths)
        cond_embed = self._conv_stem(sep_input_features, chunk_lengths)

        feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths)
        padded_mask_after_cnn = nn.utils.rnn.pad_sequence(
            [
                torch.ones(length, dtype=torch.bool, device=mix_embed.device)
                for length in feature_lens_after_cnn
            ],
            batch_first=True,
        )

        positional_embedding = (
            self.positional_embedding.positional_embedding[: mix_embed.shape[1], :]
            .unsqueeze(0)
            .to(mix_embed.dtype)
        )
        mix_embed = mix_embed + positional_embedding
        cond_embed = cond_embed + positional_embedding

        hidden_states = mix_embed[padded_mask_after_cnn]
        cond = cond_embed[padded_mask_after_cnn]

        cu_chunk_lens = [0]
        window_aftercnn = padded_mask_after_cnn.shape[-1] * (self.n_window_infer // (self.n_window * 2))
        for cnn_len in aftercnn_lens:
            cu_chunk_lens += [window_aftercnn] * (cnn_len // window_aftercnn)
            remainder = cnn_len % window_aftercnn
            if remainder != 0:
                cu_chunk_lens += [remainder]
        cu_seqlens = torch.tensor(cu_chunk_lens, device=aftercnn_lens.device).cumsum(-1, dtype=torch.int32)

        cond = self.cond_proj(cond)
        modulation = self.film_proj(cond)  # shared across layers, computed once

        for i, encoder_layer in enumerate(self.layers):
            gate = torch.sigmoid(self.layer_gates(hidden_states, cond, i))
            hidden_states = hidden_states + gate * modulation
            layer_outputs = encoder_layer(hidden_states, cu_seqlens)
            hidden_states = layer_outputs[0]

        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.proj2(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states)


class _LocalCrossAttention(nn.Module):
    """Windowed multi-head cross-attention: each query position attends only
    to key/value positions within `window` frames of it. sep_feat/mix_feat
    are time-aligned (same underlying audio encoded twice), so the useful
    information from the other stream for frame t is expected to live near
    frame t, not anywhere in the whole utterance -- a full T x T attention
    would be both more expensive and a weaker inductive bias here."""

    def __init__(self, d_model, num_heads, window):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window = window
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query, kv):
        """query, kv: (T, d_model) -- a single utterance (the tower already
        processes one audio at a time; see get_audio_features)."""
        t = query.shape[0]
        q = self.q_proj(query).view(t, self.num_heads, self.head_dim).transpose(0, 1)
        k = self.k_proj(kv).view(t, self.num_heads, self.head_dim).transpose(0, 1)
        v = self.v_proj(kv).view(t, self.num_heads, self.head_dim).transpose(0, 1)

        idx = torch.arange(t, device=query.device)
        band = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() <= self.window
        attn_mask = torch.zeros(t, t, device=query.device, dtype=q.dtype)
        attn_mask = attn_mask.masked_fill(~band, torch.finfo(q.dtype).min)

        attn_out = F.scaled_dot_product_attention(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), attn_mask=attn_mask
        )
        attn_out = attn_out.squeeze(0).transpose(0, 1).reshape(t, -1)
        return self.out_proj(attn_out)


class Qwen3ASRAudioEncoderCrossAttn(Qwen3ASRAudioEncoderSepBase):
    """cross_attn: run the pretrained tower on BOTH streams (as late_gate
    does), then replace the linear gate with local windowed cross-attention
    so the model can compare sep_feat/mix_feat per-frame and decide what to
    pull from mix, instead of a gate that only ever reads sep_feat in
    isolation:
        fused = sep + CrossAttn(query=sep, key=mix, value=mix)
    Two tower passes, same cost as late_gate. TRUE bit-exact no-op at init
    (the attention module's out_proj weight/bias are zeroed, so its output is
    exactly zero regardless of the attention weights)."""

    fusion_type = FUSION_CROSS_ATTN

    def __init__(self, config):
        super().__init__(config)
        num_heads = getattr(config, "cross_attn_heads", 8)
        window = getattr(config, "cross_attn_window", 50)
        self.cross_attn = _LocalCrossAttention(config.output_dim, num_heads, window)
        self.reset_fusion_params()

    def reset_fusion_params(self):
        nn.init.zeros_(self.cross_attn.out_proj.weight)
        nn.init.zeros_(self.cross_attn.out_proj.bias)

    def fuse(self, sep_input_features, mix_input_features, sep_feature_lens, mix_feature_lens):
        """Args match the per-audio slices used in `get_audio_features`: each
        `*_input_features` is a single audio's mel tensor `(num_mel_bins, T)`
        and each `*_feature_lens` is a length-1 LongTensor."""
        sep_out = super().forward(sep_input_features, feature_lens=sep_feature_lens)
        mix_out = super().forward(mix_input_features, feature_lens=mix_feature_lens)
        sep_feat = sep_out.last_hidden_state
        mix_feat = mix_out.last_hidden_state
        sep_feat, mix_feat = self._warn_token_count_mismatch(
            "Qwen3ASRAudioEncoderCrossAttn.fuse", sep_feat, mix_feat
        )

        # Positional, not query=/kv= keywords: PEFT's ModulesToSaveWrapper
        # (when "fusion" is ft_mode="full" alongside some OTHER lora region,
        # see modeling.py's apply_train_mode) proxies forward(self, x, *args,
        # **kwargs) -- an all-keyword call has no positional arg for it to
        # bind `x` to and raises TypeError regardless of the keyword names.
        # Same constraint _FDDTLayerGates.forward's call site works around.
        fused = sep_feat + self.cross_attn(sep_feat, mix_feat)
        return BaseModelOutput(last_hidden_state=fused)


FUSION_ENCODER_CLASSES = {
    FUSION_LATE_GATE: Qwen3ASRAudioEncoderLateGate,
    FUSION_EARLY_CONV: Qwen3ASRAudioEncoderEarlyConv,
    FUSION_FDDT: Qwen3ASRAudioEncoderFDDT,
    FUSION_CROSS_ATTN: Qwen3ASRAudioEncoderCrossAttn,
}


def _build_audio_tower(audio_config):
    """Look up the fusion-strategy class by `audio_config.fusion_type` and
    construct it. Adding a new fusion type means adding one class above and
    one entry here -- no other class's code needs to change."""
    fusion_type = getattr(audio_config, "fusion_type", FUSION_LATE_GATE)
    try:
        encoder_cls = FUSION_ENCODER_CLASSES[fusion_type]
    except KeyError:
        raise ValueError(f"fusion_type must be one of {FUSION_TYPES}, got {fusion_type!r}")
    return encoder_cls._from_config(audio_config)


class Qwen3ASRThinkerForConditionalGenerationSep(Qwen3ASRThinkerForConditionalGeneration):
    """Thinker that can consume an optional mixture stream and fuse it in the
    audio tower before the LLM. Falls back to the exact single-stream path when
    no mixture features are provided."""

    def __init__(self, config):
        super().__init__(config)
        # Swap in the fusion-capable tower. `from_pretrained` runs after
        # construction and loads the pretrained tower weights into it; the new
        # fusion params keep their __init__ values until the loader calls
        # `reset_fusion_params()` again post-load.
        self.audio_tower = _build_audio_tower(config.audio_config)

    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        feature_attention_mask: Optional[torch.LongTensor] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
        mix_input_features: Optional[torch.FloatTensor] = None,
        mix_feature_attention_mask: Optional[torch.LongTensor] = None,
    ):
        # No mixture stream -> identical to the pretrained single-stream path.
        if mix_input_features is None:
            return super().get_audio_features(
                input_features,
                feature_attention_mask=feature_attention_mask,
                audio_feature_lengths=audio_feature_lengths,
            )

        if feature_attention_mask is not None:
            sep_lens = torch.sum(feature_attention_mask, dim=1)
        else:
            sep_lens = audio_feature_lengths
        mix_lens = torch.sum(mix_feature_attention_mask, dim=1)

        # Audio encoder runs one audio at a time to keep precision (mirrors the
        # base implementation), fusing the two streams per audio.
        audio_features = []
        for sep_f, sep_l, mix_f, mix_l in zip(
            input_features, sep_lens, mix_input_features, mix_lens
        ):
            fused_output = self.audio_tower.fuse(
                sep_f[:, :sep_l],
                mix_f[:, :mix_l],
                sep_l.unsqueeze(0),
                mix_l.unsqueeze(0),
            )
            audio_features.append(fused_output.last_hidden_state)
        return torch.cat(audio_features, dim=0)

    def forward(
        self,
        input_ids=None,
        input_features=None,
        attention_mask=None,
        feature_attention_mask=None,
        audio_feature_lengths=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        rope_deltas=None,
        labels=None,
        use_cache=None,
        cache_position=None,
        mix_input_features=None,
        mix_feature_attention_mask=None,
        **kwargs,
    ) -> Union[tuple, Qwen3ASRThinkerCausalLMOutputWithPast]:
        """Copy of the upstream thinker forward with two optional kwargs plumbed
        into `get_audio_features`. Everything else (rope, decoder, LM head, loss)
        is byte-for-byte the upstream behaviour."""
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # Merge text, audios (fusing the mixture stream when present).
        if input_features is not None:
            audio_features = self.get_audio_features(
                input_features,
                feature_attention_mask=feature_attention_mask,
                audio_feature_lengths=audio_feature_lengths,
                mix_input_features=mix_input_features,
                mix_feature_attention_mask=mix_feature_attention_mask,
            )
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            audio_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        else:
            audio_feature_lengths = None

        if attention_mask is not None and position_ids is None:
            if (
                cache_position is None
                or (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
            ):
                delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                position_ids, rope_deltas = self.get_rope_index(attention_mask)
                rope_deltas = rope_deltas - delta0
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length = input_ids.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.get_text_config().vocab_size
            )

        return Qwen3ASRThinkerCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        input_features=None,
        feature_attention_mask=None,
        mix_input_features=None,
        mix_feature_attention_mask=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            **kwargs,
        )
        # Only feed the mixture stream on the prefill step, mirroring how the
        # base class drops input_features after the first decode step.
        if cache_position is not None and cache_position[0] == 0:
            model_inputs["mix_input_features"] = mix_input_features
            model_inputs["mix_feature_attention_mask"] = mix_feature_attention_mask
        else:
            model_inputs["mix_input_features"] = None
            model_inputs["mix_feature_attention_mask"] = None
        return model_inputs


class Qwen3ASRForConditionalGenerationSep(Qwen3ASRForConditionalGeneration):
    """Top-level wrapper that builds the fusion-capable thinker. Loads pretrained
    weights via `from_pretrained` exactly like upstream; the fusion params are
    the only new (missing-from-checkpoint) parameters."""

    def __init__(self, config):
        # Skip the base __init__ (which builds the non-Sep thinker) and go
        # straight to PreTrainedModel init, then build the Sep thinker.
        super(Qwen3ASRForConditionalGeneration, self).__init__(config)
        self.config = config
        self.thinker = Qwen3ASRThinkerForConditionalGenerationSep._from_config(config.thinker_config)
        self.post_init()


__all__ = [
    "Qwen3ASRAudioEncoderSepBase",
    "Qwen3ASRAudioEncoderLateGate",
    "Qwen3ASRAudioEncoderEarlyConv",
    "Qwen3ASRAudioEncoderFDDT",
    "Qwen3ASRAudioEncoderCrossAttn",
    "Qwen3ASRThinkerForConditionalGenerationSep",
    "Qwen3ASRForConditionalGenerationSep",
    "FUSION_LATE_GATE",
    "FUSION_EARLY_CONV",
    "FUSION_FDDT",
    "FUSION_CROSS_ATTN",
    "FUSION_TYPES",
    "FUSION_ENCODER_CLASSES",
]
