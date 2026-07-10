# coding=utf-8
# Mega-ASR-SEP: mixture-conditioned fusion on top of Qwen3-ASR.
#
# This module DOES NOT edit the upstream model. It vendors the pristine upstream
# code in `_modeling_qwen3_asr_base.py` (verbatim diff base, used for regression
# checks) and adds thin `*Sep` subclasses here. All fusion-specific logic lives
# in this file so the diff against upstream is isolated and reviewable.
#
# Fusion mechanism (initialised to a no-op):
#     gate  = sigmoid(fusion_gate(sep_feat))    # ~0 at init (bias = -5)
#     fused = sep_feat + gate * mix_feat        # ~= sep_feat at init
#
# Everything downstream of the audio tower (placeholder scatter, LLM decoder,
# LM head, loss) is untouched: one audio placeholder span, same length, same
# position. The LLM's input contract is unchanged.
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
FUSION_LATE_GATE = "late_gate"   # gated residual on the tower OUTPUT (2 tower passes)
FUSION_EARLY_CONV = "early_conv"  # parallel conv on the mixture mel INPUT (1 tower pass)
FUSION_TYPES = (FUSION_LATE_GATE, FUSION_EARLY_CONV)

logger = logging.get_logger(__name__)


class Qwen3ASRAudioEncoderSep(Qwen3ASRAudioEncoder):
    """Audio tower with mixture-conditioned fusion. Two strategies, selected by
    `config.fusion_type`:

    * "late_gate"  — `forward_fused`: run the pretrained tower on BOTH streams
      and merge their outputs with a per-channel gate,
          fused = sep + sigmoid(fusion_gate(sep)) * mix
      Two tower passes. Near-no-op at init (gate = sigmoid(-5) ~= 0.0067).

    * "early_conv" — `forward_earlyfuse`: add a parallel first conv on the
      mixture mel, summed into conv2d1's output BEFORE the encoder,
          embed = gelu(conv2d1(sep) + conv2d1_mix(mix))
      Equivalent to widening conv2d1 to 2 input channels, but leaves the
      pretrained conv2d1 untouched. One tower pass. TRUE bit-exact no-op at
      init (conv2d1_mix weight = 0).

    The base `forward()` (single-stream) is inherited unchanged, so a baseline
    run is numerically identical to upstream.
    """

    def __init__(self, config):
        super().__init__(config)
        self.use_fusion = getattr(config, "use_fusion", False)
        self.fusion_type = getattr(config, "fusion_type", FUSION_LATE_GATE)
        if self.fusion_type not in FUSION_TYPES:
            raise ValueError(
                f"fusion_type must be one of {FUSION_TYPES}, got {self.fusion_type!r}"
            )

        if self.fusion_type == FUSION_LATE_GATE:
            # Per-channel gate on the tower output (output_dim wide).
            self.fusion_gate = nn.Linear(config.output_dim, config.output_dim)
        elif self.fusion_type == FUSION_EARLY_CONV:
            # Parallel first conv on the mixture stream; mirrors conv2d1 but with
            # a single input channel and no bias (conv2d1 supplies the bias).
            self.conv2d1_mix = nn.Conv2d(
                1, config.downsample_hidden_size, 3, 2, padding=1, bias=False
            )
        self.reset_fusion_params()

    def reset_fusion_params(self):
        """(Re)initialise the fusion parameters to a (near-)no-op.

        MUST be re-applied AFTER `from_pretrained`: the fusion params are absent
        from pretrained checkpoints, so HF flags them as missing keys and re-runs
        `_init_weights` on them, overwriting these values (e.g. fusion_gate bias
        -> 0 => gate = 0.5). The model loader calls this again post-load.
        """
        if hasattr(self, "fusion_gate"):
            self.reset_fusion_gate()
        if hasattr(self, "conv2d1_mix"):
            # Zero weight => conv2d1_mix contributes nothing at init => the early
            # fusion path reproduces the single-stream output bit-for-bit.
            nn.init.zeros_(self.conv2d1_mix.weight)

    def reset_fusion_gate(self):
        nn.init.zeros_(self.fusion_gate.weight)
        nn.init.constant_(self.fusion_gate.bias, -5.0)  # sigmoid(-5) ~= 0.0067

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

    def forward_earlyfuse(
        self,
        sep_input_features,
        mix_input_features,
        sep_feature_lens,
        mix_feature_lens,
    ):
        """Early fusion: stack the mixture mel as a second conv input channel
        (implemented as `conv2d1(sep) + conv2d1_mix(mix)`), then run the rest of
        the tower exactly as upstream. Mirrors the base `forward()` chunking.

        Args match the per-audio slices used in `get_audio_features`.
        """
        mix_input_features = self._align_mix_len(
            sep_input_features, mix_input_features, "forward_earlyfuse"
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

    def fuse(self, sep_input_features, mix_input_features, sep_feature_lens, mix_feature_lens):
        """Dispatch to the configured fusion strategy."""
        if self.fusion_type == FUSION_EARLY_CONV:
            return self.forward_earlyfuse(
                sep_input_features, mix_input_features, sep_feature_lens, mix_feature_lens
            )
        return self.forward_fused(
            sep_input_features, mix_input_features, sep_feature_lens, mix_feature_lens
        )

    def forward_fused(
        self,
        sep_input_features,
        mix_input_features,
        sep_feature_lens,
        mix_feature_lens,
    ):
        """Encode both streams with the pretrained tower and gate-fuse them.

        Args match the per-audio slices used in `get_audio_features`: each
        `*_input_features` is a single audio's mel tensor `(num_mel_bins, T)`
        and each `*_feature_lens` is a length-1 LongTensor.
        """
        sep_out = super().forward(sep_input_features, feature_lens=sep_feature_lens)
        mix_out = super().forward(mix_input_features, feature_lens=mix_feature_lens)
        sep_feat = sep_out.last_hidden_state  # (num_tokens_sep, output_dim)
        mix_feat = mix_out.last_hidden_state  # (num_tokens_mix, output_dim)

        if sep_feat.shape[0] != mix_feat.shape[0]:
            # Separated and mixture streams should yield the same number of audio
            # tokens (same waveform length -> same mel length -> same tokens).
            # A mismatch means upstream alignment drift; fix it in preprocessing
            # (see PLAN Phase 4). Here we degrade gracefully: truncate to the
            # shorter length so training does not crash.
            n = min(sep_feat.shape[0], mix_feat.shape[0])
            logger.warning_once(
                "forward_fused: sep/mix audio-token count mismatch "
                f"({sep_feat.shape[0]} vs {mix_feat.shape[0]}); truncating to {n}. "
                "Investigate separated/mixture alignment upstream."
            )
            sep_feat = sep_feat[:n]
            mix_feat = mix_feat[:n]

        gate = torch.sigmoid(self.fusion_gate(sep_feat))
        fused = sep_feat + gate * mix_feat
        return BaseModelOutput(last_hidden_state=fused)


class Qwen3ASRThinkerForConditionalGenerationSep(Qwen3ASRThinkerForConditionalGeneration):
    """Thinker that can consume an optional mixture stream and fuse it in the
    audio tower before the LLM. Falls back to the exact single-stream path when
    no mixture features are provided."""

    def __init__(self, config):
        super().__init__(config)
        # Swap in the fusion-capable tower. `from_pretrained` runs after
        # construction and loads the pretrained tower weights into it; the new
        # fusion params (fusion_gate or conv2d1_mix) keep their __init__ values
        # until the loader calls `reset_fusion_params()` again post-load.
        self.audio_tower = Qwen3ASRAudioEncoderSep._from_config(config.audio_config)

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
    weights via `from_pretrained` exactly like upstream; `fusion_gate` is the only
    new (missing-from-checkpoint) parameter."""

    def __init__(self, config):
        # Skip the base __init__ (which builds the non-Sep thinker) and go
        # straight to PreTrainedModel init, then build the Sep thinker.
        super(Qwen3ASRForConditionalGeneration, self).__init__(config)
        self.config = config
        self.thinker = Qwen3ASRThinkerForConditionalGenerationSep._from_config(config.thinker_config)
        self.post_init()


__all__ = [
    "Qwen3ASRAudioEncoderSep",
    "Qwen3ASRThinkerForConditionalGenerationSep",
    "Qwen3ASRForConditionalGenerationSep",
]
