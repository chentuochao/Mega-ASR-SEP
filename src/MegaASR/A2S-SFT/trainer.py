# coding=utf-8
import os
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import Trainer

import action_format
import modeling


class MegaASRTrainer(Trainer):
    """Trainer for Mega-ASR LoRA SFT."""

    def __init__(self, *args, processor=None, base_model_path: str = "",
                 merged_from_lora_path: str = "", lr_encoder: float = 1e-5,
                 lr_aligner: float = 1e-5, lr_llm: float = 1e-5, lr_fusion: float = 1e-5,
                 fusion_type: str = "late_gate", use_fusion: bool = False,
                 action_loss_weight: float = 1.0,
                 action_format_name: str = action_format.ACTION_FIRST,
                 action_token_lr: float = 1e-5, action_metadata: Optional[dict] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = processor
        self.base_model_path = base_model_path
        self.merged_from_lora_path = merged_from_lora_path
        self.lr_encoder = lr_encoder
        self.lr_aligner = lr_aligner
        self.lr_llm = lr_llm
        self.lr_fusion = lr_fusion
        if action_loss_weight <= 0:
            raise ValueError(f"action_loss_weight must be positive, got {action_loss_weight}")
        self.action_loss_weight = action_loss_weight
        self.action_format = action_format.validate_format(action_format_name)
        self.action_token_lr = action_token_lr
        # Store the configuration needed to reconstruct action checkpoints.
        self.action_metadata = dict(action_metadata or {})
        self.protocol_token_ids = None
        if action_format.is_special(self.action_format):
            self.protocol_token_ids = action_format.resolve_protocol_token_ids(processor.tokenizer)
        # Needed by classify_param_region to recognize this fusion_type's
        # own param names (they differ per type -- see modeling._FUSION_MODULE_NAMES).
        self._fusion_module_names = list(modeling._FUSION_MODULE_NAMES[fusion_type]) if use_fusion else []

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Select the loss for the batch's ASR or action protocol.

        Judge batches use weighted action cross-entropy at the query position.
        action_first batches partition causal-token loss into action and
        transcript spans. action_special_v1 uses separate decoder-pass losses.
        ASR batches without action masks use the Transformers Trainer loss.
        """
        if getattr(self, "action_format", None) == action_format.ACTION_JUDGE_V1:
            inputs = dict(inputs)
            labels = inputs.pop("labels")
            weights = inputs.pop("action_sample_weight").float()
            allowed = [self.protocol_token_ids[a] for a in action_format.ACTIONS]
            if not torch.isin(labels, torch.tensor(allowed, device=labels.device)).all():
                raise ValueError("Judge targets must be RESPOND or REJECT token IDs")
            outputs = model(**inputs, judge_only=True)
            ce = F.cross_entropy(outputs["logits"][:, -1].float(), labels, reduction="none")
            loss = (ce * weights).sum() / weights.sum()
            self.last_action_loss = loss.detach()
            return (loss, outputs) if return_outputs else loss
        if "action_input_ids" in inputs:
            return self._two_pass_loss(model, inputs, return_outputs)
        if "action_loss_mask" not in inputs:
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        inputs = dict(inputs)
        action_mask = inputs.pop("action_loss_mask")
        transcript_mask = inputs.pop("transcript_loss_mask")
        sample_weight = inputs.pop("action_sample_weight", None)
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_action_mask = action_mask[:, 1:].bool()
        shift_transcript_mask = transcript_mask[:, 1:].bool()
        valid = shift_labels.ne(-100)
        shift_action_mask &= valid
        shift_transcript_mask &= valid
        if not torch.any(shift_action_mask):
            raise ValueError("Action batch contains no active action tokens after causal shift")
        if not torch.any(shift_transcript_mask):
            raise ValueError("Action batch contains no active transcript tokens after causal shift")

        token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view_as(shift_labels)

        if sample_weight is None:
            action_weights = shift_action_mask.to(token_loss.dtype)
        else:
            action_weights = (
                shift_action_mask.to(token_loss.dtype)
                * sample_weight.to(token_loss.dtype).view(-1, 1)
            )
        action_loss = (token_loss * action_weights).sum() / action_weights.sum().clamp_min(1.0)
        transcript_weights = shift_transcript_mask.to(token_loss.dtype)
        transcript_loss = (
            (token_loss * transcript_weights).sum()
            / transcript_weights.sum().clamp_min(1.0)
        )
        loss = transcript_loss + self.action_loss_weight * action_loss

        # Retain component losses without synchronizing device tensors to the CPU.
        self.last_action_loss = action_loss.detach()
        self.last_transcript_loss = transcript_loss.detach()
        return (loss, outputs) if return_outputs else loss

    def _two_pass_loss(self, model, inputs, return_outputs):
        """action_special_v1:
            L_transcript = mean CE over the transcript pass's supervised tokens
                           (transcript + EOS; prompt and padding ignored)
            L_action     = sum_i w_i * CE(action_logits_i, action_i) / sum_i w_i
                           at the query position, over the FULL expanded vocab
            L_total      = L_transcript + action_loss_weight * L_action
        The model forward runs both decoder passes (modeling.two_pass_forward).
        """
        inputs = dict(inputs)
        labels = inputs.pop("labels")
        action_labels = inputs.pop("action_labels")
        sample_weight = inputs.pop("action_sample_weight", None)
        protocol_ids = getattr(self, "protocol_token_ids", None)
        if protocol_ids is not None:
            allowed = torch.tensor([protocol_ids[a] for a in action_format.ACTIONS],
                                   device=action_labels.device)
            if not torch.isin(action_labels, allowed).all():
                raise ValueError("An action-pass target is not an action token ID")
        outputs = model(**inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
        action_logits = outputs["action_logits"]
        if action_logits.shape[0] != action_labels.shape[0]:
            raise ValueError("Action pass must produce exactly one prediction per sample")

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        transcript_mask = shift_labels.ne(-100)
        if not torch.all(transcript_mask.any(dim=1)):
            raise ValueError("Every sample needs transcript supervision (at least EOS)")
        token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view_as(shift_labels)
        transcript_weights = transcript_mask.to(token_loss.dtype)
        transcript_loss = (token_loss * transcript_weights).sum() / transcript_weights.sum()

        action_ce = F.cross_entropy(action_logits.float(), action_labels, reduction="none")
        weights = (torch.ones_like(action_ce) if sample_weight is None
                   else sample_weight.to(action_ce.dtype))
        action_loss = (action_ce * weights).sum() / weights.sum()
        loss = transcript_loss + self.action_loss_weight * action_loss.to(transcript_loss.dtype)

        self.last_action_loss = action_loss.detach()
        self.last_transcript_loss = transcript_loss.detach()
        return (loss, outputs) if return_outputs else loss

    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        dtype = getattr(self.model, "dtype", None)
        if dtype is None:
            return inputs
        for k, v in list(inputs.items()):
            if torch.is_tensor(v) and v.is_floating_point():
                inputs[k] = v.to(dtype=dtype)
        return inputs

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        # This override bypasses Trainer.save_model's distributed write guard.
        # Only the saving rank may write shared adapter/tokenizer artifacts.
        if not self.args.should_save:
            return
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if hasattr(self.model.thinker, "peft_config"):
            # LoRA: PeftModel.save_pretrained writes adapter_model.safetensors
            # keyed relative to the thinker (base_model.model.*), which is what
            # PeftModel.from_pretrained expects when re-applying the adapter
            # on top of a separately-loaded base model.
            self.model.thinker.save_pretrained(output_dir, safe_serialization=True)
        else:
            # Full fine-tune: save the TOP-LEVEL wrapper, not just `.thinker`.
            # `model.thinker.state_dict()` keys have no "thinker." prefix, but
            # the copied config.json (below, via MakeCheckpointInferableCallback)
            # describes the wrapper class (Qwen3ASRForConditionalGeneration),
            # whose from_pretrained expects "thinker.*"-prefixed keys. Saving
            # only `.thinker` here would silently produce a checkpoint where
            # every key mismatches on reload -- from_pretrained then
            # re-initializes ~everything instead of raising, so this fails
            # silently rather than loudly. Save the wrapper so keys line up.
            self.model.save_pretrained(output_dir, safe_serialization=True)

        if self.processor is not None:
            self.processor.save_pretrained(output_dir)
        self._write_text(output_dir, "base_model.txt", self.base_model_path)
        self._write_text(output_dir, "merged_from_lora.txt", self.merged_from_lora_path)
        if action_format.is_special(self.action_format) and self.args.should_save:
            self._write_action_checkpoint(output_dir)

        if hasattr(self.model.thinker, "peft_config"):
            # LoRA adapter checkpoint: save_pretrained already wrote
            # adapter_model.safetensors. Strip any full-model files that may be
            # left over in this output_dir from a previous full-fine-tune run
            # (or an old checkpoint layout) -- NOT run for a full fine-tune
            # itself, where these files ARE the actual trained weights.
            for name in ["model.safetensors", "pytorch_model.bin",
                         "model.safetensors.index.json", "pytorch_model.bin.index.json"]:
                path = os.path.join(output_dir, name)
                if os.path.exists(path):
                    os.remove(path)

    def _write_action_checkpoint(self, output_dir: str):
        """Save protocol rows and reconstruction metadata for action checkpoints."""
        if not hasattr(self.model.thinker, "peft_config"):
            raise ValueError("action_special_v1 checkpoints must be PEFT adapter checkpoints")
        rows = modeling.get_action_rows(self.model)
        # The saved processor must reproduce the in-memory mapping exactly.
        action_format.verify_token_mapping(
            self.processor.tokenizer, self._action_tokens_block(rows)
        )
        modeling.save_action_rows(self.model, output_dir)
        metadata = dict(self.action_metadata)
        metadata.update({
            "metadata_version": action_format.METADATA_VERSION,
            "action_format": action_format.ACTION_SPECIAL_V1,
            "protocol_tokens": self._action_tokens_block(rows),
            "layout": "transcript pass [prompt]->transcript<EOS>; action pass "
                      "[prompt]<|action_query|>->action token",
            "vocabulary": dict(rows.info),
            "rows_file": action_format.ROWS_FILENAME,
            "materialized": False,
            "action_token_lr": self.action_token_lr,
            "action_token_weight_decay": 0.0,
            "action_loss_weight": self.action_loss_weight,
            "base_model_path": self.base_model_path,
            "production_adapter_path": self.merged_from_lora_path or None,
            "global_step": self.state.global_step,
        })
        if self.action_format == action_format.ACTION_JUDGE_V1:
            metadata.update(metadata_version=3, action_format=self.action_format,
                            layout="[audio prompt] generated_transcription<EOS><|action_query|> -> action",
                            transcription_mode="production_only", candidate_field="generated_transcription")
        action_format.write_checkpoint_metadata(output_dir, metadata)

    @staticmethod
    def _action_tokens_block(rows):
        return {
            role: {"token": action_format.PROTOCOL_TOKENS[role], "id": token_id}
            for role, token_id in zip(action_format.PROTOCOL_ROLES, rows.token_id_list)
        }

    @staticmethod
    def _write_text(output_dir: str, name: str, text: str):
        if text:
            with open(os.path.join(output_dir, name), "w", encoding="utf-8") as f:
                f.write(text + "\n")

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        model = model or self.model
        metadata = action_format.read_checkpoint_metadata(resume_from_checkpoint)
        special = action_format.is_special(self.action_format)
        if metadata is not None and metadata["action_format"] != self.action_format:
            raise ValueError("Cannot resume across different action objectives/formats")
        if special and (metadata is None or not action_format.is_special(metadata["action_format"])):
            raise ValueError(
                f"{resume_from_checkpoint} is not an action_special_v1 checkpoint; refusing a "
                "legacy fallback into a changed-vocabulary run (start fresh from the "
                "production checkpoint instead)"
            )
        if not special and metadata is not None and action_format.is_special(metadata["action_format"]):
            raise ValueError(f"{resume_from_checkpoint} is an action_special_v1 checkpoint but "
                             f"this run uses action_format={self.action_format!r}")
        adapter_path = os.path.join(resume_from_checkpoint, "adapter_model.safetensors")
        if os.path.isfile(adapter_path):
            if special:
                action_format.verify_token_mapping(
                    self.processor.tokenizer, metadata["protocol_tokens"]
                )
                rows = modeling.get_action_rows(model)
                for key in ("embedding_rows_expanded", "lm_head_rows_expanded",
                            "tied_word_embeddings", "tokenizer_len_expanded"):
                    if rows.info.get(key) != metadata["vocabulary"].get(key):
                        raise ValueError(f"Resume vocabulary mismatch for {key}: run="
                                         f"{rows.info.get(key)} checkpoint={metadata['vocabulary'].get(key)}")
                modeling.load_action_rows(model, resume_from_checkpoint)
            # Verified load: every saved adapter tensor must land exactly.
            modeling.load_adapter_weights_strict(model.thinker, resume_from_checkpoint)
            print(f"[resume] loaded and verified adapter weights from {resume_from_checkpoint}"
                  + (" plus action rows" if special else ""))
            return
        if special:
            raise FileNotFoundError(f"Missing adapter weights in {resume_from_checkpoint}")
        return super()._load_from_checkpoint(resume_from_checkpoint, model=model)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        # classify_param_region (modeling.py) is the SAME classifier
        # apply_train_mode uses to decide freeze/lora/full per region, so a
        # param's LR group here can never drift from which region actually
        # trained it. Substring-based, so it works unchanged on this
        # trainer's "thinker."-prefixed names, whether or not LoRA/PEFT
        # wrapping added its own prefixes/infixes on top.
        groups = {r: [] for r in (*modeling.REGIONS, modeling.ACTION_ROWS_REGION, "other")}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                region = modeling.classify_param_region(name, self._fusion_module_names)
                groups[region].append(param)

        lrs = {"encoder": self.lr_encoder, "aligner": self.lr_aligner, "llm": self.lr_llm,
               "fusion": self.lr_fusion, modeling.ACTION_ROWS_REGION: self.action_token_lr,
               "other": self.args.learning_rate}
        # Action rows get no weight decay; frozen original vocabulary rows are
        # not optimizer parameters at all, so decay cannot reach them either.
        decay = {modeling.ACTION_ROWS_REGION: 0.0}
        optim_groups = [
            {"params": params, "lr": lrs[name],
             "weight_decay": decay.get(name, self.args.weight_decay)}
            for name, params in groups.items() if params
        ]
        if action_format.is_special(self.action_format):
            if not groups[modeling.ACTION_ROWS_REGION]:
                raise ValueError("action_special_v1 run has no trainable action rows")
            if groups["other"]:
                raise ValueError("action_special_v1 run has unclassified trainable parameters")

        if self.args.process_index == 0:
            for name, params in groups.items():
                print(f"[optimizer] {name:11s}: {sum(p.numel() for p in params)} params")

        self.optimizer = torch.optim.AdamW(
            optim_groups,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )
        return self.optimizer
