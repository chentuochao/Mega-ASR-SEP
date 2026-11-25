"""Frozen production transcription followed by transcript-conditioned judging."""
from time import perf_counter

import torch

import action_format
import modeling
from dataloader import _build_judge_batch, _model_prefix


def _to_model(batch, model):
    device = next(model.parameters()).device
    return {k: v.to(device=device, dtype=model.dtype if v.is_floating_point() else v.dtype)
            for k, v in batch.items()}


@torch.no_grad()
def judge_candidate(model, processor, sep_audio, mix_audio, candidate,
                    language='English', decoding_mode='unconstrained'):
    if model.training:
        raise ValueError('Call model.eval() before inference')
    # The dummy label only satisfies the common collator; it never enters forward.
    batch = _build_judge_batch(processor, [{'generated_transcription': candidate, 'action': 'RESPOND'}],
                              [sep_audio], language, 'none', 1.0)
    prompt = _model_prefix(processor, {}, language, 'none')
    mix = processor(text=[prompt], audio=[mix_audio], return_tensors='pt', padding=True)
    batch['mix_input_features'] = mix['input_features']
    batch['mix_feature_attention_mask'] = mix['feature_attention_mask']
    batch.pop('labels')
    batch.pop('action_sample_weight')
    logits = model(**_to_model(batch, model), judge_only=True)['logits'][0, -1].float()
    ids = action_format.resolve_protocol_token_ids(processor.tokenizer)
    token = action_format.choose_action_token(logits, ids, decoding_mode)
    action, failure = action_format.parse_action_id(token, ids)
    return {'generated_transcription': candidate, 'action_token_id': int(token),
            'parsed_action': action, 'action_valid': failure is None,
            'effective_action': action if action is not None else 'REJECT',
            'action_failure_reason': failure,
            'action_logits': {a: float(logits[ids[a]]) for a in action_format.ACTIONS}}


@torch.no_grad()
def transcribe_and_judge(model, judge_processor, production_processor, sep_audio, mix_audio,
                         language='English', max_new_tokens=256, decoding_mode='unconstrained',
                         measure_time=False):
    """Transcribe in production mode, then judge in a separate decoder context.

    Transcription uses the base processor and disables judge LoRA and row
    overrides. Judging uses the extended processor and recomputes audio features.
    Decoder caches are not shared between the two adapter modes.
    """
    if model.training:
        raise ValueError('Call model.eval() before inference')
    def timestamp():
        if next(model.parameters()).device.type == 'cuda':
            torch.cuda.synchronize(next(model.parameters()).device)
        return perf_counter()

    started = timestamp() if measure_time else None
    prompt = _model_prefix(production_processor, {}, language, 'none')
    inputs = production_processor(text=[prompt], audio=[sep_audio], return_tensors='pt', padding=True)
    mix = production_processor(text=[prompt], audio=[mix_audio], return_tensors='pt', padding=True)
    inputs['mix_input_features'] = mix['input_features']
    inputs['mix_feature_attention_mask'] = mix['feature_attention_mask']
    inputs = _to_model(inputs, model)
    with modeling.production_transcription_mode(model):
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    sequences = generated.sequences if hasattr(generated, 'sequences') else generated
    continuation = sequences[0, inputs['input_ids'].shape[1]:].tolist()
    candidate = production_processor.tokenizer.decode(continuation, skip_special_tokens=True,
                                                      clean_up_tokenization_spaces=False)
    transcribed = timestamp() if measure_time else None
    result = judge_candidate(model, judge_processor, sep_audio, mix_audio, candidate,
                             language, decoding_mode)
    eos = modeling.generation_eos_ids(production_processor.tokenizer)
    result['transcript_complete'] = bool(continuation and continuation[-1] in eos)
    if not result['transcript_complete']:
        result['effective_action'] = 'REJECT'
    if measure_time:
        finished = timestamp()
        result['stage_seconds'] = {'asr': transcribed - started, 'judge': finished - transcribed}
    return result
