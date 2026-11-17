"""Candidate-specific judge supervision; references never enter model inputs."""
import hashlib
import json
import math
import os
import re
import string
from pathlib import Path

import jiwer
from num2words import num2words

SCHEMA = 'audio_candidate_judge_v1'
NORMALIZATION = 'production_normalization_num2words_v1'


def normalize(text):
    if not isinstance(text, str):
        raise ValueError('Reference and candidate must be strings')
    text = text.lower().replace('%', ' percent').replace('&', ' and').replace('@', ' at')
    text = text.replace('-', ' ').replace('per cent', 'percent')
    text = re.sub(r'\b\d+\b', lambda m: num2words(int(m.group())).replace('-', ' ').replace(',', ''), text)
    return ' '.join(text.translate(str.maketrans('', '', string.punctuation)).split())


def candidate_wer(reference, candidate):
    return float(jiwer.wer(normalize(reference), normalize(candidate)))


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_judge_manifest(path, checkpoint, threshold):
    metadata = json.loads(Path(str(path) + '.meta.json').read_text())
    if metadata.get('manifest_schema') != SCHEMA or metadata.get('normalization') != NORMALIZATION:
        raise ValueError(f'{path}: not a prepared judge manifest')
    if metadata.get('manifest_sha256') != fingerprint(path):
        raise ValueError(f'{path}: manifest checksum mismatch')
    if os.path.normpath(metadata.get('checkpoint_id', '')) != os.path.normpath(checkpoint):
        raise ValueError(f'{path}: candidate generation checkpoint mismatch')
    if metadata.get('wer_threshold') != threshold:
        raise ValueError(f'{path}: WER threshold mismatch')
    if not isinstance(metadata.get('generation_config'), dict) or not metadata['generation_config']:
        raise ValueError(f'{path}: missing candidate decoding configuration')
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows or len(rows) != metadata.get('total_rows'):
        raise ValueError(f'{path}: empty or truncated manifest')
    for i, row in enumerate(rows):
        for field in ('audio_sep', 'audio_mix'):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f'{path}:{i}: missing {field}')
        if row.get('prompt') or row.get('quality_context'):
            raise ValueError(f'{path}:{i}: judge does not accept prompt/quality hints')
        wer = candidate_wer(row['text'], row['generated_transcription'])
        if not math.isfinite(wer) or not math.isclose(wer, float(row['label_wer']), abs_tol=1e-10):
            raise ValueError(f'{path}:{i}: WER does not describe this candidate')
        if row['action'] != ('RESPOND' if wer <= threshold else 'REJECT'):
            raise ValueError(f'{path}:{i}: action does not describe this candidate')
    return metadata
