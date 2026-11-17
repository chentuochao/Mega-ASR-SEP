# coding=utf-8
"""Define action protocols, token mappings, targets, and output parsing.

``action_judge_v1`` predicts an action from audio and a production candidate.
Training supervises only the query-position action token. Inference switches
from frozen production transcription to a separate judge context.

``action_first`` generates ACTION and TRANSCRIPT fields in one text sequence.
``action_special_v1`` generates a transcript and predicts a dedicated action
token in separate decoder passes. Its action query sees audio and the prompt,
but does not attend to the generated transcript.

Token spellings and checkpoint validation are shared by training and evaluation.
"""

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

ACTION_FIRST = "action_first"
ACTION_SPECIAL_V1 = "action_special_v1"
ACTION_JUDGE_V1 = "action_judge_v1"
ACTION_FORMATS = (ACTION_FIRST, ACTION_SPECIAL_V1, ACTION_JUDGE_V1)

ACTIONS = ("RESPOND", "REJECT")
QUERY = "QUERY"
# Deterministic registration order: RESPOND, REJECT, then the query token.
# The two action tokens are the action pass's outputs; the query token is an
# input only (it marks "decide the action here") and is never a target.
PROTOCOL_ROLES = (*ACTIONS, QUERY)
PROTOCOL_TOKENS = {
    "RESPOND": "<|action_respond|>",
    "REJECT": "<|action_reject|>",
    QUERY: "<|action_query|>",
}
ACTION_TOKENS = {action: PROTOCOL_TOKENS[action] for action in ACTIONS}

# Checkpoint artifacts for protocols with dedicated action-token rows.
METADATA_FILENAME = "action_format.json"
ROWS_FILENAME = "action_token_rows.safetensors"
# 2 = transcript-first + separate action pass with a query token.
METADATA_VERSION = 2

# Ordinary text whose original rows (mean over its tokens, encoded with
# add_special_tokens=False) initialize each protocol token's rows.
INIT_POLICY = "mean_of_ordinary_label_rows"
INIT_SOURCE_TEXT = {"RESPOND": "RESPOND", "REJECT": "REJECT", QUERY: "ACTION"}

# Legacy textual format: the leading structure is enforced, the transcript is
# captured as-is.
_LEGACY_PATTERN = re.compile(r"\AACTION: (RESPOND|REJECT)\nTRANSCRIPT: ([\s\S]*)\Z")


def validate_format(action_format: str) -> str:
    if action_format not in ACTION_FORMATS:
        raise ValueError(f"Unknown action_format {action_format!r}; choices: {ACTION_FORMATS}")
    return action_format


def is_special(action_format: Optional[str]) -> bool:
    return action_format in (ACTION_SPECIAL_V1, ACTION_JUDGE_V1)


# --------------------------------------------------------------------------
# Tokenizer registration / validation
# --------------------------------------------------------------------------

def _added_special_strings(tokenizer) -> set:
    values = set(getattr(tokenizer, "additional_special_tokens", None) or [])
    values.update(getattr(tokenizer, "all_special_tokens", None) or [])
    return values


def register_protocol_tokens(tokenizer) -> Dict[str, int]:
    """Fresh-run only: add the three protocol spellings as additional special
    tokens.

    Existing special tokens are preserved (``replace_additional_special_tokens
    =False``). Re-registering on a tokenizer that already carries them as
    special tokens is a no-op. A spelling that already exists as an *ordinary*
    vocabulary entry is refused rather than silently reused.
    """
    vocab = tokenizer.get_vocab()
    special = _added_special_strings(tokenizer)
    to_add = []
    for role in PROTOCOL_ROLES:
        spelling = PROTOCOL_TOKENS[role]
        if spelling in vocab and spelling not in special:
            raise ValueError(
                f"Protocol spelling {spelling!r} already exists as an ordinary token "
                f"(id={vocab[spelling]}); refusing to reuse it"
            )
        if spelling not in vocab:
            to_add.append(spelling)
    if to_add:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": to_add},
            replace_additional_special_tokens=False,
        )
    return resolve_protocol_token_ids(tokenizer)


def resolve_protocol_token_ids(tokenizer, forbidden_ids: Iterable[int] = ()) -> Dict[str, int]:
    """Return {role: id} for RESPOND, REJECT, and QUERY on a tokenizer that
    already carries the tokens.

    Never registers anything. Raises if a spelling is missing, does not encode
    to exactly one ID, collides with another protocol token / EOS / PAD / UNK,
    or hits one of ``forbidden_ids`` (e.g. audio placeholder IDs).
    """
    special = _added_special_strings(tokenizer)
    ids = {}
    for role in PROTOCOL_ROLES:
        spelling = PROTOCOL_TOKENS[role]
        if spelling not in special:
            raise ValueError(
                f"Tokenizer does not carry {spelling!r} as a special token. Protocol "
                "tokens are only registered for a fresh run; load the checkpoint's "
                "own tokenizer instead of registering on reload."
            )
        encoded = list(tokenizer(spelling, add_special_tokens=False)["input_ids"])
        if len(encoded) != 1:
            raise ValueError(f"{spelling!r} encodes to {encoded}, expected exactly one ID")
        ids[role] = int(encoded[0])
    if len(set(ids.values())) != len(ids):
        raise ValueError(f"Protocol tokens share an ID: {ids}")
    reserved = {
        "eos": tokenizer.eos_token_id,
        "pad": tokenizer.pad_token_id,
        "unk": tokenizer.unk_token_id,
    }
    forbidden = set(int(x) for x in forbidden_ids)
    for role, token_id in ids.items():
        for name, value in reserved.items():
            if value is not None and token_id == value:
                raise ValueError(f"{role} token id {token_id} equals the {name} token id")
        if token_id in forbidden:
            raise ValueError(f"{role} token id {token_id} collides with a reserved model token")
    return ids


def action_ids_only(protocol_ids: Dict[str, int]) -> Dict[str, int]:
    return {action: protocol_ids[action] for action in ACTIONS}


def verify_token_mapping(tokenizer, expected: Dict[str, Any], forbidden_ids: Iterable[int] = ()):
    """Check that a loaded tokenizer reproduces a saved mapping exactly.

    ``expected`` is the metadata's ``protocol_tokens`` block:
    ``{role: {"token": spelling, "id": int}}``.
    """
    for role in PROTOCOL_ROLES:
        entry = expected.get(role)
        if not entry or entry.get("token") != PROTOCOL_TOKENS[role]:
            raise ValueError(f"Saved protocol-token metadata for {role} is missing or has "
                             f"an unexpected spelling: {entry}")
    actual = resolve_protocol_token_ids(tokenizer, forbidden_ids)
    saved = {role: int(expected[role]["id"]) for role in PROTOCOL_ROLES}
    if actual != saved:
        raise ValueError(f"Tokenizer protocol mapping {actual} does not match saved mapping {saved}")
    return actual


def check_reserved_text(value: Any, field_name: str, row_index: Optional[int] = None):
    """Reject reserved action-token spellings in manifest-provided text."""
    if value in (None, ""):
        return
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    hits = [s for s in PROTOCOL_TOKENS.values() if s in text]
    if hits:
        where = f" (row {row_index})" if row_index is not None else ""
        raise ValueError(
            f"Field {field_name!r}{where} contains reserved protocol token spelling(s) {hits}; "
            "these are reserved in action_special_v1 and may not appear in manifests"
        )


# --------------------------------------------------------------------------
# Target construction
# --------------------------------------------------------------------------

def build_legacy_target_parts(action: str, transcript: str, eos: str):
    """action_first: return (action_text, transcript_text) of the single target."""
    if action not in ACTIONS:
        raise ValueError(f"Invalid action {action!r}; expected one of {ACTIONS}")
    return f"ACTION: {action}\n", f"TRANSCRIPT: {transcript}{eos}"


def build_transcript_target(transcript: str, eos: str) -> str:
    """action_special_v1 transcript pass: exactly the ordinary ASR target."""
    return f"{transcript}{eos}"


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

@dataclass
class ActionParse:
    """Parsed action output.

    ``parsed_action`` is None whenever ``action_valid`` is False. The effective
    policy action is REJECT unless the action is a valid RESPOND AND the
    transcript passed validation -- a RESPOND transcript is never forwarded
    unvalidated -- but a fallback REJECT is never counted as a generated
    action (``parsed_action`` keeps what the model actually chose).
    ``format_valid`` describes the transcript pass and is None when no
    transcript was generated (early REJECT).
    """
    parsed_action: Optional[str]
    effective_action: str
    action_valid: bool
    format_valid: Optional[bool]
    transcript: Optional[str]
    transcript_generated: bool
    failure_reason: Optional[str]
    action_failure_reason: Optional[str] = None
    raw_ids: List[int] = field(default_factory=list)
    raw_text: str = ""
    action_token_id: Optional[int] = None

    def to_dict(self):
        return asdict(self)


def parse_legacy_text(text: str) -> ActionParse:
    """Parse the legacy ``action_first`` ordinary-text continuation."""
    # Text that does not match the legacy structure yields REJECT and no transcript.
    match = _LEGACY_PATTERN.fullmatch(text.rstrip())
    if not match:
        return ActionParse(None, "REJECT", False, False, "", True,
                           "legacy_format_mismatch", "legacy_format_mismatch", raw_text=text)
    return ActionParse(match.group(1), match.group(1), True, True,
                       match.group(2).strip(), True, None, raw_text=text)


def parse_action_id(action_token_id: Optional[int], protocol_ids: Dict[str, int]):
    """Map the action pass's chosen token to (action or None, failure reason)."""
    if action_token_id is None:
        return None, "no_action_step"
    id_to_action = {protocol_ids[a]: a for a in ACTIONS}
    action = id_to_action.get(int(action_token_id))
    if action is not None:
        return action, None
    if int(action_token_id) == protocol_ids[QUERY]:
        return None, "query_token_predicted"
    return None, "non_action_token"


def parse_transcript_ids(
    continuation_ids: Sequence[int],
    protocol_ids: Dict[str, int],
    eos_ids: Iterable[int],
    decode,
    pad_id: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
):
    """Validate the transcript pass from generated IDs (prefix already
    stripped by the actual input width). ``decode`` must NOT skip special
    tokens, so no protocol token can silently vanish into the text.

    Returns (transcript, format_valid, reason). The transcript is whatever
    text precedes the first EOS (or a protocol token), so invalid outputs
    still contribute their partial text to WER.
    """
    ids = [int(x) for x in continuation_ids]
    eos = set(int(x) for x in eos_ids)
    protocol = set(protocol_ids.values())
    eos_index = next((i for i, t in enumerate(ids) if t in eos), None)
    body = ids[:eos_index] if eos_index is not None else ids
    tail = ids[eos_index + 1:] if eos_index is not None else []

    reason = None
    protocol_at = next((i for i, t in enumerate(body) if t in protocol), None)
    if protocol_at is not None:
        reason = "protocol_token_in_transcript"
        body = body[:protocol_at]
    if eos_index is None:
        truncated = max_new_tokens is not None and len(ids) >= max_new_tokens
        reason = reason or ("max_new_tokens_truncation" if truncated else "missing_eos")
    elif not all(t in eos or (pad_id is not None and t == pad_id) for t in tail):
        reason = reason or "content_after_eos"
    transcript = decode(body).strip() if body else ""
    return transcript, reason is None, reason


def combine_parse(
    continuation_ids: Optional[Sequence[int]],
    action_token_id: Optional[int],
    protocol_ids: Dict[str, int],
    eos_ids: Iterable[int],
    decode,
    pad_id: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
) -> ActionParse:
    """Combine the two passes into one result. ``continuation_ids`` is None
    when the transcript pass was skipped (early REJECT)."""
    action, action_reason = parse_action_id(action_token_id, protocol_ids)
    action_valid = action is not None
    if continuation_ids is None:
        return ActionParse(action, action if action_valid else "REJECT", action_valid, None,
                           None, False, action_reason, action_reason,
                           action_token_id=action_token_id)
    transcript, format_valid, reason = parse_transcript_ids(
        continuation_ids, protocol_ids, eos_ids, decode, pad_id, max_new_tokens
    )
    ids = [int(x) for x in continuation_ids]
    effective = "RESPOND" if action == "RESPOND" and format_valid else "REJECT"
    return ActionParse(
        action, effective, action_valid, format_valid,
        transcript, True, action_reason or reason, action_reason, ids,
        decode(ids) if ids else "", action_token_id,
    )


DECODING_MODES = ("unconstrained", "constrained")


def choose_action_token(action_logits, protocol_ids: Dict[str, int], decoding_mode: str) -> int:
    """Greedy pick at the query position. 'unconstrained' takes the argmax over
    the full vocabulary (the baseline; may be invalid). 'constrained' restricts
    the choice to the two action IDs (always valid; report it separately)."""
    if decoding_mode == "unconstrained":
        return int(action_logits.argmax().item())
    if decoding_mode == "constrained":
        best = max(ACTIONS, key=lambda a: float(action_logits[protocol_ids[a]]))
        return protocol_ids[best]
    raise ValueError(f"Unknown decoding_mode {decoding_mode!r}")


# --------------------------------------------------------------------------
# Checkpoint metadata
# --------------------------------------------------------------------------

def read_checkpoint_metadata(checkpoint_dir: str) -> Optional[dict]:
    path = os.path.join(checkpoint_dir, METADATA_FILENAME)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        metadata = json.load(handle)
    expected_version = 3 if metadata.get("action_format") == ACTION_JUDGE_V1 else METADATA_VERSION
    if metadata.get("metadata_version") != expected_version:
        raise ValueError(f"Unsupported action metadata version in {path}: "
                         f"{metadata.get('metadata_version')} (expected {expected_version})")
    validate_format(metadata.get("action_format", ""))
    return metadata


def write_checkpoint_metadata(checkpoint_dir: str, metadata: dict):
    path = os.path.join(checkpoint_dir, METADATA_FILENAME)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path
