# Pinned VoiceBench sd-QA source

These files are copied unchanged from the revision recorded in `PROVENANCE.json`.
The upstream license is included.
Do not edit the copied source to change evaluation behavior.

`scripts/voicebench_sdqa.py` reads the original response prompts and grader settings from this source.
It executes the original `QAEvaluator` and `majority_vote` methods.
PEDANT uses `qa_metrics==0.2.17` with separately pinned classifier and vectorizer files.
Initialization loads those local files instead of downloading mutable files into the installed package.
The evaluation methods remain unchanged.

`PEDANT_RESOURCES.json` records the fixed classifier source paths and checksums.
Run `python scripts/download_voicebench_resources.py --out-dir /path/to/new-resources` from the repository root to retrieve them.
The upstream classifier filenames have no extension; the local filenames use `.pkl`.

The responding model receives the generated transcript.
The reference question and answer enter evaluation only, except for the explicit reference-text baseline.
