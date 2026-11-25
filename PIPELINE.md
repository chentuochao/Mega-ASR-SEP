# End-to-end action and QA evaluation

The evaluator connects production ASR, the trained action judge, a QA responder, and QA correctness scoring.
It accepts prepared separated-target and mixture audio. It does not run audio separation.
Use an `action_judge_v1` checkpoint from the [action-model implementation](docs/action-model.md).

## Model stages

```text
paired audio → production ASR → generated transcript
paired audio + generated transcript → action judge → RESPOND or REJECT
RESPOND + generated transcript → text model → answer → correctness scoring
REJECT → unanswered request
```

`scripts/eval_qa_pipeline.py` runs `eval_judge.py` in pipeline mode.
Transcription disables the judge adapter and protocol-row overrides.
Judging enables them in a separate decoder context.
Incomplete transcripts and invalid actions retain the action evaluator's REJECT policy.
REJECT produces no QA request. Retranscription and clarification are future extensions.

The default `judge` policy follows the effective action.
`always_respond` answers every generated transcript to measure the effect of rejection.
`reference_text` answers the reference question to measure the responder's text-only performance.
Only the explicit `reference_text` baseline supplies a reference question to the responder.
Answer keys enter scoring only.

Direct inference uses separate processes for ASR, QA, and local correctness grading.
Each process group exits before the next model loads.
`--gpu-audit` checks every allocated GPU before and after each model stage.
Set `CUDA_VISIBLE_DEVICES` to the allocated devices when using this option.
The HTTP mode uses an independently managed chat-completions service.

## Input data

Use one JSON object per line:

```json
{"sample_id":"q0","audio_sep":"/data/q0-sep.wav","audio_mix":"/data/q0-mix.wav","text":"What is two plus two?","qa_task":"exact_match","qa_answers":["4","four"],"benchmark":"custom"}
```

Use unique string identifiers and absolute audio paths.
`text` must contain the reference transcription of the complete spoken question.
For multiple-choice tasks, include the spoken options in that question.

| `qa_task` | Required scoring fields | Evaluation |
|---|---|---|
| `exact_match` | `qa_answers` | Normalized final-answer match |
| `multiple_choice` | `qa_answers`, `qa_choices` | One allowed option letter |
| `voicebench_sdqa` | `qa_answers`, `benchmark_record.prompt`, `benchmark_record.reference` | VoiceBench PEDANT and optional model grading |
| `unscored` | None | Generation only |

VCTK can measure ASR and action behavior, but it has no QA answer keys.
Use prepared VoiceBench sd-QA pairs for evaluation across all stages.

Convert an existing sd-QA manifest containing `audio_sep`, `audio_mix`, `text`, `prompt`, and `reference`:

```bash
python scripts/prepare_sdqa_manifest.py \
  --source "$SOURCE_MANIFEST" --out-jsonl "$RUN/qa.jsonl"
```

The converter preserves the source and writes a checksum sidecar.
Add `--stage-audio-dir "$JOB_TMP/audio"` to copy audio into a new temporary directory.
Extract archives on the compute node before conversion.

For other datasets, `scripts/prepare_qa_benchmark.py` joins benchmark metadata to prepared audio pairs by `source_id`.
Its `--help` describes local exports, pinned Hugging Face revisions, and task selection.
Open-ended VoiceBench subsets require their own evaluators; `unscored` does not implement the 1–5 quality rubric.

## Environments and model files

Keep the production ASR, text inference, and CPU scoring environments separate.
Set `ASR_PYTHON` and `QA_PYTHON` to their executable paths or executable container wrappers.
The parent evaluator uses only Python's standard library.

The verified FP8 runtime uses vLLM `0.24.0+cu129`, PyTorch `2.11.0+cu129`, and Transformers `5.8.0`.
Use a compatible NVIDIA driver and CUDA runtime.
The CPU scoring environment uses Python 3.12:

```bash
python3.12 -m venv "$RUN/scoring-env"
SCORING_PYTHON="$RUN/scoring-env/bin/python"
"$SCORING_PYTHON" -m pip install -r requirements-voicebench-eval.txt
python scripts/download_voicebench_resources.py --out-dir "$RUN/eval_resources"
```

The resource downloader verifies fixed classifier checksums before saving them.
The evaluator also verifies vendored VoiceBench source checksums before loading its prompts and evaluator methods.
See [source provenance](third_party/voicebench_sdqa/README.md).

Download the selected text models from an environment with `huggingface_hub`:

```bash
python scripts/download_response_model.py \
  --model Qwen/Qwen3.6-35B-A3B-FP8 --out-dir "$QA_MODEL"
python scripts/download_response_model.py \
  --model Qwen/Qwen3.8-27B-FP8 --out-dir "$GRADER_MODEL"
```

The downloader resolves a model commit and records file checksums.
Use `--revision` with a full commit identifier to repeat a specific download.
Use separate directories for different model revisions.

## Run a smoke test

Run from the repository root on an allocated compute node.
Set `RUN`, `MANIFEST`, `JUDGE_CHECKPOINT`, `QA_MODEL`, and `GRADER_MODEL` to your paths.
Use a new output directory for every invocation.

| GPU configuration | `TP` | GPUs required in total |
|---|---:|---:|
| One L40S, 48 GiB | `1` | 1 |
| Two Quadro RTX 6000 cards, 24 GiB each, on one node | `2` | 2 |

Both configurations passed individual model inference and complete pipeline smoke tests with an 8,192-token context.
vLLM divides the model weights when tensor parallelism is `2`.
The RTX configuration uses float16 compute and Marlin weight-only FP8 kernels with dtype `auto`.
Reuse the same allocated devices for each stage. A40 compatibility requires a separate test.

```bash
TP=1  # Use 2 for the two-card RTX configuration.
python scripts/eval_qa_pipeline.py \
  --manifest "$MANIFEST" --checkpoint "$JUDGE_CHECKPOINT" \
  --asr-python "$ASR_PYTHON" --qa-python "$QA_PYTHON" \
  --qa-model-path "$QA_MODEL" --qa-backend vllm \
  --qa-max-model-len 8192 --qa-max-tokens 2048 \
  --qa-gpu-memory-utilization 0.90 --qa-enforce-eager \
  --qa-tensor-parallel-size "$TP" --qa-dtype auto \
  --local-grader-model-path "$GRADER_MODEL" \
  --local-grader-max-model-len 8192 --local-grader-max-tokens 1024 \
  --local-grader-gpu-memory-utilization 0.90 --local-grader-enforce-eager \
  --local-grader-tensor-parallel-size "$TP" --local-grader-dtype auto \
  --qa-eval-python "$SCORING_PYTHON" \
  --voicebench-resources "$RUN/eval_resources" --voicebench-gpt-mode off \
  --policies judge always_respond reference_text --gpu-audit \
  --limit 8 --out-dir "$RUN/smoke8"
```

Remove `--limit 8` and select a new output directory for the full benchmark.
Use `--base-model-path` and `--production-adapter-path` if checkpoint dependencies moved.

For PEDANT-only evaluation, omit all `--local-grader-*` arguments and retain `--voicebench-gpt-mode off`.
For official GPT grading, omit those arguments and select `--voicebench-gpt-mode required`.
That mode reads `OPENAI_API_KEY` from the environment and uses the pinned upstream `gpt-4o-mini` configuration.
The local Qwen correctness score remains separate from the official GPT score.

For an existing QA service, replace direct QA arguments with `--qa-endpoint` and `--qa-model`.
Use the full chat-completions URL. Authentication reads `QA_API_KEY` when present.
`infer_vllm.py --task respond` also accepts `--qa-model-path` and `--transcript` for a single direct response.
`--task serve-qa` exposes a localhost chat-completions endpoint and `/health`.
The original ASR command remains the default task.

## Results and correctness

`asr_action/` contains WER, action accuracy, and action confusion metrics.
`generation.jsonl` preserves QA generations before VoiceBench scoring.
`persample.jsonl` contains joined predictions, responses, and scores.
`summary.json` contains coverage, answered accuracy, full-manifest task success, latency, and source checksums.
`model_stages.jsonl` records process exits and GPU checks.
Local grading also saves all raw votes in `local_grader_votes.jsonl`.

For sd-QA, the upstream rubric asks whether an answer matches a reference answer.
Three Yes/No votes determine model-graded correctness.
PEDANT uses the upstream learned answer-matching classifier, rather than exact string matching.
The upstream summary names its PEDANT percentage `panda`.
Answered accuracy excludes rejected requests. Full-manifest task success counts them as unanswered.
Model errors remain errors and cause a nonzero exit status.
VoiceBench scores generated text at the token limit; strict exact-match mode treats truncated completions as errors.

Add `--flagged-dir "$RUN/flagged"` to copy bad-WER audio for inspection.
The default flag threshold matches the action evaluation threshold.
`--flag-wer-threshold 0.75` flags WER strictly greater than 0.75, regardless of the predicted action.
This inspection does not change routing or scoring.

## Replay, resume, and tests

Replace `--checkpoint` with `--asr-results-dir` to replay a completed, checksummed ASR/action stage.
Keep the manifest bytes, row order, selected limit, and prediction files unchanged.
Add `--qa-resume-dir` to reuse a completed prefix from an interrupted direct QA stage.
Resume requires matching model provenance, generation settings, and input records.
It writes to a new output directory and preserves previous results.

Run the focused tests with the CPU scoring environment:

```bash
VOICEBENCH_TEST_RESOURCES="$RUN/eval_resources" \
  "$SCORING_PYTHON" -m unittest discover -s tests -v
```

Tests check routing, reference isolation, real PEDANT parity, HTTP failures, resume validation, and serial worker lifetimes.
GPU model execution requires a separate compute-node smoke test.
