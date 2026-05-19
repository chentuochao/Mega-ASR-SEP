## ASR Evaluation

We provide a simple evaluation script for running ASR inference and computing WER/CER.  
The input file should be a JSONL file. Each line only needs two required fields:

```json
{"audio": "examples/audio/noise.wav", "answer": "I usually take the quieter road home because the main street gets crowded after work."}
```


The script will keep all original fields and append the following fields to the output JSONL:

```text
prediction  # model transcription
metric      # "wer" for English samples, "cer" for Chinese samples
wer         # WER/CER score value; CER is also stored in this field for compatibility
num_edits   # edit distance between prediction and ground truth
ref_len     # number of reference words or characters
```

We use Qwen3-ASR as the default inference model:

```python
model = Qwen3ASRModel.from_pretrained(
    args.model_path,
    dtype=torch.bfloat16 if use_cuda else torch.float32,
    device_map="cuda:0" if use_cuda else "cpu",
    max_inference_batch_size=BATCH_SIZE,
    max_new_tokens=MAX_NEW_TOKENS,
)
```

If you want to evaluate another ASR model, replace this part with your own model-loading and inference logic, while keeping the input and output JSONL format unchanged.

### Run Evaluation

```bash
python eval/evaluate_asr.py \
  --model_path ckpt/Mega-ASR \
  --input_jsonl examples/test.jsonl \
  --output_jsonl outputs/pred_with_wer.jsonl
```