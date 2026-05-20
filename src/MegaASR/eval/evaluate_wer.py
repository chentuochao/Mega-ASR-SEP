# coding=utf-8
import argparse
import json
import re
import unicodedata
from pathlib import Path
import editdistance
import torch
from tqdm import tqdm
from whisper_normalizer.english import EnglishTextNormalizer
from cn_tn import TextNorm

BATCH_SIZE = 8
MAX_NEW_TOKENS = 256
EN_NORM = EnglishTextNormalizer()
ZH_NORM = TextNorm()
ASR_PREFIX = re.compile(r"^\s*language\s+[^<\s]+<asr_text>", re.I)

def clean_text(text):
    text = str(text or "").strip()
    if "<asr_text>" in text:
        text = text.split("<asr_text>", 1)[1]
    return ASR_PREFIX.sub("", text).strip()

def normalize(text, zh=False):
    text = clean_text(text)
    text = unicodedata.normalize("NFKC", text)
    if zh:
        text = ZH_NORM(text)
        text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text)
        return re.sub(r"\s+", "", text)
    text = EN_NORM(text).lower()
    text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text)
    return " ".join(text.split())

def compute_error(ref, pred):
    zh = re.search(r"[\u4e00-\u9fff]", ref + pred) is not None
    ref_units = list(normalize(ref, zh=True)) if zh else normalize(ref).split()
    pred_units = list(normalize(pred, zh=True)) if zh else normalize(pred).split()

    edits = editdistance.eval(ref_units, pred_units)
    ref_len = len(ref_units)
    score = edits / ref_len if ref_len else float(len(pred_units) > 0)
    return ("cer" if zh else "wer"), score, edits, ref_len

def resolve_audio(path, jsonl_path):
    path = Path(path)
    if path.is_absolute():
        return str(path)
    jsonl_dir_path = Path(jsonl_path).resolve().parent / path
    return str(jsonl_dir_path if jsonl_dir_path.exists() else Path.cwd() / path)

def main():
    parser = argparse.ArgumentParser("Run ASR inference and compute WER/CER.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    args = parser.parse_args()
    from qwen_asr import Qwen3ASRModel
    use_cuda = torch.cuda.is_available()
    model = Qwen3ASRModel.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16 if use_cuda else torch.float32,
        device_map="cuda:0" if use_cuda else "cpu",
        max_inference_batch_size=BATCH_SIZE,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    outputs, total_edits, total_ref_len = [], 0, 0

    for i in tqdm(range(0, len(data), BATCH_SIZE), desc="evaluating"):
        batch = data[i:i + BATCH_SIZE]
        audio_paths = [resolve_audio(x["audio"], args.input_jsonl) for x in batch]
        results = model.transcribe(audio=audio_paths, language=None)
        results = results if isinstance(results, list) else [results]
        for item, res in zip(batch, results):
            pred = str(getattr(res, "text", res)).strip()
            metric, score, edits, ref_len = compute_error(item["answer"], pred)
            item["prediction"] = pred
            item["metric"] = metric
            item["wer"] = round(float(score), 6)
            item["num_edits"] = int(edits)
            item["ref_len"] = int(ref_len)
            total_edits += edits
            total_ref_len += ref_len
            outputs.append(item)

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"samples: {len(outputs)}")
    print(f"overall_error: {total_edits / total_ref_len if total_ref_len else 0.0:.6f}")
    print(f"saved: {out_path}")

if __name__ == "__main__":
    main()