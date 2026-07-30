# coding=utf-8
"""Generate a Qwen3-ASR-SEP finetuning dataset (Mega-ASR-SEP's audio_sep /
audio_mix / text JSONL schema) by running a trained speech-separation model's
own inference on synthetic multi-speaker/noise scenes.

For each synthetic scene (see src/datasets/se_parsed_data_multi_noise_asr.py,
same class test/test_se.py evaluates with):
  - audio_mix  = the raw PRE-SEPARATION mixture (device-1 reference mic,
                 channel 0) -- exactly what test_se.py calls `mixture`.
  - audio_sep  = the SEPARATION MODEL'S OWN prediction (NOT the oracle clean
                 target `target_at_dev1`!) -- this is what a production
                 pipeline actually feeds Qwen3-ASR, over-suppression artifacts
                 and all. That's the whole point of the fusion project: giving
                 Qwen3-ASR access to `audio_mix` so it can pull back content
                 `audio_sep` suppressed. Training on the oracle target instead
                 would teach the model nothing about real separator behavior.
  - text       = the near/target speaker's ground-truth transcript.

Samples with no target speaker, or no transcript (the near speaker can be
drawn from a real pre-recorded pool instead of synthetic RIR+LibriTTS -- see
`rir_ratio` below), are skipped.

Scenes are drawn IN ORDER (idx = 0, 1, 2, ...) via the dataset's own default
seeding, not an arbitrary/unbounded seed counter: `dataset[idx]` resolves to
seed=idx for val/test, or idx + epoch*len(dataset) for train (see
Dataset.__getitem__/set_epoch). If `--num_train` exceeds the run's
samples_per_epoch, later scenes come from subsequent epochs (still idx order
within each epoch). val/test re-seeds purely off idx regardless of epoch, so
`--num_val` is capped at samples_per_epoch -- repeating an epoch there would
just replay identical scenes.

Usage (run from anywhere; the Audio_Agent_Perception repo root is added to
sys.path automatically):

    python prepare_data/build_qwen_asr_dataset.py \
        --run_dir /home/ubuntu/Hearvana/datasets/Results_SE/SingleSE/earbud_tfmlp_1mic_finetuning_low_snr \
        --output_dir /home/ubuntu/Hearvana/datasets/Mix_Qwen_ASR_dataset \
        --num_train 5000 --num_val 200

`--run_dir` must contain config.json + checkpoints/{best,last}.pt, i.e. exactly
what test/test_se.py takes as its positional `run_dir` argument. This reuses
that run's own `train_data_args` / `val_data_args` (same rooms/noise/LibriTTS
pools the separation model was trained/evaluated on) but FORCES `rir_ratio=1.0`
by default, so every near speaker goes through the synthetic RIR+LibriTTS path
and therefore has a transcript. Some runs' own val config uses a lower
rir_ratio (real pre-recorded near speaker, no transcript -- tuned for
realistic SI-SDR eval), which yields ZERO usable transcripts; pass
`--rir_ratio -1` to keep the run's own config value instead of overriding it.
"""
import argparse
import json
import os
import sys

import soundfile as sf
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import src.utils as utils  # noqa: E402
from src.datasets.se_parsed_data_multi_noise_asr import Dataset  # noqa: E402
from src.metrics.metrics import Metrics  # noqa: E402

SR = 16000
SNR_MIN = -15
SNR_MAX = 8

def build_dataset(config, split, rir_ratio):
    key = "train_data_args" if split == "train" else "val_data_args"
    if key not in config:
        raise KeyError(f"config.json has no '{key}' section")
    data_args = dict(config[key])
    data_args["split"] = split
    if rir_ratio is not None:
        data_args["rir_ratio"] = rir_ratio
    data_args["augmentations"] = []
    data_args["snr_min"] = SNR_MIN 
    data_args["snr_max"] = SNR_MAX 
    data_args["samples_per_epoch"] = None
    return Dataset(**data_args)


@torch.no_grad()
def separate(model, inputs, device):
    """Mirrors test/test_se.py's run_testcase: run the separation model on one
    scene and return its own (imperfect) single-channel prediction as a CPU
    torch tensor (kept as torch, not numpy, so SI-SDR can be computed against
    the oracle target before converting to numpy for saving)."""
    x_11 = inputs["audio_at_dev1_from_dev1"].unsqueeze(0).to(device)
    x_12 = inputs["audio_at_dev1_from_dev2"].unsqueeze(0).to(device)
    comm_delay = inputs["comm_delay_chunks"]

    drc_gain1 = inputs.get("drc_gain1")
    if drc_gain1 is not None:
        drc_gain1 = drc_gain1.unsqueeze(0).to(device)
        x_11 = x_11 * drc_gain1
        x_12 = x_12 * drc_gain1

    outputs = model(x_11, x_12, comm_delay)
    output = outputs["output"]

    if drc_gain1 is not None:
        output = output / drc_gain1

    output = output.squeeze(0)
    assert output.shape[0] == 1, "expected single-channel separation output"
    return output.cpu()


def generate_split(model, config, split, num_samples, rir_ratio,
                    audio_dir, device, min_duration, max_duration):
    dataset = build_dataset(config, split, rir_ratio)
    dataset_len = len(dataset)
    os.makedirs(audio_dir, exist_ok=True)

    # Mirrors test/test_se.py's SI-SDR computation: si_sdr(est, gt) against the
    # oracle clean target `target_at_dev1` (never saved/trained on -- only used
    # here to measure how much the separator already improved/hurt the signal).
    si_sdr_metric = Metrics("si_sdr")
    sisdri_values = []

    # Iterate idx = 0, 1, 2, ... in order (dataset[idx] with the default
    # seed=None resolves to seed=idx for val/test, or idx + epoch*len(dataset)
    # for train -- see Dataset.__getitem__/set_epoch). val/test always seed
    # purely off idx regardless of epoch, so repeating an epoch there would
    # just replay identical scenes: cap instead of looping forever.
    print(split, dataset_len, "*********")
    if split != "train" and num_samples > dataset_len:
        print(f"[{split}] requested {num_samples} > samples_per_epoch={dataset_len}; "
              f"{split}/test re-seeds purely off idx (no epoch variation), "
              f"so capping at {dataset_len}.")
        num_samples = dataset_len

    records = []
    skipped = {"no_speaker": 0, "no_transcript": 0, "duration": 0, "in_snr": 0, "error": 0}
    epoch, idx = 0, 0
    while len(records) < num_samples:
        if idx >= dataset_len:
            epoch += 1
            dataset.set_epoch(epoch)
            idx = 0

        cur_idx = idx
        idx += 1  # always advance, even on failure, so a broken idx can't loop forever

        try:
            inputs, targets = dataset[cur_idx]

            if targets["num_target_speakers"] <= 0:
                skipped["no_speaker"] += 1
                continue
            text = (targets["transcript"] or "").strip()
            if not text:
                skipped["no_transcript"] += 1
                continue

            mixture = inputs["audio_at_dev1_from_dev1"][0:1]
            duration_sec = mixture.shape[-1] / SR
            if duration_sec < min_duration or duration_sec > max_duration:
                skipped["duration"] += 1
                continue

            target = targets["target_at_dev1"]  # oracle clean target; SI-SDR reference only
            input_sisdr = si_sdr_metric(est=mixture, gt=target).item()
            if input_sisdr < SNR_MIN - 2:
                skipped["in_snr"] += 1
                continue

            sep_audio_t = separate(model, inputs, device)

            sep_sisdr = si_sdr_metric(est=sep_audio_t, gt=target).item()
            sisdr_i = sep_sisdr - input_sisdr
            sisdri_values.append(sisdr_i)

            sep_audio = sep_audio_t.numpy()
            mix_audio = mixture.numpy()

            out_idx = len(records)
            sep_path = os.path.join(audio_dir, f"{out_idx:06d}_sep.wav")
            mix_path = os.path.join(audio_dir, f"{out_idx:06d}_mix.wav")
            # gt_path = os.path.join(audio_dir, f"{out_idx:06d}_gt.wav")
            sf.write(sep_path, sep_audio.T, SR)
            sf.write(mix_path, mix_audio.T, SR)
            # sf.write(gt_path, target.T, SR)

            print(f"[{split}] sample {out_idx:06d}  input_sisdr={input_sisdr:.2f} dB  "
                  f"sisdr_i={sisdr_i:+.2f} dB")

            records.append({
                "audio_sep": os.path.abspath(sep_path),
                "audio_mix": os.path.abspath(mix_path),
                "text": text,
                "input_sisdr": input_sisdr,
                "sisdr_i": sisdr_i,
                "libritts_path": targets["libritts_path"],
            })
        except Exception as e:
            skipped["error"] += 1
            print(f"[{split}] idx={cur_idx} raised {type(e).__name__}: {e} -- skipping sample")
            continue

        if len(records) % 100 == 0:
            avg_sisdri = sum(sisdri_values) / len(sisdri_values)
            print(f"[{split}] {len(records)}/{num_samples}  skipped={skipped}  "
                  f"avg_sisdr_i so far={avg_sisdri:.2f} dB")

    avg_sisdri = sum(sisdri_values) / len(sisdri_values) if sisdri_values else float("nan")
    print(f"[{split}] done: {len(records)} samples  skipped={skipped}  "
          f"avg_sisdr_i={avg_sisdri:.2f} dB")
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True,
                     help="Separation model run dir (contains config.json + checkpoints/)")
    ap.add_argument("--output_dir", required=True,
                     help="e.g. /home/ubuntu/Hearvana/datasets/Mix_Qwen_ASR_dataset")
    ap.add_argument("--num_train", type=int, default=5000)
    ap.add_argument("--num_val", type=int, default=200)
    ap.add_argument("--rir_ratio", type=float, default=1.0,
                     help="Force the synthetic RIR+LibriTTS near-speaker path so every "
                          "sample has a transcript. Pass -1 to keep the run's own "
                          "config value instead of overriding it.")
    ap.add_argument("--min_duration", type=float, default=1.0)
    ap.add_argument("--max_duration", type=float, default=30.0)
    ap.add_argument("--use_last", action="store_true",
                     help="Use checkpoints/last.pt instead of checkpoints/best.pt")
    ap.add_argument("--device", default=None, help="e.g. cuda:0, cpu (default: auto)")
    args = ap.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    rir_ratio = None if args.rir_ratio < 0 else args.rir_ratio

    with open(os.path.join(args.run_dir, "config.json")) as f:
        config = json.load(f)

    module, _ = utils.load_pretrained(args.run_dir, return_params=True, use_last=args.use_last)
    model = module.model.eval().to(device)

    os.makedirs(args.output_dir, exist_ok=True)

    for split, num_samples in [("train", args.num_train), ("val", args.num_val)]:
        if num_samples <= 0:
            continue
        audio_dir = os.path.join(args.output_dir, "audio", split)
        records = generate_split(
            model, config, split, num_samples, rir_ratio,
            audio_dir, device, args.min_duration, args.max_duration,
        )
        jsonl_path = os.path.join(args.output_dir, f"{split}.jsonl")
        with open(jsonl_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"[{split}] wrote {jsonl_path}")


if __name__ == "__main__":
    main()
