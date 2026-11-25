"""Run model stages in separate process groups, with optional GPU-idle barriers."""
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def allocated_gpu_uuids():
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if not visible or visible in ('-1', 'NoDevFiles'):
        raise ValueError('--gpu-audit requires explicit CUDA_VISIBLE_DEVICES for the allocated GPU')
    result = subprocess.run(['nvidia-smi', '-i', visible, '--query-gpu=uuid', '--format=csv,noheader'],
                            check=True, text=True, capture_output=True)
    return set(result.stdout.strip().splitlines())


def wait_gpu_idle(uuids, timeout=30):
    deadline = time.monotonic() + timeout
    while True:
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                                 '--format=csv,noheader,nounits'], check=True, text=True, capture_output=True)
        active = [dict(gpu_uuid=row[0].strip(), pid=row[1].strip(), used_memory_MiB=row[2].strip())
                  for row in csv.reader(result.stdout.splitlines()) if len(row) == 3 and row[0].strip() in uuids]
        if not active:
            return dict(gpu_uuids=sorted(uuids), active_compute_processes=[])
        if time.monotonic() >= deadline:
            raise RuntimeError(f'The allocated GPU is still occupied; the next model was not loaded: {active}')
        time.sleep(1)


def run_model_stage(name, command, log_path, events_path, *, gpu_audit=False, accepted_codes=(0,)):
    """Wait for model exit and terminate remaining workers before the next stage."""
    uuids = allocated_gpu_uuids() if gpu_audit else None
    before = wait_gpu_idle(uuids) if uuids else None
    started = time.perf_counter()
    event = dict(stage=name, command=[str(arg) for arg in command],
                 started_utc=datetime.now(timezone.utc).isoformat(), gpu_before=before)
    with Path(log_path).open('x') as log:
        process = subprocess.Popen([str(arg) for arg in command], stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        event['pid'] = process.pid
        try:
            code = process.wait()
        finally:
            # This process group belongs only to this stage, including vLLM workers.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            if process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    event.update(returncode=code, ended_utc=datetime.now(timezone.utc).isoformat(),
                 wall_seconds=time.perf_counter() - started,
                 gpu_after=wait_gpu_idle(uuids) if uuids else None)
    with Path(events_path).open('a') as handle:
        handle.write(json.dumps(event, allow_nan=False) + '\n')
    if code not in accepted_codes:
        raise subprocess.CalledProcessError(code, command)
    return event
