import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
from nemo.collections.asr.models import ASRModel

# Usage: python compare.py [audio]   (default: files/260318_1611.mp3)
CHUNK_SECONDS = 15
BATCH_SIZE = 16
DEVICES = ["cpu", "mps"]

audio = sys.argv[1] if len(sys.argv) > 1 else "files/260318_1611.mp3"

asr_model = ASRModel.restore_from("./model/shenava-koochik-v1.0.nemo", map_location="cpu")
asr_model.change_decoding_strategy(decoder_type="ctc")
asr_model.eval()

with tempfile.TemporaryDirectory() as tmp:
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", audio, "-ar", "16000", "-ac", "1",
         "-f", "segment", "-segment_time", str(CHUNK_SECONDS), f"{tmp}/c_%04d.wav"],
        check=True,
    )
    chunks = sorted(str(p) for p in Path(tmp).glob("c_*.wav"))
    print(f"{audio}: {len(chunks)} chunks of {CHUNK_SECONDS}s")

    results = {}
    for device in DEVICES:
        if device == "mps" and not torch.backends.mps.is_available():
            print("mps not available, skipping")
            continue
        asr_model.to(device)
        
        asr_model.transcribe(chunks[:BATCH_SIZE], batch_size=BATCH_SIZE, verbose=False)

        start = time.perf_counter()
        texts = [r.text for r in asr_model.transcribe(chunks, batch_size=BATCH_SIZE, verbose=False)]
        elapsed = time.perf_counter() - start
        results[device] = (elapsed, texts)
        print(f"{device}: {elapsed:.2f}s")

print("\n=== Summary")
for device, (elapsed, texts) in results.items():
    print(f"{device:>4}: {elapsed:8.2f}s")
if len(results) == 2:
    cpu, mps = results["cpu"][0], results["mps"][0]
    print(f"mps is {cpu / mps:.2f}x {'faster' if mps < cpu else 'slower'} than cpu")
    same = results["cpu"][1] == results["mps"][1]
    print("transcripts identical:", same)
