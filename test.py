import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from nemo.collections.asr.models import ASRModel


CHUNK_SECONDS = 15

paths = sys.argv[1:] or sorted(
    str(p) for p in Path("files").iterdir() if p.is_file() and p.suffix != ".json"
)
if not paths:
    sys.exit("No audio found. Usage: python test.py <audio> [more ...]")

asr_model = ASRModel.restore_from("./model/shenava-koochik-v1.0.nemo")

asr_model.change_decoding_strategy(decoder_type="ctc")

with tempfile.TemporaryDirectory() as tmp:
    for path in paths:
        start = time.perf_counter()
        stem = Path(tmp) / Path(path).name.replace(".", "_")

        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-i", path, "-ar", "16000", "-ac", "1",
             "-f", "segment", "-segment_time", str(CHUNK_SECONDS), f"{stem}_%03d.wav"],
            check=True,
        )
        chunks = sorted(str(p) for p in Path(tmp).glob(f"{stem.name}_*.wav"))
        texts = [r.text for r in asr_model.transcribe(chunks)]
        elapsed = round(time.perf_counter() - start, 2)
        result = {
            "file": path,
            "processing_seconds": elapsed,
            "chunks": texts,
            "text": " ".join(texts),
        }
        # Same name as the audio file, with a .json extension, next to it.
        output = Path(path).with_suffix(".json")
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n=== {path}\n{result['text']}\nWrote {output} (took {elapsed}s)")
