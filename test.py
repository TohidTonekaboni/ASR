import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from transcribe import transcribe_voice


parser = argparse.ArgumentParser()
parser.add_argument("paths", nargs="*", help="audio files (default: everything in files/)")
parser.add_argument("--device", default="cpu", help="cpu, mps, cuda, ...")
args = parser.parse_args()

paths = args.paths or sorted(
    str(p) for p in Path("files").iterdir() if p.is_file() and p.suffix != ".json"
)
if not paths:
    parser.error("No audio found.")

for path in paths:
    start = time.perf_counter()
    text, chunks = transcribe_voice(path, device="mps")
    elapsed = round(time.perf_counter() - start, 2)
    result = {
        "file": path,
        "device": args.device,
        "transcribed_at": datetime.now().isoformat(timespec="seconds"),
        "processing_seconds": elapsed,
        "chunks": chunks,
        "text": text,
    }
    # Same name as the audio file, with a .json extension, next to it.
    output = Path(path).with_suffix(".json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== {path}\n{text}\nWrote {output} (took {elapsed}s)")
