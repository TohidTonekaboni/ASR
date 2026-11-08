"""Build a WER evaluation set from long recordings.

Pipeline: resample -> split at quiet points -> transcribe with timestamps (OpenAI)
-> merge -> group into utterances -> cut audio -> normalize text -> NeMo manifest.

Usage:
    put OPENAI_API_KEY=... in .env (loaded with python-dotenv)
    optional: PROXY_HOST, PROXY_PORT, PROXY_USERNAME, PROXY_PASSWORD in .env to route
    OpenAI traffic through a proxy (PROXY_SCHEME defaults to http; socks5 needs
    `pip install httpx[socks]`)
    python evaluation_preparing.py [--model whisper-1] [--limit-files N]

Raw API responses are cached in <out>/work/raw, so an interrupted run resumes
without paying for finished pieces again.

NOTE: the transcripts are model output, not human-verified. The clips listed in
review.jsonl are a random sample to correct by hand; fill in "text_corrected".
"""

import argparse
import json
import os
import random
import re
import subprocess
import unicodedata
from pathlib import Path
from urllib.parse import quote

import numpy as np
import soundfile as sf
from dotenv import load_dotenv

SR = 16000
PIECE_SECONDS = 600      # ~19 MB as 16 kHz mono WAV, under the 25 MB upload limit
CUT_SEARCH_SECONDS = 15  # look this far around the target to find a quiet cut point
MIN_UTT, TARGET_MAX_UTT, HARD_MAX_UTT = 3.0, 20.0, 45.0
MIN_KEEP = 1.0           # drop clips shorter than this
GAP_SPLIT = 1.0          # a pause this long ends an utterance once it is >= MIN_UTT
PAD = 0.1                # seconds of padding when cutting, timestamps are loose


# ---------- text normalization (apply the same function to model output when scoring) ----------

_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_CHARS = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ۀ": "ه", "ة": "ه", "‌": " ", "ـ": ""})
_PUNCT = re.compile(r"[\.\,\!\?\:\;\"\'\(\)\[\]\{\}«»“”‘’…،؛؟\-–—_/\\*#@]+")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_CHARS).translate(_DIGITS)
    text = re.sub(r"[ً-ٰٟ]", "", text)  # Arabic diacritics
    text = _PUNCT.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------- audio ----------

def to_16k(src: Path, dst: Path):
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ar", str(SR), "-ac", "1", str(dst)],
        check=True,
    )


def piece_boundaries(audio: np.ndarray) -> list[int]:
    """Sample indices to cut at: about every PIECE_SECONDS, at the quietest 100 ms nearby."""
    n, win = len(audio), SR // 10
    cuts = [0]
    target = PIECE_SECONDS * SR
    while target < n - PIECE_SECONDS * SR // 4:  # avoid a tiny last piece
        lo = max(cuts[-1] + SR, target - CUT_SEARCH_SECONDS * SR)
        hi = min(n - win, target + CUT_SEARCH_SECONDS * SR)
        seg = np.abs(audio[lo:hi].astype(np.float32))
        usable = (len(seg) // win) * win
        energy = seg[:usable].reshape(-1, win).mean(axis=1)
        cut = lo + int(energy.argmin()) * win + win // 2
        cuts.append(cut)
        target = cut + PIECE_SECONDS * SR
    cuts.append(n)
    return cuts


# ---------- transcription ----------

def transcribe_piece(client, path: Path, model: str, cache: Path) -> list[dict]:
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    kwargs = {"model": model, "language": "fa"}
    if "diarize" in model:
        kwargs.update(response_format="diarized_json", chunking_strategy="auto")
    else:
        kwargs.update(response_format="verbose_json", timestamp_granularities=["segment"])
    with open(path, "rb") as f:
        resp = client.audio.transcriptions.create(file=f, **kwargs)
    data = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
    segments = [
        {"start": float(s["start"]), "end": float(s["end"]), "text": (s.get("text") or "").strip()}
        for s in data.get("segments") or []
    ]
    if not segments:
        raise RuntimeError(f"{model} returned no timestamped segments for {path.name}; "
                           f"keys: {list(data)}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(segments, ensure_ascii=False), encoding="utf-8")
    return segments


# ---------- utterances ----------

def group_segments(segments: list[dict]) -> list[dict]:
    """Merge consecutive segments into utterances of about MIN_UTT..TARGET_MAX_UTT seconds."""
    out, cur = [], None

    def flush():
        nonlocal cur
        if cur:
            out.append(cur)
        cur = None

    for s in sorted(segments, key=lambda s: s["start"]):
        if not s["text"] or s["end"] <= s["start"]:
            continue
        if cur:
            gap = s["start"] - cur["end"]
            too_long = s["end"] - cur["start"] > TARGET_MAX_UTT
            pause = gap >= GAP_SPLIT and cur["end"] - cur["start"] >= MIN_UTT
            if too_long or pause:
                flush()
        if cur:
            cur["end"] = s["end"]
            cur["text"] += " " + s["text"]
        else:
            cur = dict(s)
    flush()
    return [u for u in out if MIN_KEEP <= u["end"] - u["start"] <= HARD_MAX_UTT]


# ---------- main ----------

def process_recording(client, src: Path, out: Path, model: str) -> list[dict]:
    rec = src.stem
    work = out / "work"
    wav16 = work / "16k" / f"{rec}.wav"
    to_16k(src, wav16)
    audio, _ = sf.read(wav16, dtype="int16")
    cuts = piece_boundaries(audio)

    segments = []
    for k, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
        piece = work / "pieces" / f"{rec}_{k:03d}.wav"
        cache = work / "raw" / f"{rec}_{k:03d}.json"
        if not cache.exists():
            piece.parent.mkdir(parents=True, exist_ok=True)
            sf.write(piece, audio[a:b], SR, subtype="PCM_16")
        offset = a / SR
        for s in transcribe_piece(client, piece, model, cache):
            segments.append({**s, "start": s["start"] + offset, "end": s["end"] + offset})
        piece.unlink(missing_ok=True)
        print(f"  {rec} piece {k + 1}/{len(cuts) - 1} done", flush=True)

    clips_dir = out / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, u in enumerate(group_segments(segments)):
        text = normalize_text(u["text"])
        if not text:
            continue
        start, end = max(0.0, u["start"] - PAD), min(len(audio) / SR, u["end"] + PAD)
        clip = clips_dir / f"{rec}_{i:05d}.wav"
        sf.write(clip, audio[int(start * SR):int(end * SR)], SR, subtype="PCM_16")
        rows.append({
            "audio_filepath": str(clip),
            "duration": round(end - start, 3),
            "text": text,
            "source": rec,
            "start": round(start, 3),
            "text_raw": u["text"],
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--voices", default="evaluation_dataset/voices")
    p.add_argument("--out", default="evaluation_dataset/prepared")
    p.add_argument("--model", default="whisper-1",
                   help="needs segment timestamps: whisper-1 or gpt-4o-transcribe-diarize")
    p.add_argument("--limit-files", type=int, help="only process the first N recordings (for a trial run)")
    p.add_argument("--review-clips", type=int, default=300, help="size of the random hand-review sample")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import httpx
    from openai import OpenAI  # imported here so normalize_text can be reused without it
    load_dotenv()
    host, port = os.environ.get("PROXY_HOST"), os.environ.get("PROXY_PORT")
    if host and port:
        user, password = os.environ.get("PROXY_USERNAME"), os.environ.get("PROXY_PASSWORD")
        auth = f"{quote(user, safe='')}:{quote(password or '', safe='')}@" if user else ""
        scheme = os.environ.get("PROXY_SCHEME", "http")
        print(f"using proxy {host}:{port}", flush=True)
        client = OpenAI(http_client=httpx.Client(proxy=f"{scheme}://{auth}{host}:{port}", timeout=600))
    else:
        client = OpenAI()  # httpx still honors HTTPS_PROXY / ALL_PROXY if set

    out = Path(args.out)
    sources = sorted(Path(args.voices).glob("*.wav"))[: args.limit_files]
    if not sources:
        p.error(f"no .wav files in {args.voices}")

    rows = []
    for src in sources:
        print(f"{src.name}", flush=True)
        rows += process_recording(client, src, out, args.model)

    with open(out / "manifest.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("audio_filepath", "duration", "text", "source")},
                               ensure_ascii=False) + "\n")

    random.Random(args.seed).shuffle(sample := list(rows))
    with open(out / "review.jsonl", "w", encoding="utf-8") as f:
        for r in sample[: args.review_clips]:
            f.write(json.dumps({"audio_filepath": r["audio_filepath"], "text_gpt": r["text_raw"],
                                "text_corrected": ""}, ensure_ascii=False) + "\n")

    durs = np.array([r["duration"] for r in rows])
    print(f"\n{len(rows)} clips, {durs.sum() / 3600:.2f} h from {len(sources)} recordings")
    print(f"duration s: min {durs.min():.1f}, median {np.median(durs):.1f}, max {durs.max():.1f}")
    print(f"wrote {out / 'manifest.jsonl'} and {out / 'review.jsonl'}")


if __name__ == "__main__":
    main()
