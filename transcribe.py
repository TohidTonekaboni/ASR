import subprocess
import tempfile
from pathlib import Path

from nemo.collections.asr.models import ASRModel


CHUNK_SECONDS = 15
MODEL_PATH = "./model/shenava-koochik-v1.0.nemo"

_models = {}


def _disable_pretokenize(model):
    # transcribe() doesn't expose pretokenize, so inject it into the dataloader config.
    original = model._setup_dataloader_from_config

    def setup(config):
        config["pretokenize"] = False
        return original(config)

    model._setup_dataloader_from_config = setup


def _load_model(device):
    if device not in _models:
        model = ASRModel.restore_from(MODEL_PATH, map_location=device)
        model.change_decoding_strategy(decoder_type="ctc")
        model.to(device)
        _disable_pretokenize(model)
        _models[device] = model
    return _models[device]


def transcribe_voice(path, device="cpu"):
    """Transcribe an audio file on the given device ("cpu", "mps", "cuda", ...).

    Returns (text, chunk_texts).
    """
    model = _load_model(device)
    with tempfile.TemporaryDirectory() as tmp:
        stem = Path(tmp) / "chunk"
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-i", str(path), "-ar", "16000", "-ac", "1",
             "-f", "segment", "-segment_time", str(CHUNK_SECONDS), f"{stem}_%03d.wav"],
            check=True,
        )
        chunks = sorted(str(p) for p in Path(tmp).glob("chunk_*.wav"))
        texts = [r.text for r in model.transcribe(chunks)]
    return " ".join(texts), texts
