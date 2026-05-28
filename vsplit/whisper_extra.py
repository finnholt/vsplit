"""Optional Whisper transcription. Requires `pip install vsplit[whisper]`."""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def is_available() -> bool:
    try:
        import whisper  # noqa: F401
        return True
    except ImportError:
        return False


def transcribe(
    video_path: str,
    output_dir: Optional[str] = None,
    model_size: str = "base",
    language: Optional[str] = None,
) -> str:
    """Run Whisper, return the path to the generated SRT.

    ``language`` is auto-detected when None.
    """
    try:
        import whisper
    except ImportError as e:
        raise RuntimeError(
            "Whisper not installed. Run `pip install vsplit[whisper]`."
        ) from e

    src = Path(video_path)
    out_dir = Path(output_dir) if output_dir else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    srt_path = out_dir / f"{src.stem}.srt"

    logger.info(f"Loading Whisper model: {model_size}")
    model = whisper.load_model(model_size)
    logger.info(f"Transcribing {src.name} (language={language or 'auto'})...")
    result = model.transcribe(str(src), language=language, verbose=False)

    lines = []
    for i, seg in enumerate(result.get("segments", []), start=1):
        start = _fmt(seg["start"])
        end = _fmt(seg["end"])
        text = seg["text"].strip()
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    srt_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Wrote {srt_path}")
    return str(srt_path)


def _fmt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
