"""Centralised ffmpeg/ffprobe binary resolution."""

import shutil


def ffmpeg_bin() -> str:
    """Prefer system ffmpeg, fall back to imageio-ffmpeg's bundled binary."""
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as e:
        raise RuntimeError(
            "ffmpeg not found. Install system ffmpeg (`brew install ffmpeg`) "
            "or `pip install imageio-ffmpeg`."
        ) from e


def ffprobe_bin() -> str:
    """ffprobe must come from the system — imageio-ffmpeg doesn't bundle it."""
    system = shutil.which("ffprobe")
    if not system:
        raise RuntimeError(
            "ffprobe not found. Install system ffmpeg/ffprobe (`brew install ffmpeg`)."
        )
    return system
