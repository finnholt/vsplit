"""Cut a source video into per-segment mp4 files using ffmpeg."""

import logging
import re
import subprocess
from pathlib import Path
from typing import Dict, List

from vsplit.ffmpeg_utils import ffmpeg_bin

logger = logging.getLogger(__name__)

_SAFE_NAME_RE = re.compile(r"[^\w一-鿿\-]+")


def _sanitize(name: str, max_len: int = 40) -> str:
    cleaned = _SAFE_NAME_RE.sub("_", name).strip("_")
    return (cleaned or "segment")[:max_len]


def export_segments(
    video_path: str,
    segments: List[Dict],
    output_dir: str,
    *,
    reencode: bool = True,
) -> List[str]:
    """Cut ``video_path`` into one mp4 per segment.

    ``reencode=True`` (default) re-encodes with libx264, so boundaries are
    frame-accurate. ``reencode=False`` uses stream-copy — much faster, but
    cuts snap to the nearest keyframe and the resulting file duration may
    differ from the requested range by several seconds.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = Path(video_path)
    exported: List[str] = []

    for seg in segments:
        start = float(seg.get("start_seconds", 0.0))
        end = float(seg.get("end_seconds", 0.0))
        if end <= start:
            logger.warning(f"Skipping segment with non-positive duration: {seg}")
            continue
        duration = end - start

        idx = int(seg.get("index", len(exported) + 1))
        title = _sanitize(str(seg.get("title", f"segment{idx}")))
        out_path = out_dir / f"{idx:02d}_{title}.mp4"

        cmd = [ffmpeg_bin(), "-y", "-loglevel", "error"]
        if reencode:
            # Accurate seek: -ss/-t AFTER -i decodes from the start and uses
            # output timestamps, giving frame-precise boundaries.
            cmd += [
                "-i", str(src),
                "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-c:a", "aac", "-movflags", "+faststart",
                str(out_path),
            ]
        else:
            cmd += [
                "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{duration:.3f}",
                "-c", "copy", "-avoid_negative_ts", "make_zero",
                str(out_path),
            ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(f"ffmpeg failed for segment {idx}: {result.stderr.strip()}")
            continue
        exported.append(str(out_path))
        logger.info(f"Exported {out_path.name} ({start:.2f}s → {end:.2f}s, {duration:.2f}s)")

    return exported
