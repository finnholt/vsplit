"""SRT parsing helpers."""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def parse_srt_file(srt_path: str) -> List[Dict[str, Any]]:
    """Parse an SRT file into a list of {start_time, end_time, text} dicts."""
    entries: List[Dict[str, Any]] = []
    try:
        content = Path(srt_path).read_text(encoding="utf-8").strip()
        for block in content.split("\n\n"):
            lines = block.strip().split("\n")
            if len(lines) >= 3:
                m = re.match(
                    r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})",
                    lines[1],
                )
                if m:
                    entries.append({
                        "start_time": m.group(1),
                        "end_time": m.group(2),
                        "text": " ".join(lines[2:]),
                    })
    except Exception as e:
        logger.error(f"Error parsing SRT {srt_path}: {e}")
    return entries


def time_to_seconds(time_str: str) -> float:
    """Convert 'HH:MM:SS' or 'HH:MM:SS,mmm' to seconds."""
    if "," in time_str:
        time_part, ms = time_str.split(",")
        ms_val = int(ms)
    else:
        time_part = time_str
        ms_val = 0
    parts = time_part.split(":")
    h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
    return h * 3600 + m * 60 + s + ms_val / 1000.0


def seconds_to_hms(seconds: float) -> str:
    """Convert seconds to 'HH:MM:SS'."""
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"
