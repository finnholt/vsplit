"""Video scene splitting via VL + transcript fusion (with PySceneDetect fallback)."""

import json
import logging
import re
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from vsplit.ffmpeg_utils import ffmpeg_bin, ffprobe_bin
from vsplit.llm import get_client
from vsplit.srt import parse_srt_file, seconds_to_hms, time_to_seconds

logger = logging.getLogger(__name__)


class SceneSplitter:
    """Split a video into semantically coherent segments.

    Two paths:
      * ``analyze_with_vl`` — primary: extract frames → VL model labels actions →
        text model groups them, optionally fused with an SRT transcript.
      * ``analyze_visual_only`` — fallback: PySceneDetect ContentDetector.
    """

    def __init__(
        self,
        provider: str = "qwen",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        text_model: Optional[str] = None,
        vl_model: Optional[str] = None,
        threshold: float = 40.0,
        max_segment_seconds: float = 15.0,
        min_segment_seconds: float = 3.0,
    ):
        self.provider = provider.lower()
        self.text_model = (text_model or "").strip() or None
        self.vl_model = (vl_model or "").strip() or None
        self.threshold = threshold
        self.max_segment_seconds = max_segment_seconds
        self.min_segment_seconds = min_segment_seconds
        self.llm_client = get_client(self.provider, api_key=api_key, base_url=base_url)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_video_duration(self, video_path: str) -> float:
        try:
            result = subprocess.run(
                [
                    ffprobe_bin(), "-v", "error", "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1", video_path,
                ],
                capture_output=True, text=True,
            )
            return float(result.stdout.strip() or 0.0)
        except Exception as e:
            logger.warning(f"ffprobe failed for {video_path}: {e}")
            return 0.0

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
        fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fenced:
            return json.loads(fenced.group(1))
        loose = re.search(r"\{.*\}", text, re.DOTALL)
        if loose:
            return json.loads(loose.group(0))
        raise json.JSONDecodeError("No JSON object found", text, 0)

    def _detect_scene_cuts(self, video_path: str) -> List[float]:
        try:
            from scenedetect import SceneManager, open_video
            from scenedetect.detectors import ContentDetector
        except ImportError:
            logger.warning(
                "PySceneDetect not installed — install with `pip install vsplit[scenedetect]` "
                "to enable visual_only fallback. Returning no cuts."
            )
            return []

        try:
            video = open_video(video_path, framerate=None)
            scene_manager = SceneManager()
            scene_manager.add_detector(ContentDetector(threshold=self.threshold))
            scene_manager.detect_scenes(video, frame_skip=0, show_progress=False)
            scenes = scene_manager.get_scene_list()
            cuts = [scene[0].get_seconds() for scene in scenes[1:]]
            logger.info(
                f"🎬 PySceneDetect: {len(cuts)} cuts — "
                + ", ".join(f"{c:.1f}s" for c in cuts)
            )
            return cuts
        except Exception as e:
            logger.warning(f"PySceneDetect failed: {e}")
            return []

    def _groups_to_segments(self, groups: List[Dict], part_name: str) -> Dict[str, Any]:
        segments: List[Dict[str, Any]] = []
        for i, g in enumerate(groups):
            start_s = float(str(g.get("start", "0")).replace("s", "").strip())
            end_s = float(str(g.get("end", "0")).replace("s", "").strip())
            segments.append({
                "index": i + 1,
                "title": str(g.get("title", f"片段{i + 1}")),
                "start_time": seconds_to_hms(start_s),
                "end_time": seconds_to_hms(end_s),
                "start_seconds": start_s,
                "end_seconds": end_s,
                "duration_seconds": int(end_s - start_s),
                "boundary_reason": "vl grouping",
            })
        logger.info(f"✅ VL grouped {part_name}: {len(segments)} segments")
        return self._wrap(segments, part_name)

    def _wrap(self, segments: List[Dict[str, Any]], part_name: str) -> Dict[str, Any]:
        return {
            "video_part": part_name,
            "segments": segments,
            "total_segments": len(segments),
            "analysis_timestamp": datetime.now().isoformat() + "Z",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_with_vl(
        self,
        video_path: str,
        part_name: str = "part01",
        transcript_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Primary path: VL model + transcript fusion."""
        if not hasattr(self.llm_client, "chat_with_images"):
            logger.warning(
                f"Provider {self.provider} does not support VL — falling back to visual_only."
            )
            return self.analyze_visual_only(video_path, part_name)

        transcript_block = ""
        if transcript_path and Path(transcript_path).exists():
            entries = parse_srt_file(transcript_path)
            if entries:
                lines = []
                for e in entries:
                    start_s = time_to_seconds(e["start_time"])
                    end_s = time_to_seconds(e["end_time"])
                    text = (e.get("text") or "").strip()
                    if text:
                        lines.append(f"[{start_s:.1f}s-{end_s:.1f}s] {text}")
                transcript_block = "\n".join(lines)
                logger.info(
                    f"📝 Transcript fused into VL grouping for {part_name}: {len(lines)} lines"
                )

        logger.info(f"✂️  VL scene detection for {part_name}...")

        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir) / "frames"
            frames_dir.mkdir()

            subprocess.run(
                [
                    ffmpeg_bin(), "-i", video_path,
                    "-vf", "fps=1/1", "-q:v", "3",
                    f"{frames_dir}/frame_%04d.jpg",
                    "-y", "-loglevel", "error",
                ],
                check=True,
            )

            frame_files = sorted(frames_dir.glob("frame_*.jpg"))
            if not frame_files:
                logger.warning(f"No frames extracted from {part_name}, falling back")
                return self.analyze_visual_only(video_path, part_name)

            timestamps = [float(i) for i in range(len(frame_files))]
            logger.info(f"Extracted {len(frame_files)} frames from {part_name}")

            all_frames: List[Dict] = []
            all_vl_cuts: List[Dict] = []
            batch_size = 18

            for i in range(0, len(frame_files), batch_size):
                batch_files = [str(f) for f in frame_files[i:i + batch_size]]
                batch_ts = timestamps[i:i + len(batch_files)]
                ts_str = ", ".join(f"{t}s" for t in batch_ts)

                prompt = (
                    f"以下是视频按时间顺序截取的帧，时间戳依次为：{ts_str}。\n\n"
                    "请完成两项分析：\n\n"
                    "1. 场景切换检测：判断哪些相邻帧之间发生了场景切换。\n"
                    "   切换依据：背景变化、拍摄角度明显变化、人物完全替换。\n\n"
                    "2. 人物动作分析：描述每一帧中人物的主要动作或行为。\n\n"
                    "返回 JSON 格式：\n"
                    '{"cuts": [{"at": "12.0s", "reason": "背景变化"}], '
                    '"frames": [{"at": "0.0s", "action": "人物站立讲话"}]}'
                )

                try:
                    response = self.llm_client.chat_with_images(
                        prompt, batch_files, model=self.vl_model
                    )
                    data = self._extract_json(response)
                    all_frames.extend(data.get("frames", []))
                    all_vl_cuts.extend(data.get("cuts", []))
                    logger.info(
                        f"VL batch {i // batch_size + 1}: "
                        f"{len(data.get('cuts', []))} cuts, {len(data.get('frames', []))} frames"
                    )
                except Exception as e:
                    logger.warning(f"VL batch {i // batch_size + 1} failed: {e}")

            # Step 2: text model groups frames
            if all_frames:
                frames_text = "\n".join(
                    f"{f.get('at')}: {f.get('action', '')}" for f in all_frames
                )
                if transcript_block:
                    group_prompt = (
                        "以下是视频每帧的画面动作描述（按时间戳）：\n\n"
                        f"{frames_text}\n\n"
                        "以下是同一视频的语音转录（带时间戳，与上面共用同一时间轴）：\n\n"
                        f"{transcript_block}\n\n"
                        "请综合【画面动作变化】和【语音话题切换】两个信号，"
                        "将视频划分为语义连贯的片段。\n"
                        "切割依据（按优先级）：\n"
                        "1. 语音中的话题切换（新主题开始、明显的转折词如\"接下来/另外/我们再看\"等）\n"
                        "2. 人物行为目的改变、展示对象切换\n"
                        "3. 长停顿后的重新开始、动作节奏明显变化\n\n"
                        f"【硬约束】每个片段时长不得超过 {self.max_segment_seconds:.0f} 秒；"
                        f"不得短于 {self.min_segment_seconds:.0f} 秒。\n"
                        "超过上限：必须在内部寻找次要切割点拆分（句子边界或动作变化处）。\n"
                        "短于下限：合并到语义最相近的相邻片段。\n"
                        "切点应优先落在【语音句子结束】处，避免切断一句完整的话。\n\n"
                        "返回 JSON 格式（start/end 单位为秒）：\n"
                        '{"cuts": [{"at": "25.0s", "reason": "话题切换：从介绍转入演示"}], '
                        '"groups": [{"start": "0.0s", "end": "15.0s", "title": "开场介绍"}]}'
                    )
                else:
                    group_prompt = (
                        "以下是视频每帧的动作描述：\n\n"
                        f"{frames_text}\n\n"
                        "请将连续动作相似的帧归为一组，找出动作发生明显变化的时间点作为切割点。\n"
                        "判断依据：人物行为目的改变、展示对象切换、动作节奏明显变化。\n\n"
                        f"【重要限制】每个片段时长不得超过 {self.max_segment_seconds:.0f} 秒。\n"
                        "如果某组连续动作超过上限，必须在该组内寻找次要切割点进行拆分。\n\n"
                        "返回 JSON 格式：\n"
                        '{"cuts": [{"at": "25.0s", "reason": "动作变化"}], '
                        '"groups": [{"start": "0.0s", "end": "15.0s", "title": "展示产品"}]}'
                    )
                try:
                    group_response = self.llm_client.simple_chat(
                        group_prompt, model=self.text_model
                    )
                    group_data = self._extract_json(group_response)
                    groups = group_data.get("groups", [])
                    if groups:
                        return self._groups_to_segments(groups, part_name)
                except Exception as e:
                    logger.warning(f"Text grouping failed: {e}")

            # Fallback: raw VL cuts only
            cut_seconds: List[float] = []
            for c in all_vl_cuts:
                try:
                    cut_seconds.append(float(str(c.get("at", "0")).replace("s", "").strip()))
                except ValueError:
                    pass

            if not cut_seconds:
                logger.warning(f"No cuts from VL for {part_name}, falling back to visual_only")
                return self.analyze_visual_only(video_path, part_name)

            duration = self._get_video_duration(video_path)
            boundaries = [0.0] + sorted(set(cut_seconds)) + [duration]
            segments: List[Dict[str, Any]] = []
            for i in range(len(boundaries) - 1):
                start, end = boundaries[i], boundaries[i + 1]
                segments.append({
                    "index": i + 1,
                    "title": f"片段{i + 1}",
                    "start_time": seconds_to_hms(start),
                    "end_time": seconds_to_hms(end),
                    "start_seconds": start,
                    "end_seconds": end,
                    "duration_seconds": int(end - start),
                    "boundary_reason": "vl scene cut",
                })

            logger.info(f"✅ VL scene split {part_name}: {len(segments)} segments")
            return self._wrap(segments, part_name)

    def analyze_visual_only(
        self,
        video_path: str,
        part_name: str = "part01",
    ) -> Dict[str, Any]:
        """Fallback path: PySceneDetect only — no LLM, no transcript."""
        logger.info(
            f"✂️  Visual scene detection for {part_name} (threshold={self.threshold})..."
        )
        cuts = self._detect_scene_cuts(video_path)
        duration = self._get_video_duration(video_path)

        if not cuts:
            logger.info(f"No scene cuts detected in {part_name}, returning single segment")
            return self._wrap(
                [{
                    "index": 1,
                    "title": "片段1",
                    "start_time": "00:00:00",
                    "end_time": seconds_to_hms(duration),
                    "start_seconds": 0.0,
                    "end_seconds": duration,
                    "duration_seconds": int(duration),
                    "boundary_reason": "no scene cuts detected",
                }],
                part_name,
            )

        boundaries = [0.0] + cuts + [duration]
        segments = []
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            segments.append({
                "index": i + 1,
                "title": f"片段{i + 1}",
                "start_time": seconds_to_hms(start),
                "end_time": seconds_to_hms(end),
                "start_seconds": start,
                "end_seconds": end,
                "duration_seconds": int(end - start),
                "boundary_reason": "scene cut detected",
            })
        logger.info(f"✅ Visual scene split {part_name}: {len(segments)} segments")
        return self._wrap(segments, part_name)

    def save_to_file(self, data: Dict[str, Any], filepath: str) -> None:
        Path(filepath).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
