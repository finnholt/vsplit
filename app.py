"""vsplit — Streamlit single-page UI."""

from __future__ import annotations

import io
import json
import logging
import os
import zipfile
from pathlib import Path
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

from vsplit import whisper_extra
from vsplit.config import API_KEY_ENV_VARS, DEFAULT_PROVIDER, LLM_CONFIG
from vsplit.exporter import export_segments
from vsplit.splitter import SceneSplitter

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")

st.set_page_config(page_title="vsplit · AI 语义分镜", page_icon="✂️", layout="wide")

WORKDIR = Path(os.getenv("VSPLIT_WORKDIR", "./.vsplit_runs")).resolve()
WORKDIR.mkdir(parents=True, exist_ok=True)


# ─── Session state init ───────────────────────────────────────────────────────

st.session_state.setdefault("segments_data", None)
st.session_state.setdefault("video_local_path", None)
st.session_state.setdefault("srt_local_path", None)
st.session_state.setdefault("export_dir", None)
st.session_state.setdefault("exported_files", [])


# ─── Sidebar config ───────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ 配置")

    provider = st.selectbox(
        "Provider",
        options=list(LLM_CONFIG.keys()),
        index=list(LLM_CONFIG.keys()).index(DEFAULT_PROVIDER)
        if DEFAULT_PROVIDER in LLM_CONFIG else 0,
    )

    env_key_name = API_KEY_ENV_VARS.get(provider, "")
    env_key_value = os.getenv(env_key_name, "") if env_key_name else ""
    api_key = st.text_input(
        f"API Key ({env_key_name})",
        value=env_key_value,
        type="password",
        help="留空则用环境变量。",
    )

    base_url = st.text_input(
        "Base URL（可选）",
        value=os.getenv(f"{provider.upper()}_BASE_URL", ""),
        help="OpenAI-compatible 端点。留空使用默认。",
    )

    text_model = st.text_input(
        "文本分组模型",
        value=LLM_CONFIG[provider].get("default_model", ""),
        help="用于第二步：综合动作+转录，输出分镜片段。",
    )

    vl_model = st.text_input(
        "视觉模型 (VL)",
        value=LLM_CONFIG[provider].get("default_vl_model", ""),
        help="用于第一步：识别帧动作与场景切换。",
    )


# ─── Main panel ───────────────────────────────────────────────────────────────

st.title("✂️ vsplit · AI 语义分镜")
st.caption("上传视频，自动用视觉模型切分语义片段。")

uploaded_video = st.file_uploader(
    "📼 视频文件", type=["mp4", "mov", "mkv", "webm", "avi"]
)
if uploaded_video:
    run_dir = WORKDIR / Path(uploaded_video.name).stem
    run_dir.mkdir(parents=True, exist_ok=True)
    video_path = run_dir / uploaded_video.name
    if not video_path.exists() or video_path.stat().st_size != uploaded_video.size:
        video_path.write_bytes(uploaded_video.getbuffer())
    st.session_state["video_local_path"] = str(video_path)
    vid_l, _ = st.columns([1, 4])
    with vid_l:
        st.video(str(video_path))

whisper_model_size = "base"
whisper_lang = ""

if st.session_state["video_local_path"]:
    if whisper_extra.is_available():
        with st.expander("🎤 Whisper 转录设置（点击分析时自动生成，可选）"):
            whisper_model_size = st.selectbox(
                "Whisper 模型", ["tiny", "base", "small", "medium", "large"], index=1
            )
            whisper_lang = st.text_input("语言（留空自动检测）", value="")
            st.caption("点击下方「🚀 开始分镜分析」时会自动转录（如果之前没生成过）。")
    else:
        st.caption("（安装 `vsplit[whisper]` 可启用内置 Whisper 转录，进一步提升分镜质量。）")


# ─── Run analysis ─────────────────────────────────────────────────────────────

st.divider()

ready = bool(st.session_state["video_local_path"]) and bool(api_key or provider == "custom_openai")
run_clicked = st.button("🚀 开始分镜分析", disabled=not ready, type="primary")

if run_clicked:
    # Clear previous run results so the page doesn't show stale data
    st.session_state["segments_data"] = None
    st.session_state["exported_files"] = []
    st.session_state["export_dir"] = None
    try:
        # ── Auto-transcribe if [whisper] is installed and no SRT yet ──
        video_local = st.session_state["video_local_path"]
        run_dir = Path(video_local).parent
        expected_srt = run_dir / f"{Path(video_local).stem}.srt"

        if not st.session_state.get("srt_local_path"):
            if expected_srt.exists():
                st.session_state["srt_local_path"] = str(expected_srt)
                st.info(f"♻️ 复用已有转录：{expected_srt.name}")
            elif whisper_extra.is_available():
                with st.spinner(f"🎤 Whisper 转录中（model={whisper_model_size}）..."):
                    try:
                        srt_path = whisper_extra.transcribe(
                            video_local,
                            model_size=whisper_model_size,
                            language=whisper_lang or None,
                        )
                        st.session_state["srt_local_path"] = srt_path
                        st.success(f"已生成转录：{Path(srt_path).name}")
                    except Exception as e:
                        st.warning(f"转录失败，将无 transcript 融合继续：{e}")

        with st.spinner("正在抽帧 → 跑 VL → 文本分组..."):
            splitter = SceneSplitter(
                provider=provider,
                api_key=api_key or None,
                base_url=base_url or None,
                text_model=text_model or None,
                vl_model=vl_model or None,
            )
            result = splitter.analyze_with_vl(
                video_local,
                part_name="part01",
                transcript_path=st.session_state.get("srt_local_path"),
            )
            splitter.save_to_file(result, str(run_dir / "segments.json"))

        with st.spinner("正在用 ffmpeg 切片..."):
            clips_dir = run_dir / "clips"
            exported = export_segments(
                st.session_state["video_local_path"],
                result["segments"],
                str(clips_dir),
            )
            for seg, fp in zip(result["segments"], exported):
                seg["clip_path"] = fp

        st.session_state["segments_data"] = result
        st.session_state["export_dir"] = str(clips_dir)
        st.session_state["exported_files"] = exported
        st.success(
            f"✅ 共得到 {result['total_segments']} 个片段，已切出 {len(exported)} 个 mp4"
        )
    except Exception as e:
        st.error(f"分析失败：{e}")
        st.exception(e)


# ─── Results ──────────────────────────────────────────────────────────────────

if st.session_state.get("segments_data"):
    data = st.session_state["segments_data"]
    st.subheader(f"📊 分镜结果（{data['total_segments']} 段，已切片）")

    rows = [
        {
            "#": s["index"],
            "开始": s["start_time"],
            "结束": s["end_time"],
            "时长 (秒)": s["duration_seconds"],
            "标题": s["title"],
            "依据": s.get("boundary_reason", ""),
            "文件": Path(s["clip_path"]).name if s.get("clip_path") else "—",
        }
        for s in data["segments"]
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.markdown("#### 🎬 全部片段预览")
    segments = data["segments"]
    cols_per_row = 4
    for row_start in range(0, len(segments), cols_per_row):
        row = segments[row_start:row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, seg in zip(cols, row):
            with col:
                clip_path = seg.get("clip_path")
                clip_bytes = None
                if clip_path and Path(clip_path).exists():
                    with open(clip_path, "rb") as f:
                        clip_bytes = f.read()
                    st.video(clip_bytes)
                else:
                    st.video(
                        st.session_state["video_local_path"],
                        start_time=int(seg["start_seconds"]),
                    )
                st.markdown(f"**#{seg['index']} · {seg['title']}**")
                st.caption(
                    f"{seg['start_time']} → {seg['end_time']}（{seg['duration_seconds']}s）"
                )
                if clip_bytes is not None and clip_path:
                    st.download_button(
                        "⬇️ 下载",
                        data=clip_bytes,
                        file_name=Path(clip_path).name,
                        mime="video/mp4",
                        key=f"dl_{seg['index']}",
                        use_container_width=True,
                    )

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button(
            "⬇️ 下载 segments.json",
            data=json.dumps(data, ensure_ascii=False, indent=2),
            file_name="segments.json",
            mime="application/json",
        )
    with col_b:
        if st.session_state.get("exported_files"):
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
                for fp in st.session_state["exported_files"]:
                    zf.write(fp, arcname=Path(fp).name)
            st.download_button(
                "📦 打包下载全部 mp4 (zip)",
                data=buf.getvalue(),
                file_name="clips.zip",
                mime="application/zip",
            )
