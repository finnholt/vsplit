# vsplit — AI 语义分镜

用视觉模型（默认 `qwen3-vl-flash`）+ 语音转录融合，把视频自动切成语义连贯的短片段。

- 主路径：抽帧 → VL 模型描述帧动作 → 文本模型综合"动作变化 + 话题切换"分组
- 回退路径：PySceneDetect 视觉硬切（VL 失败时）
- 输出：`segments.json` + 可一键导出 mp4

## 安装

### macOS / Linux

```bash
# 1. 装 uv（如果还没装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 装系统 ffmpeg / ffprobe
brew install ffmpeg              # macOS
# sudo apt install ffmpeg        # Ubuntu / Debian

# 3. 同步项目依赖
uv sync                          # 基础（足以本地 SRT + VL 分析）
uv sync --extra whisper          # 内置 Whisper 转写
uv sync --extra scenedetect      # PySceneDetect 视觉回退
```

### Windows（PowerShell）

```powershell
# 1. 装 uv（如果还没装）
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. 装系统 ffmpeg / ffprobe
#    选一种方式：
#    a) winget（Windows 10/11 自带）
winget install -e --id Gyan.FFmpeg
#    b) Chocolatey
# choco install ffmpeg
#    c) Scoop
# scoop install ffmpeg
#    装完后重启终端，运行 `ffmpeg -version` 确认在 PATH 里

# 3. 同步项目依赖
uv sync
uv sync --extra whisper
uv sync --extra scenedetect
```

> Windows 注意：如果 `winget` 装完后 `ffmpeg` 仍找不到，重启 PowerShell；或手动把它的 `bin/` 目录加到系统 PATH。

## 配置

复制 `.env.example` 为 `.env` 并填入：

```
QWEN_API_KEY=sk-xxx
# 可选：自定义 OpenAI-compatible endpoint
# CUSTOM_OPENAI_API_KEY=...
# CUSTOM_OPENAI_BASE_URL=https://...
```

Windows PowerShell 复制命令：

```powershell
Copy-Item .env.example .env
notepad .env
```

## 运行

```bash
uv run streamlit run app.py     # macOS / Linux / Windows 都一样
```

浏览器打开 `http://localhost:8501`。

页面操作：
1. 上传视频
2. （可选）上传 `.srt` 字幕；没字幕但装了 `[whisper]` 可点 "生成 SRT"
3. 侧边栏选 provider / 填 API key / 调整片段约束
4. 点 "🚀 开始分镜分析"
5. 看分段表 → 点片段预览 → 导出 mp4

## Python API

```python
from vsplit.splitter import SceneSplitter

splitter = SceneSplitter(
    provider="qwen",
    api_key="sk-xxx",            # 或 QWEN_API_KEY env
    vl_model="qwen3-vl-flash",
    text_model="qwen3.6-plus",
    max_segment_seconds=15,
    min_segment_seconds=3,
)

result = splitter.analyze_with_vl(
    "input.mp4",
    transcript_path="input.srt",  # 可选
)
print(result["segments"])
splitter.save_to_file(result, "segments.json")
```

## 项目结构

```
vsplit/
├── app.py                      Streamlit 单页 UI
├── vsplit/
│   ├── config.py               LLM endpoint / 默认模型
│   ├── splitter.py             核心：VL + transcript 融合
│   ├── srt.py                  SRT 解析
│   ├── exporter.py             ffmpeg 切 mp4
│   ├── ffmpeg_utils.py         ffmpeg/ffprobe 路径解析
│   ├── whisper_extra.py        可选 Whisper 转写
│   └── llm/
│       ├── qwen.py             阿里 DashScope（含 VL）
│       └── custom_openai.py    任意 OpenAI-compatible（含 VL）
└── tests/
```

## 已知限制

- VL 抽帧固定 1 fps；动作变化极快的视频可能漏切
- 片段时长上下限是 prompt 软约束，模型偶尔会越界
- `custom_openai` 走 VL 时需要 endpoint 真支持 vision；不支持就会回退到 PySceneDetect
