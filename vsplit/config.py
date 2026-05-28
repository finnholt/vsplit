"""Minimal config for vsplit LLM clients."""

import os
from typing import Dict, Any


def _normalize_chat_completions_url(url: str) -> str:
    """Accept either a full endpoint or an OpenAI-compatible API root."""
    normalized = url.rstrip("/")
    if normalized.endswith("/chat/completions") or normalized.endswith("/generation"):
        return normalized
    if normalized.endswith("/v1") or normalized.endswith("/v4") or normalized.endswith("/compatible-mode/v1"):
        return f"{normalized}/chat/completions"
    return normalized


def _env_base_url(provider: str, default: str) -> str:
    env_value = os.getenv(f"{provider.upper()}_BASE_URL")
    if not env_value:
        return default
    return _normalize_chat_completions_url(env_value)


def _env_model(provider: str, default: str) -> str:
    return os.getenv(f"{provider.upper()}_MODEL", default)


LLM_CONFIG: Dict[str, Dict[str, Any]] = {
    "qwen": {
        "base_url": _env_base_url("qwen", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"),
        "legacy_base_url": "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
        "default_model": _env_model("qwen", "qwen3.6-plus"),
        "default_vl_model": _env_model("qwen_vl", "qwen3-vl-flash"),
        "default_params": {
            "max_tokens": 16384,
            "temperature": 0.7,
            "top_p": 0.8,
            "stream": False,
        },
        "legacy_models": ["qwen-turbo", "qwen-plus", "qwen-max", "qwen-long"],
    },
    "custom_openai": {
        "base_url": _env_base_url("custom_openai", "https://api.openai.com/v1/chat/completions"),
        "default_model": _env_model("custom_openai", ""),
        "default_params": {
            "max_tokens": 8192,
            "temperature": 0.7,
            "top_p": 0.8,
            "stream": False,
        },
    },
}

API_KEY_ENV_VARS: Dict[str, str] = {
    "qwen": "QWEN_API_KEY",
    "custom_openai": "CUSTOM_OPENAI_API_KEY",
}

DEFAULT_PROVIDER: str = os.getenv("VSPLIT_DEFAULT_PROVIDER", "qwen")
