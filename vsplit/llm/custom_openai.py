"""Generic OpenAI-compatible client (supports text + optional vision)."""

import base64
import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests

from vsplit.config import API_KEY_ENV_VARS, LLM_CONFIG, _normalize_chat_completions_url

logger = logging.getLogger(__name__)


class CustomOpenAIAPIClient:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or os.getenv(API_KEY_ENV_VARS["custom_openai"])
        resolved = (
            base_url
            or os.getenv("CUSTOM_OPENAI_BASE_URL")
            or LLM_CONFIG["custom_openai"]["base_url"]
        )
        self.base_url = _normalize_chat_completions_url(resolved)
        self.default_model = (
            os.getenv("CUSTOM_OPENAI_MODEL") or LLM_CONFIG["custom_openai"]["default_model"]
        )
        if not self.base_url:
            raise ValueError("CUSTOM_OPENAI_BASE_URL is required for custom_openai provider.")

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                resp = requests.post(self.base_url, headers=headers, json=payload, timeout=240)
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.Timeout:
                if attempt < max_attempts:
                    logger.warning(f"Custom OpenAI timeout (attempt {attempt}), retrying...")
                    continue
                raise
            except requests.exceptions.HTTPError as e:
                status = resp.status_code if resp is not None else 0
                if status in (429, 500, 502, 503) and attempt < max_attempts:
                    wait = 3 * attempt
                    logger.warning(f"Custom OpenAI {status}, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                try:
                    detail = resp.json()
                except Exception:
                    detail = resp.text if resp is not None else ""
                raise Exception(f"Custom OpenAI failed: {e}\nResponse: {detail}") from e

    @staticmethod
    def _extract_content(message: Dict[str, Any]) -> str:
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                elif item is not None:
                    parts.append(str(item))
            return "".join(parts)
        return ""

    def simple_chat(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        model = model or self.default_model
        if not model:
            raise ValueError("Model required. Set CUSTOM_OPENAI_MODEL or pass model.")
        params = LLM_CONFIG["custom_openai"]["default_params"]
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": params["max_tokens"],
            "temperature": temperature if temperature is not None else params["temperature"],
            "top_p": params["top_p"],
            "stream": params["stream"],
        }
        response = self._post(payload)
        try:
            return self._extract_content(response["choices"][0]["message"])
        except (KeyError, IndexError) as e:
            raise Exception(f"Unexpected response: {response}") from e

    def chat_with_images(
        self,
        prompt: str,
        image_paths: List[str],
        model: Optional[str] = None,
    ) -> str:
        """Vision call. Works with any OpenAI-compatible vision model (gpt-4o, qwen-vl, etc.)."""
        model = model or os.getenv("CUSTOM_OPENAI_VL_MODEL") or self.default_model
        if not model:
            raise ValueError("VL model required. Set CUSTOM_OPENAI_VL_MODEL or pass model.")
        content: List[Dict[str, Any]] = []
        for path in image_paths:
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        content.append({"type": "text", "text": prompt})
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 2048,
        }
        response = self._post(payload)
        try:
            return self._extract_content(response["choices"][0]["message"])
        except (KeyError, IndexError) as e:
            raise Exception(f"Unexpected VL response: {response}") from e
