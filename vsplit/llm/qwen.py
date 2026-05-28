"""Qwen API client — text + VL (qwen3-vl-flash by default)."""

import base64
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from vsplit.config import API_KEY_ENV_VARS, LLM_CONFIG, _normalize_chat_completions_url

logger = logging.getLogger(__name__)


@dataclass
class QwenMessage:
    role: str
    content: str


class QwenAPIClient:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or os.getenv(API_KEY_ENV_VARS["qwen"])
        resolved = base_url or os.getenv("QWEN_BASE_URL") or LLM_CONFIG["qwen"]["base_url"]
        self.base_url = _normalize_chat_completions_url(resolved)
        self.legacy_base_url = LLM_CONFIG["qwen"]["legacy_base_url"]
        self.legacy_models = LLM_CONFIG["qwen"]["legacy_models"]
        self.default_model = LLM_CONFIG["qwen"]["default_model"]
        self.default_vl_model = LLM_CONFIG["qwen"]["default_vl_model"]
        if not self.api_key:
            raise ValueError(
                f"API key required. Set {API_KEY_ENV_VARS['qwen']} env var or pass api_key."
            )

    def _is_legacy(self, model: str) -> bool:
        return model in self.legacy_models

    def _post(self, payload: Dict[str, Any], model: str) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = self.legacy_base_url if self._is_legacy(model) else self.base_url
        max_attempts = 2
        for attempt in range(1, max_attempts + 1):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=240)
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.Timeout:
                if attempt < max_attempts:
                    logger.warning(f"Qwen timeout (attempt {attempt}), retrying...")
                    continue
                raise
            except requests.exceptions.HTTPError as e:
                status = resp.status_code if resp is not None else 0
                if status >= 500 and attempt < max_attempts:
                    wait = 3 * attempt
                    logger.warning(f"Qwen 5xx ({status}), waiting {wait}s before retry...")
                    time.sleep(wait)
                    continue
                detail = ""
                try:
                    detail = resp.json()
                except Exception:
                    detail = resp.text if resp is not None else ""
                raise Exception(f"Qwen API failed: {e}\nResponse: {detail}") from e

    def simple_chat(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        model = model or self.default_model
        params = LLM_CONFIG["qwen"]["default_params"]
        if self._is_legacy(model):
            payload = {
                "model": model,
                "input": {"messages": [{"role": "user", "content": prompt}]},
                "parameters": {
                    "max_tokens": params["max_tokens"],
                    "temperature": temperature if temperature is not None else params["temperature"],
                    "top_p": params["top_p"],
                    "incremental_output": params["stream"],
                },
            }
        else:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": params["max_tokens"],
                "temperature": temperature if temperature is not None else params["temperature"],
                "top_p": params["top_p"],
                "stream": params["stream"],
            }
        response = self._post(payload, model)
        if "choices" in response:
            return response["choices"][0]["message"]["content"]
        if "output" in response:
            return response["output"]["text"]
        raise Exception(f"Unexpected Qwen response: {response}")

    def chat_with_images(
        self,
        prompt: str,
        image_paths: List[str],
        model: Optional[str] = None,
    ) -> str:
        model = model or self.default_vl_model
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
        response = self._post(payload, model)
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise Exception(f"Failed to parse VL response: {e}\nResponse: {response}") from e
