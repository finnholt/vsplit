"""LLM client registry for vsplit."""

from typing import Optional


def get_client(provider: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
    provider = (provider or "qwen").lower()
    if provider == "qwen":
        from vsplit.llm.qwen import QwenAPIClient
        return QwenAPIClient(api_key=api_key, base_url=base_url)
    if provider == "custom_openai":
        from vsplit.llm.custom_openai import CustomOpenAIAPIClient
        return CustomOpenAIAPIClient(api_key=api_key, base_url=base_url)
    raise ValueError(f"Unsupported provider: {provider}")


SUPPORTED_PROVIDERS = ("qwen", "custom_openai")
