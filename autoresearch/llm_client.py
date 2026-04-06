"""
Shared LLM provider adapter for autoresearch loops.
"""

from __future__ import annotations

import os
from typing import Optional


DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-5-mini",
}


class LlmProviderError(RuntimeError):
    """Raised when an LLM provider cannot be used."""


def normalize_provider(provider: Optional[str]) -> str:
    normalized = str(provider or "gemini").strip().lower()
    if normalized == "codex":
        return "openai"
    if normalized not in DEFAULT_MODELS:
        raise LlmProviderError(
            f"Unsupported provider '{provider}'. Expected one of: gemini, openai, codex."
        )
    return normalized


def default_model_for_provider(provider: Optional[str]) -> str:
    return DEFAULT_MODELS[normalize_provider(provider)]


def infer_provider_from_model(model: Optional[str], fallback: str = "gemini") -> str:
    model_name = str(model or "").strip().lower()
    if not model_name:
        return normalize_provider(fallback)
    if model_name.startswith("gemini"):
        return "gemini"
    if (
        model_name.startswith("gpt")
        or model_name.startswith("o")
        or "codex" in model_name
        or "openai" in model_name
    ):
        return "openai"
    return normalize_provider(fallback)


def _require_env(var_name: str, provider_name: str) -> str:
    value = str(os.getenv(var_name, "")).strip()
    if value:
        return value
    raise LlmProviderError(
        f"Missing {var_name} for {provider_name}. Put it in the repo root .env or export it in your shell."
    )


def _extract_openai_text(response) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            text_value = getattr(item, "text", None)
            if text_value:
                parts.append(text_value)
                continue
            if isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
        return "".join(parts)
    return str(content or "")


def _extract_gemini_text(response) -> str:
    text = getattr(response, "text", None)
    if text:
        return str(text)
    candidates = getattr(response, "candidates", None) or []
    parts = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        blocks = getattr(content, "parts", None) or []
        for block in blocks:
            block_text = getattr(block, "text", None)
            if block_text:
                parts.append(str(block_text))
    return "".join(parts)


def _generate_via_gemini(
    model: str,
    system_prompt: str,
    user_prompt: str,
    json_mode: bool,
) -> str:
    api_key = _require_env("GOOGLE_API_KEY", "Gemini")
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise LlmProviderError("Gemini SDK not installed. Run: pip install google-genai") from exc

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type="application/json" if json_mode else "text/plain",
    )
    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=config,
    )
    text = _extract_gemini_text(response).strip()
    if not text:
        raise LlmProviderError("Gemini returned empty content.")
    return text


def _generate_via_openai(
    model: str,
    system_prompt: str,
    user_prompt: str,
    json_mode: bool,
) -> str:
    api_key = _require_env("OPENAI_API_KEY", "OpenAI")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise LlmProviderError("OpenAI SDK not installed. Run: pip install openai") from exc

    client = OpenAI(api_key=api_key)
    request = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if json_mode:
        request["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**request)
    text = _extract_openai_text(response).strip()
    if not text:
        raise LlmProviderError("OpenAI returned empty content.")
    return text


def generate_text(
    *,
    provider: Optional[str],
    model: Optional[str],
    system_prompt: str,
    user_prompt: str,
    json_mode: bool = False,
) -> str:
    normalized_provider = normalize_provider(provider or infer_provider_from_model(model))
    resolved_model = str(model or default_model_for_provider(normalized_provider)).strip()
    if not resolved_model:
        raise LlmProviderError("Model name resolved to an empty string.")
    if normalized_provider == "gemini":
        return _generate_via_gemini(
            model=resolved_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_mode=json_mode,
        )
    if normalized_provider == "openai":
        return _generate_via_openai(
            model=resolved_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_mode=json_mode,
        )
    raise LlmProviderError(f"Unsupported provider '{normalized_provider}'.")
