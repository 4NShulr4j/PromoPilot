"""Groq client helpers for PromoPilot.

GPT-OSS on Groq is a reasoning model. For more reliable JSON generation:
- send one user message (Groq's reasoning guidance recommends avoiding system prompts),
- set low reasoning effort and hide the reasoning channel,
- first request JSON Object Mode,
- if the provider rejects JSON mode with an empty-generation validation error,
  retry once in plain text mode and validate JSON locally in story_analyzer.py.

No paid services or automatic billing are used.
"""
from __future__ import annotations

import os
import re
from typing import Any

from dotenv import load_dotenv


def _setting(name: str, default: str = "") -> str:
    """Read a setting from environment/.env, then Streamlit Cloud secrets."""
    load_dotenv()
    value = os.getenv(name, "").strip()
    if value:
        return value
    try:
        import streamlit as st
        secret_value = st.secrets.get(name, default)
        return str(secret_value).strip() if secret_value is not None else default
    except Exception:
        # Streamlit is optional for CLI use and secrets may not be configured.
        return default


def get_groq_api_key() -> str:
    return _setting("GROQ_API_KEY")


def get_groq_client():
    api_key = get_groq_api_key()
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to .env for local use or configure "
            "GROQ_API_KEY in Streamlit app secrets for cloud deployment."
        )
    try:
        from groq import Groq
    except ImportError as exc:
        raise RuntimeError(
            "The Groq SDK is not installed. Run: pip install groq"
        ) from exc
    return Groq(api_key=api_key)


def get_model_name() -> str:
    return _setting("GROQ_MODEL", "openai/gpt-oss-20b")


def _is_json_mode_validation_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "json_validate_failed" in message
        or "failed to validate json" in message
        or "failed_generation" in message
    )


def _is_token_budget_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    message = str(exc).lower()
    return (
        status_code == 413
        or "tokens per minute" in message
        or ("requested" in message and "rate_limit_exceeded" in message)
    )


def request_json_object(
    *,
    system_prompt: str,
    user_prompt: str,
    max_completion_tokens: int = 1400,
) -> tuple[str, dict[str, Any]]:
    """Call Groq and return model text; the caller validates JSON locally."""
    client = get_groq_client()
    model = get_model_name()

    # Groq's current reasoning guidance recommends putting instructions in a
    # user message rather than a system message for reasoning models.
    combined_prompt = (
        system_prompt.strip()
        + "\n\n"
        + user_prompt.strip()
        + "\n\nOUTPUT CONTRACT: Return exactly one valid JSON object. "
          "Do not include Markdown fences, comments, or prose outside the JSON."
    )

    request_args: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": combined_prompt}],
        "response_format": {"type": "json_object"},
        "max_completion_tokens": max_completion_tokens,
    }

    # These parameters are supported by GPT-OSS on Groq. Low reasoning effort
    # leaves more of the completion budget for the actual JSON payload.
    if model.startswith("openai/gpt-oss"):
        request_args["reasoning_effort"] = os.getenv(
            "GROQ_REASONING_EFFORT", "low"
        ).strip()
        request_args["include_reasoning"] = False

    used_fallback = False
    response_mode = "json_object"

    try:
        response = client.chat.completions.create(**request_args)
    except Exception as exc:
        if _is_token_budget_error(exc):
            reduced = max(600, max_completion_tokens // 2)
            if reduced >= max_completion_tokens:
                raise
            request_args["max_completion_tokens"] = reduced
            try:
                response = client.chat.completions.create(**request_args)
                max_completion_tokens = reduced
            except Exception as retry_exc:
                if not _is_json_mode_validation_error(retry_exc):
                    raise
                # Continue below with the plain-text JSON fallback.
                used_fallback = True
            else:
                used_fallback = False
        elif _is_json_mode_validation_error(exc):
            used_fallback = True
        else:
            raise

        if used_fallback:
            # Some model/API combinations can return an empty failed_generation
            # in JSON mode. Retry once without provider-side JSON validation;
            # story_analyzer then performs json.loads + Pydantic validation.
            request_args.pop("response_format", None)
            request_args["max_completion_tokens"] = max_completion_tokens
            request_args["messages"] = [{
                "role": "user",
                "content": combined_prompt
                + "\n\nReturn a compact JSON object. Keep strings concise."
            }]
            response = client.chat.completions.create(**request_args)
            response_mode = "text_json_fallback"

    if not getattr(response, "choices", None):
        raise RuntimeError("Groq returned no completion choices.")

    choice = response.choices[0]
    content = choice.message.content
    if not content or not content.strip():
        reason = getattr(choice, "finish_reason", None)
        refusal = getattr(choice.message, "refusal", None)
        raise RuntimeError(
            "Groq returned empty message content. "
            f"finish_reason={reason!r}; refusal={refusal!r}. "
            "Try a smaller chunk or a non-reasoning model available on your free tier."
        )

    # Plain-text fallback occasionally adds Markdown fences. Strip only a
    # single outer JSON fence; original returned content is otherwise retained.
    cleaned = content.strip()
    if response_mode == "text_json_fallback":
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL | re.IGNORECASE)
        if fenced:
            cleaned = fenced.group(1).strip()

    if getattr(choice, "finish_reason", None) == "length":
        raise RuntimeError(
            "Groq stopped because max_completion_tokens was reached. "
            "Retry with a smaller chunk or a larger free-tier-safe completion budget."
        )

    usage: dict[str, Any] = {}
    if getattr(response, "usage", None) is not None:
        usage = {
            "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
            "completion_tokens": getattr(response.usage, "completion_tokens", None),
            "total_tokens": getattr(response.usage, "total_tokens", None),
        }

    metadata = {
        "provider": "groq",
        "model": model,
        "response_mode": response_mode,
        "reasoning_effort": request_args.get("reasoning_effort"),
        "include_reasoning": request_args.get("include_reasoning"),
        "finish_reason": getattr(choice, "finish_reason", None),
        "max_completion_tokens": request_args.get("max_completion_tokens"),
        "usage": usage,
        "json_mode_fallback_used": response_mode == "text_json_fallback",
    }
    return cleaned, metadata
