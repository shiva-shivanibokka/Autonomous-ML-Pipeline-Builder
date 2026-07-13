"""
core.providers — unified LLM provider factory.

Replaces the copy-pasted `_get_llm()` / `PROVIDERS` dict that appears in 6+
repos across this portfolio. One place to add a new provider; every agent
module imports from here.

Supports:
  - Anthropic Claude  (claude-sonnet-5, claude-opus-4-8, claude-haiku-4-5)
  - OpenAI GPT        (gpt-4o, gpt-4o-mini)
  - Groq              (llama-3.3-70b, llama-3.1-8b)

Usage:
    from core.providers import get_llm, PROVIDER_MODELS

    llm = get_llm("anthropic", api_key="sk-ant-...", model="claude-haiku-4-5-20251001")
    response = llm.invoke("Hello")
"""

from __future__ import annotations

from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

# ── Provider model catalogue ───────────────────────────────────────────────────
PROVIDER_MODELS: dict[str, list[str]] = {
    "anthropic": [
        "claude-sonnet-5",
        "claude-opus-4-8",
        "claude-haiku-4-5-20251001",
    ],
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
    ],
    "groq": [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
    ],
}

# Default model for each provider — used when no model is specified
PROVIDER_DEFAULTS: dict[str, str] = {
    "anthropic": "claude-sonnet-5",
    "openai": "gpt-4o",
    "groq": "llama-3.3-70b-versatile",
}

# Models that are cost-efficient for high-volume code generation tasks
CODEGEN_MODELS: dict[str, str] = {
    "anthropic": "claude-haiku-4-5-20251001",
    "openai": "gpt-4o-mini",
    "groq": "llama-3.1-8b-instant",
}

# Groq uses the OpenAI-compatible SDK with a custom base URL
_GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def get_llm(
    provider: str,
    api_key: str,
    model: str = "",
    temperature: float = 0.0,
    max_tokens: int = 4096,
    **kwargs: Any,
) -> ChatAnthropic | ChatOpenAI:
    """
    Return a LangChain chat model for the given provider.

    Args:
        provider:    One of "anthropic", "openai", "groq".
        api_key:     API key for the provider.
        model:       Model name. If empty, uses the provider default.
        temperature: Sampling temperature (0.0 = deterministic).
        max_tokens:  Maximum tokens in the completion.
        **kwargs:    Additional kwargs forwarded to the model constructor.

    Returns:
        A LangChain BaseChatModel instance ready for .invoke() / .ainvoke().

    Raises:
        ValueError: If provider is not recognised.
    """
    provider = provider.lower().strip()

    if not model:
        model = PROVIDER_DEFAULTS.get(provider, "")

    if provider == "anthropic":
        return ChatAnthropic(
            api_key=api_key,  # type: ignore[arg-type]
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    if provider == "openai":
        return ChatOpenAI(
            api_key=api_key,  # type: ignore[arg-type]
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    if provider == "groq":
        return ChatOpenAI(
            api_key=api_key,  # type: ignore[arg-type]
            base_url=_GROQ_BASE_URL,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    raise ValueError(
        f"Unknown provider '{provider}'. "
        f"Supported providers: {list(PROVIDER_MODELS.keys())}"
    )


def get_codegen_llm(
    provider: str,
    api_key: str,
    temperature: float = 0.1,
    max_tokens: int = 8192,
) -> ChatAnthropic | ChatOpenAI:
    """
    Return a cost-efficient LLM for high-volume code generation sub-agents.

    Uses the cheaper/faster model for each provider (Haiku, gpt-4o-mini, llama-8b).
    Code generation benefits from slightly non-zero temperature for variation on retry.
    """
    model = CODEGEN_MODELS.get(provider, PROVIDER_DEFAULTS.get(provider, ""))
    return get_llm(
        provider=provider,
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
