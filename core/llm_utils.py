"""
core.llm_utils — shared utilities for structured LLM output parsing.

Replaces the copy-pasted `_strip_fences()` / `_extract_content()` helpers
that appear in AutoGrader-Agent, Autonomous-Research-Report-Agent, and
LLM-Halucination-Detection.

Usage:
    from core.llm_utils import strip_fences, parse_structured_output
    from myagent.schemas import MySchema

    raw = llm.invoke(prompt).content
    data = parse_structured_output(raw, MySchema)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Type, TypeVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def strip_fences(text: str) -> str:
    """
    Remove markdown code fences from an LLM response.

    Handles all of:
        ```json\\n{...}\\n```
        ```python\\n...\\n```
        ```\\n...\\n```
        {... raw JSON without fences ...}

    Returns the inner content with leading/trailing whitespace stripped.
    """
    text = text.strip()

    # Match ```[optional-lang]\\n content \\n```
    fence_pattern = re.compile(r"^```(?:[a-zA-Z0-9_+-]*)?\n?(.*?)\n?```$", re.DOTALL)
    match = fence_pattern.match(text)
    if match:
        return match.group(1).strip()

    return text


def extract_json(text: str) -> str:
    """
    Extract the first valid JSON object or array from a string.

    Useful when the LLM adds prose before/after the JSON blob.
    Falls back to the original text if no JSON is found.
    """
    # Try to find a JSON object {...} or array [...]
    for pattern in (r"\{.*\}", r"\[.*\]"):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            candidate = match.group(0)
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                continue

    return text


def parse_structured_output(
    raw: str,
    schema: Type[T],
    *,
    strict: bool = False,
) -> T:
    """
    Parse a raw LLM string response into a Pydantic model.

    Pipeline:
        1. strip_fences() — remove markdown code fences
        2. extract_json() — find the JSON blob if prose surrounds it
        3. json.loads() — parse to dict
        4. schema(**data) — validate with Pydantic v2

    Args:
        raw:    The raw string from llm.invoke(...).content
        schema: A Pydantic BaseModel subclass to validate against
        strict: If True, re-raises ValidationError instead of returning None

    Returns:
        A validated instance of `schema`.

    Raises:
        ValueError: If JSON parsing fails.
        ValidationError: If Pydantic validation fails and strict=True.
    """
    cleaned = strip_fences(raw)
    cleaned = extract_json(cleaned)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM response could not be parsed as JSON.\n"
            f"Raw response (first 500 chars):\n{raw[:500]}\n"
            f"JSON error: {exc}"
        ) from exc

    try:
        return schema(**data)
    except ValidationError as exc:
        if strict:
            raise
        logger.warning(
            "Pydantic validation failed for schema %s: %s. "
            "Attempting field-by-field construction.",
            schema.__name__,
            exc,
        )
        raise


def safe_parse(
    raw: str,
    schema: Type[T],
    fallback: T | None = None,
) -> T | None:
    """
    Like parse_structured_output but returns `fallback` instead of raising.

    Useful in agent nodes where a bad LLM response should not crash the pipeline
    — the agent can detect `None` and retry.
    """
    try:
        return parse_structured_output(raw, schema)
    except (ValueError, ValidationError) as exc:
        logger.warning("safe_parse failed for %s: %s", schema.__name__, exc)
        return fallback


def extract_content(response: Any) -> str:
    """
    Extract the string content from various LLM response types.

    Handles:
        - LangChain AIMessage (.content attribute)
        - Raw string responses
        - Dict responses with "content" key
    """
    if hasattr(response, "content"):
        content = response.content
        # LangChain sometimes returns a list of content blocks (tool use)
        if isinstance(content, list):
            return " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        return str(content)

    if isinstance(response, dict):
        return str(response.get("content", response))

    return str(response)


def build_system_prompt(role: str, context: str = "") -> str:
    """
    Construct a clean system prompt for an agent node.

    Enforces JSON-only output to reduce parse failures.
    """
    base = (
        f"You are {role}.\n\n"
        "IMPORTANT: Respond ONLY with valid JSON. "
        "Do not include any prose, markdown fences, or explanation outside the JSON. "
        "Your entire response must be parseable by json.loads()."
    )
    if context:
        base += f"\n\nContext:\n{context}"
    return base
