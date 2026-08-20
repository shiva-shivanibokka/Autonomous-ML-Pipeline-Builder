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

    Scans candidate start positions with json.raw_decode rather than matching
    `\\{.*\\}` under DOTALL. That pattern is greedy: it spans from the first
    brace to the last one, so a closing sentence like "Note: set {b} later."
    swallowed the prose into the candidate and failed to parse. The orchestrator
    raises when its plan will not parse, so one stray brace failed the whole run.
    """
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch not in "{[":
            continue
        try:
            _, end = decoder.raw_decode(text, i)
        except ValueError:
            continue  # not the start of a well-formed value — keep scanning
        return text[i:end]

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
        strict: Kept for call-site compatibility; validation failures always
                raise. Use safe_parse() when a bad response should not stop
                the pipeline.

    Returns:
        A validated instance of `schema`.

    Raises:
        ValueError:      If JSON parsing fails.
        ValidationError: If Pydantic validation fails.
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
        # This used to log "Attempting field-by-field construction." and then
        # raise without attempting anything — a log line describing work that
        # did not happen, in the one place someone reads to find out why.
        logger.warning(
            "Pydantic validation failed for schema %s: %s", schema.__name__, exc
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
