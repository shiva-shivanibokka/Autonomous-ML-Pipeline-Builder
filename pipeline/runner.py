"""
pipeline.runner — public entry point for running the ML pipeline.

Provides two interfaces:
  1. run_pipeline()  — blocking, returns final state dict
  2. stream_pipeline() — generator that yields log lines as they arrive

Both interfaces are used by the FastAPI backend and Gradio UI.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Callable, Generator, Optional

from agents.state import AgentState
from core.config import settings
from pipeline.graph import get_compiled_graph

logger = logging.getLogger(__name__)

# Root directory for all per-run artifacts. Each run gets its own subdirectory
# (outputs/<pipeline_id>) so concurrent runs never clobber each other's files.
ARTIFACTS_ROOT = Path("outputs")


def _build_initial_state(
    csv_path: str,
    business_problem: str,
    provider: str,
    api_key: str,
    model_name: str,
    pipeline_id: str | None = None,
) -> AgentState:
    """Construct the initial AgentState for a new pipeline run."""
    # Resolve API key: user-supplied takes priority, then fall back to env
    resolved_key = api_key.strip() or settings.get_api_key(provider)

    run_id = pipeline_id or str(uuid.uuid4())
    output_dir = ARTIFACTS_ROOT / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    return AgentState(
        csv_path=csv_path,
        business_problem=business_problem,
        provider=provider,
        api_key=resolved_key,
        model_name=model_name,
        pipeline_id=run_id,
        output_dir=str(output_dir),
        status="running",
        current_step="orchestrator",
        error=None,
        dataset_profile=None,
        feature_result=None,
        model_results={},
        evaluation_result=None,
        deployment_artifacts=None,
        logs=[],
    )


def run_pipeline(
    csv_path: str,
    business_problem: str,
    provider: str = "anthropic",
    api_key: str = "",
    model_name: str = "",
    pipeline_id: str | None = None,
) -> AgentState:
    """
    Run the full ML pipeline synchronously.

    Args:
        csv_path:         Local path to the uploaded CSV file.
        business_problem: Plain English description of the ML task.
        provider:         LLM provider ("anthropic", "openai", "groq").
        api_key:          API key (falls back to env var if empty).
        model_name:       Model name (falls back to provider default if empty).
        pipeline_id:      Reuse this run id (so the caller can find artifacts);
                          a new one is generated if omitted.

    Returns:
        The final AgentState after all agents have run.
    """
    from core.providers import PROVIDER_DEFAULTS

    if not model_name:
        model_name = PROVIDER_DEFAULTS.get(provider, "")

    initial_state = _build_initial_state(
        csv_path=csv_path,
        business_problem=business_problem,
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        pipeline_id=pipeline_id,
    )

    graph = get_compiled_graph()
    logger.info(
        "Starting pipeline %s [%s/%s]",
        initial_state["pipeline_id"],
        provider,
        model_name,
    )

    final_state = graph.invoke(initial_state)
    return final_state


def run_pipeline_streaming(
    csv_path: str,
    business_problem: str,
    provider: str = "anthropic",
    api_key: str = "",
    model_name: str = "",
    pipeline_id: str | None = None,
    on_update: Optional[Callable[[AgentState], None]] = None,
) -> AgentState:
    """
    Run the pipeline, invoking `on_update(state)` after each agent completes so the
    caller can persist live progress, and returning the final state.

    This is the API's entry point: it gets both incremental logs (for the frontend's
    live view) and the final state (for the result), which the plain generator can't
    provide together.
    """
    from core.providers import PROVIDER_DEFAULTS

    if not model_name:
        model_name = PROVIDER_DEFAULTS.get(provider, "")

    initial_state = _build_initial_state(
        csv_path=csv_path,
        business_problem=business_problem,
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        pipeline_id=pipeline_id,
    )

    graph = get_compiled_graph()
    final_state: AgentState = initial_state
    for chunk in graph.stream(initial_state, stream_mode="values"):
        final_state = chunk
        if on_update is not None:
            try:
                on_update(chunk)
            except Exception:  # persistence must never crash the pipeline
                logger.warning("on_update callback failed", exc_info=True)
    return final_state


def stream_pipeline(
    csv_path: str,
    business_problem: str,
    provider: str = "anthropic",
    api_key: str = "",
    model_name: str = "",
) -> Generator[str, None, None]:
    """
    Run the pipeline and yield log lines as each agent completes.

    Used by the Gradio UI to stream output in real time.

    Yields:
        Individual log line strings as agents complete their steps.
        Yields a final summary line when the pipeline finishes.
    """
    from core.providers import PROVIDER_DEFAULTS

    if not model_name:
        model_name = PROVIDER_DEFAULTS.get(provider, "")

    initial_state = _build_initial_state(
        csv_path=csv_path,
        business_problem=business_problem,
        provider=provider,
        api_key=api_key,
        model_name=model_name,
    )

    graph = get_compiled_graph()
    seen_log_count = 0

    # LangGraph .stream() yields partial state updates after each node
    for chunk in graph.stream(initial_state, stream_mode="values"):
        current_logs = chunk.get("logs", [])
        new_logs = current_logs[seen_log_count:]
        for line in new_logs:
            yield line
        seen_log_count = len(current_logs)

        # Yield error immediately if pipeline failed
        if chunk.get("error"):
            yield f"[PIPELINE FAILED] {chunk['error']}"
            return

    yield "[PIPELINE COMPLETE]"
