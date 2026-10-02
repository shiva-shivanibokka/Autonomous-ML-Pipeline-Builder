"""
sandbox.executor — E2B cloud sandbox execution with self-correction loop.

The self-correction loop:
  1. Execute code in E2B (or subprocess fallback)
  2. If execution fails, extract the traceback
  3. Ask the LLM to fix the code based on the error
  4. Retry with the fixed code (max `max_retries` attempts)

E2B gives us a real isolated Python environment where we can:
  - Install packages (pip install ...)
  - Run generated code without it touching this host
  - Upload CSV files
  - Download outputs (transformed CSVs, plots)

Falls back to a local subprocess if E2B_API_KEY is not set. That runs the
generated code on this host, so it is gated behind ALLOW_LOCAL_EXEC and is for
local development only.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from core.config import settings
from core.llm_utils import extract_content, strip_fences

logger = logging.getLogger(__name__)


# ── Self-correction prompt ────────────────────────────────────────────────────

CORRECTION_SYSTEM_PROMPT = (
    "You are an expert Python debugger. "
    "Given a Python script and the error it produced, fix the bug. "
    "Return ONLY the corrected Python code — no explanation, no fences. "
    "Make the minimal change needed to fix the error. "
    "Do not change the overall structure or logic of the script."
)


def _ask_llm_to_fix(code: str, error: str, llm: Any) -> str:
    """Ask the LLM to fix a Python script given its error output."""
    user_prompt = (
        f"This Python script failed with the following error:\n\n"
        f"```\n{error[:1500]}\n```\n\n"
        f"Here is the script:\n\n"
        f"```python\n{code[:3000]}\n```\n\n"
        "Fix the script. Return ONLY the corrected Python code."
    )
    response = llm.invoke(
        [
            SystemMessage(content=CORRECTION_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
    )
    return strip_fences(extract_content(response))


# ── E2B execution ─────────────────────────────────────────────────────────────


def _format_execution_error(err: Any) -> str:
    """
    Render an E2B ExecutionError as text.

    `Execution.error` is a single object or None — not a list. This used to be
    iterated as `for e in (execution.error or [])`, which is empty-and-harmless
    when nothing went wrong and a TypeError the moment something did: the one
    path the self-correction loop exists to handle was the one that crashed.
    """
    if not err:
        return ""
    name = getattr(err, "name", "") or ""
    value = getattr(err, "value", "") or ""
    traceback = getattr(err, "traceback", "") or ""
    headline = ": ".join(p for p in (str(name), str(value)) if p)
    return f"{headline}\n{traceback}".strip() if traceback else (headline or str(err))


def _execute_e2b(code: str, csv_path: str, timeout: int) -> dict:
    """Execute code in an E2B cloud sandbox."""
    try:
        from e2b_code_interpreter import Sandbox  # type: ignore
    except ImportError:
        raise RuntimeError(
            "e2b-code-interpreter not installed. Run: pip install -r requirements.txt"
        )

    sbx = Sandbox(api_key=settings.e2b_api_key, timeout=timeout)
    try:
        # Install ML dependencies
        sbx.commands.run(
            "pip install pandas numpy scikit-learn lightgbm xgboost -q",
            timeout=60,
        )

        # Upload CSV
        if csv_path and Path(csv_path).exists():
            with open(csv_path, "rb") as f:
                sbx.files.write("/data/input.csv", f)

        # Execute
        execution = sbx.run_code(code)
        stdout = "\n".join(str(o) for o in execution.logs.stdout)
        stderr = "\n".join(str(e) for e in execution.logs.stderr)
        error = _format_execution_error(execution.error)

        # Try to download output CSV
        output_csv_path = ""
        try:
            content = sbx.files.read("/data/processed.csv")
            with tempfile.NamedTemporaryFile(
                suffix=".csv", delete=False, mode="wb"
            ) as f:
                f.write(content if isinstance(content, bytes) else content.encode())
                output_csv_path = f.name
        except Exception as exc:
            # Not an error on its own: plenty of generated scripts legitimately
            # write nothing. Log it, because when the script *was* supposed to
            # emit a CSV this is the only place that would say why.
            logger.info("no processed.csv returned from the sandbox: %s", exc)

        failed = bool(error) or bool(stderr and "Error" in stderr)
        return {
            "success": not failed,
            "stdout": stdout,
            "stderr": stderr,
            "error_text": error or stderr,
            # No fallback to the raw input: an empty path means "nothing was
            # written", which execute_with_retry turns into a loud failure when
            # the caller requires an output.
            "output_csv_path": output_csv_path,
        }
    finally:
        sbx.kill()


# ── Subprocess fallback ───────────────────────────────────────────────────────


def _rewrite_path_constant(code: str, name: str, value: str) -> str:
    """
    Point a generated script's path constant at a real local file.

    Matched by regex rather than by an exact string. This used to be
    `code.replace("INPUT_CSV_PATH = '/data/input.csv'", ...)`, which missed the
    moment the model wrote double quotes or spaced the assignment differently.
    The substitution then silently did nothing, the script went looking for
    /data/input.csv on the host, and the self-correction loop spent three LLM
    calls trying to fix a bug that was ours.
    """
    pattern = re.compile(rf"^{re.escape(name)}\s*=\s*.+$", re.MULTILINE)
    replacement = f"{name} = {value!r}"
    new_code, n = pattern.subn(lambda _: replacement, code, count=1)
    if n == 0:
        # The script never declared it — prepend so the name at least exists.
        return f"{replacement}\n{code}"
    return new_code


def _output_csv_for(csv_path: str) -> str:
    """
    Where a generated script should write its processed CSV.

    Derived with pathlib rather than `csv_path.replace(".csv", "_processed.csv")`,
    which rewrote every occurrence and so mangled any path with ".csv" in a
    parent directory. Empty input is legitimate — some generated scripts take no
    CSV at all — and must not raise.
    """
    if not csv_path:
        return ""
    p = Path(csv_path)
    return str(p.with_name(f"{p.stem}_processed{p.suffix or '.csv'}"))


def _execute_subprocess(code: str, csv_path: str, timeout: int) -> dict:
    """Execute code in a local subprocess with timeout (fallback for local dev)."""
    output_csv = _output_csv_for(csv_path)
    code_with_path = _rewrite_path_constant(code, "INPUT_CSV_PATH", csv_path)
    code_with_path = _rewrite_path_constant(
        code_with_path, "OUTPUT_CSV_PATH", output_csv
    )

    with tempfile.NamedTemporaryFile(
        suffix=".py", mode="w", delete=False, encoding="utf-8"
    ) as f:
        f.write(code_with_path)
        script_path = f.name

    # A stale output from an earlier attempt must not count as this attempt's.
    if output_csv and Path(output_csv).exists():
        os.unlink(output_csv)

    try:
        result = subprocess.run(
            # sys.executable, not "python": inside a venv, a container, or on a
            # Linux box where only python3 is on PATH, "python" is either the
            # wrong interpreter or missing entirely.
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        success = result.returncode == 0
        return {
            "success": success,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "error_text": result.stderr if not success else "",
            # No silent fallback to the raw CSV (see execute_with_retry).
            "output_csv_path": (
                output_csv if output_csv and Path(output_csv).exists() else ""
            ),
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Subprocess timed out after {timeout}s",
            "error_text": f"Timeout after {timeout}s",
            "output_csv_path": "",
        }
    finally:
        os.unlink(script_path)


# ── Public API ────────────────────────────────────────────────────────────────


def execute_with_retry(
    code: str,
    csv_path: str,
    llm: Any,
    max_retries: int = 3,
    timeout: int | None = None,
    require_output: bool = False,
) -> dict:
    """
    Execute code in a sandbox with self-correction on failure.

    Args:
        code:        Python code string to execute.
        csv_path:    Local path to the CSV (uploaded to sandbox).
        llm:         LangChain LLM instance for self-correction.
        max_retries: Maximum number of correction attempts.
        timeout:     Execution timeout in seconds.
        require_output: If True, a run that exits cleanly but writes no
                     OUTPUT_CSV_PATH counts as a failure (and is fed to the
                     self-correction loop). The feature engineer sets this:
                     it used to fall back silently to the raw CSV, so a
                     script that did nothing looked like a success.

    Returns:
        Dict with keys:
            success (bool), stdout (str), final_code (str),
            attempts (int), last_error (str), output_csv_path (str)
    """
    _timeout = timeout or settings.sandbox_timeout_seconds
    use_e2b = settings.execution_backend == "e2b" and bool(settings.e2b_api_key.strip())

    # Security gate: never run LLM-generated code on the host unless the operator
    # has explicitly opted in via ALLOW_LOCAL_EXEC (local dev only). In production
    # this stays off, so a missing E2B key fails loudly instead of silently running
    # arbitrary code on the server.
    if not use_e2b and not settings.allow_local_exec:
        raise RuntimeError(
            "Refusing to execute generated code: E2B sandbox is not configured and "
            "host execution is disabled. Set E2B_API_KEY (recommended), or set "
            "ALLOW_LOCAL_EXEC=true for local development only."
        )

    current_code = code
    last_error = ""

    for attempt in range(1, max_retries + 1):
        logger.info("Sandbox execution attempt %d/%d", attempt, max_retries)

        try:
            if use_e2b:
                result = _execute_e2b(current_code, csv_path, _timeout)
            else:
                result = _execute_subprocess(current_code, csv_path, _timeout)
        except Exception as exc:
            result = {
                "success": False,
                "stdout": "",
                "stderr": str(exc),
                "error_text": str(exc),
                "output_csv_path": "",
            }

        if result["success"] and require_output and not result.get("output_csv_path"):
            result = {
                **result,
                "success": False,
                "error_text": (
                    "The script exited without error but did not write its output "
                    "CSV to OUTPUT_CSV_PATH. It must save the transformed DataFrame "
                    "with df.to_csv(OUTPUT_CSV_PATH, index=False)."
                ),
            }

        if result["success"]:
            return {
                "success": True,
                "stdout": result["stdout"],
                "final_code": current_code,
                "attempts": attempt,
                "last_error": "",
                "output_csv_path": result["output_csv_path"],
            }

        last_error = result["error_text"] or result["stderr"]
        logger.warning("Attempt %d failed: %s", attempt, last_error[:200])

        if attempt < max_retries:
            logger.info("Asking LLM to fix the code...")
            try:
                current_code = _ask_llm_to_fix(current_code, last_error, llm)
            except Exception as fix_exc:
                logger.error("LLM fix failed: %s", fix_exc)
                break

    return {
        "success": False,
        "stdout": "",
        "final_code": current_code,
        "attempts": max_retries,
        "last_error": last_error,
        "output_csv_path": "",
    }
