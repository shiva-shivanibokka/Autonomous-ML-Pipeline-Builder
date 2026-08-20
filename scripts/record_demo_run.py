#!/usr/bin/env python
"""
Record one real pipeline run so the console can be shown without a backend.

The backend is a long-lived container that needs an LLM key and a paid sandbox,
so it is not hosted. This captures an actual run — its log stream, timing,
results and generated artifacts — into web/public/demo/, where the frontend
replays it as static files. Nothing is synthesised: if it is on the page, it
came out of this run.

    python scripts/record_demo_run.py --csv data/telco_churn.csv \\
        --problem "Predict which customers churn. The target column is Churn." \\
        --provider anthropic --api-key $ANTHROPIC_API_KEY

Then commit web/public/demo/ and deploy web/ to Vercel with
NEXT_PUBLIC_DEMO_MODE=1.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from core.providers import PROVIDER_DEFAULTS  # noqa: E402
from pipeline.runner import run_pipeline_streaming  # noqa: E402

DEMO_DIR = ROOT / "web" / "public" / "demo"
ARTIFACT_DIR = DEMO_DIR / "artifacts"

# Mirrors api.main._ALLOWED_ARTIFACTS — what a run is expected to produce.
ARTIFACTS = [
    "pipeline.py",
    "requirements.txt",
    "fastapi_endpoint.py",
    "Dockerfile",
    "openapi_spec.json",
    "shap_summary.png",
    "model.pkl",
    "feature_schema.json",
]

# Big enough to be interesting, small enough to serve from a static host.
MAX_ARTIFACT_MB = 20


def _result_payload(state: dict, logs: list[str]) -> dict:
    """Build the same shape /pipeline/{id}/result returns, from final state."""
    eval_result = state.get("evaluation_result") or {}
    winner = eval_result.get("winner_model", "")
    model_results = state.get("model_results") or {}
    artifacts = state.get("deployment_artifacts") or {}
    return {
        "pipeline_id": "demo",
        "status": state.get("status", "completed"),
        "winner_model": winner,
        "primary_metric": eval_result.get("primary_metric"),
        "metrics": (model_results.get(winner) or {}).get("metrics", {}),
        "justification": eval_result.get("justification"),
        "bias_warnings": eval_result.get("bias_warnings", []),
        "comparison_table": eval_result.get("comparison_table", []),
        "has_shap_plot": bool(eval_result.get("shap_plot_path")),
        "has_pipeline_code": bool(artifacts.get("pipeline_code")),
        "has_fastapi_endpoint": bool(artifacts.get("fastapi_code")),
        "has_dockerfile": bool(artifacts.get("dockerfile")),
        "logs": logs,
    }


def _copy_artifacts(output_dir: Path) -> list[str]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in ARTIFACTS:
        src = output_dir / name
        if not src.exists():
            print(f"  - {name}: not produced by this run")
            continue
        size_mb = src.stat().st_size / 1e6
        if size_mb > MAX_ARTIFACT_MB:
            print(f"  - {name}: {size_mb:.1f} MB, over the {MAX_ARTIFACT_MB} MB cap - skipped")
            continue
        shutil.copy2(src, ARTIFACT_DIR / name)
        copied.append(name)
        print(f"  + {name} ({size_mb:.2f} MB)")
    return copied


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True, help="Path to the CSV to run on")
    ap.add_argument("--problem", required=True, help="The plain-English goal")
    ap.add_argument("--provider", default="anthropic", choices=["anthropic", "openai", "groq"])
    ap.add_argument("--api-key", default="", help="Defaults to the provider's env var")
    ap.add_argument("--model", default="", help="Model id; provider default if omitted")
    args = ap.parse_args()

    api_key = args.api_key or os.environ.get(
        {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "groq": "GROQ_API_KEY",
        }[args.provider],
        "",
    )
    if not api_key:
        print(f"No API key for {args.provider}. Pass --api-key or set the env var.")
        return 2

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"No such CSV: {csv_path}")
        return 2

    # Check the execution gate up front. The Feature Engineer runs generated
    # code, and without a backend it raises — but only after the Orchestrator
    # and Data Analyst have already made their LLM calls. Failing here costs
    # nothing instead of two paid calls.
    from core.config import settings

    use_e2b = settings.execution_backend == "e2b" and bool(settings.e2b_api_key.strip())
    if not use_e2b and not settings.allow_local_exec:
        print(
            "Generated code has nowhere to run, and the pipeline needs it at the\n"
            "Feature Engineer step. Pick one:\n\n"
            "  Run it on this machine (simplest; it is your own code and repo,\n"
            "  but it does execute model-written Python locally):\n"
            "      PowerShell:  $env:ALLOW_LOCAL_EXEC='true'; "
            "$env:EXECUTION_BACKEND='subprocess'\n"
            "      bash:        export ALLOW_LOCAL_EXEC=true "
            "EXECUTION_BACKEND=subprocess\n\n"
            "  Or use an isolated cloud sandbox - set E2B_API_KEY "
            "(https://e2b.dev)."
        )
        return 2

    df = pd.read_csv(csv_path)
    pipeline_id = "demo"
    output_dir = ROOT / "outputs" / pipeline_id

    frames: list[dict] = []
    started = time.time()

    def on_update(state: dict) -> None:
        # One frame per progress callback: what the UI's poller would have seen.
        frames.append(
            {
                "t": round(time.time() - started, 2),
                "current_step": state.get("current_step", "orchestrator"),
                "status": state.get("status", "running"),
                "n_logs": len(state.get("logs", [])),
            }
        )
        step = frames[-1]["current_step"]
        print(f"  [{frames[-1]['t']:7.2f}s] {step}")

    print(f"Running the pipeline on {csv_path.name} via {args.provider}...")
    final_state = run_pipeline_streaming(
        csv_path=str(csv_path),
        business_problem=args.problem,
        provider=args.provider,
        api_key=api_key,
        model_name=args.model,
        pipeline_id=pipeline_id,
        on_update=on_update,
    )

    duration = round(time.time() - started, 2)
    logs = final_state.get("logs", [])
    status = final_state.get("status", "completed")

    if status != "completed":
        print(f"\nRun ended as '{status}': {final_state.get('error')}")
        print("Not recording a failed run - fix the cause and try again.")
        return 1

    # Close the timeline so the replay ends on the real final state.
    frames.append(
        {"t": duration, "current_step": "deployment_agent", "status": status, "n_logs": len(logs)}
    )

    print("\nCopying artifacts:")
    copied = _copy_artifacts(output_dir)

    run = {
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": {"name": csv_path.name, "n_rows": len(df), "n_cols": len(df.columns)},
        "business_problem": args.problem,
        "provider": args.provider,
        # Resolve the default rather than recording "<provider> default". The
        # banner names the model that produced the run, and "anthropic default"
        # tells a reader nothing and stops meaning the same thing the moment
        # the catalogue moves.
        "model_name": args.model or PROVIDER_DEFAULTS.get(args.provider, args.provider),
        "duration_seconds": duration,
        "frames": frames,
        "logs": logs,
        "result": _result_payload(final_state, logs),
        "artifacts": copied,
    }

    payload = json.dumps(run, indent=2, default=str)

    # This file gets committed to a public repo and served as a static asset.
    # Nothing here is supposed to carry the key, but "supposed to" is not a
    # guarantee worth publishing on: check the actual bytes before writing.
    if api_key and api_key in payload:
        print(
            "\nREFUSING TO WRITE: the API key appears in the recording.\n"
            "Something logged it. Fix that before recording again - this file\n"
            "is committed to a public repository."
        )
        return 1
    for artifact in ARTIFACT_DIR.glob("*"):
        if artifact.suffix in {".py", ".json", ".txt", ""} and api_key:
            try:
                if api_key in artifact.read_text(encoding="utf-8", errors="ignore"):
                    print(f"\nREFUSING TO WRITE: the API key appears in {artifact.name}.")
                    return 1
            except OSError:
                pass

    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    out = DEMO_DIR / "run.json"
    out.write_text(payload, encoding="utf-8")

    print(f"\nRecorded {len(frames)} frames and {len(logs)} log lines in {duration:.0f}s.")
    print(f"Wrote {out.relative_to(ROOT)}")
    print("\nNext: commit web/public/demo/, then deploy web/ to Vercel with")
    print("      NEXT_PUBLIC_DEMO_MODE=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
