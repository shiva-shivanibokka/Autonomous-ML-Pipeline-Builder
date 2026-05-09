"""
ui.app — Gradio 5 frontend for the Autonomous ML Pipeline Builder.

Layout:
  Tab 1: Build Pipeline  — upload CSV, describe problem, run, watch live log
  Tab 2: Results         — model comparison table, winner metrics, SHAP plot
  Tab 3: Download        — download generated artifacts (pipeline.py, Dockerfile, etc.)
  Tab 4: About           — what this project does, how it works

The UI connects directly to the pipeline runner (no FastAPI needed for local use).
For the deployed HF Spaces version, it calls the FastAPI backend via HTTP.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import gradio as gr
import pandas as pd

from core.ui.components import BASE_CSS, make_provider_row, make_log_panel
from core.config import settings

# ── Determine backend mode ────────────────────────────────────────────────────
# Direct mode: Gradio calls pipeline.runner directly (local + HF Spaces)
# API mode: Gradio calls FastAPI backend over HTTP (full Docker deployment)
API_BASE = os.getenv("API_BASE_URL", "")
USE_API_MODE = bool(API_BASE.strip())

DEMO_CSV_URL = (
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/creditcard.csv"
)


# ── Pipeline runner (direct mode) ─────────────────────────────────────────────


def _run_pipeline_direct(
    csv_file,
    business_problem: str,
    provider: str,
    model: str,
    api_key: str,
):
    """Generator: runs pipeline directly and yields log lines for streaming."""
    if csv_file is None:
        yield "Please upload a CSV file first."
        return

    if not business_problem.strip():
        yield "Please describe the business problem."
        return

    resolved_key = api_key.strip() or settings.get_api_key(provider)
    if not resolved_key:
        yield f"No API key found for {provider}. Paste one in the API Key field or set it in .env."
        return

    from pipeline.runner import stream_pipeline

    csv_path = csv_file.name if hasattr(csv_file, "name") else str(csv_file)

    yield f"Starting pipeline for: {business_problem[:60]}..."
    yield f"Provider: {provider} / Model: {model}"
    yield "─" * 60

    for line in stream_pipeline(
        csv_path=csv_path,
        business_problem=business_problem,
        provider=provider,
        api_key=resolved_key,
        model_name=model,
    ):
        yield line


# ── Results helpers ────────────────────────────────────────────────────────────


def _get_results_markdown() -> str:
    """Read the latest pipeline result from outputs/ directory."""
    artifacts_dir = Path("outputs")
    if not artifacts_dir.exists():
        return "No pipeline results yet. Run a pipeline first."

    lines = ["## Pipeline Results\n"]

    pipeline_py = artifacts_dir / "pipeline.py"
    if pipeline_py.exists():
        lines.append(
            f"**pipeline.py** — {pipeline_py.stat().st_size // 1024} KB generated"
        )

    fastapi_py = artifacts_dir / "fastapi_endpoint.py"
    if fastapi_py.exists():
        lines.append(
            f"**fastapi_endpoint.py** — {fastapi_py.stat().st_size // 1024} KB generated"
        )

    dockerfile = artifacts_dir / "Dockerfile"
    if dockerfile.exists():
        lines.append("**Dockerfile** — Ready for containerization")

    shap_png = artifacts_dir / "shap_summary.png"
    if shap_png.exists():
        lines.append("**SHAP Summary Plot** — See Downloads tab")

    return (
        "\n\n".join(lines) if len(lines) > 1 else "Run a pipeline to see results here."
    )


def _load_artifact(filename: str) -> str | None:
    """Load an artifact file for download."""
    path = Path("outputs") / filename
    return str(path) if path.exists() else None


# ── CSV preview helper ─────────────────────────────────────────────────────────


def _preview_csv(csv_file) -> tuple[str, str]:
    """Return a markdown preview of the uploaded CSV."""
    if csv_file is None:
        return "", ""
    try:
        df = pd.read_csv(csv_file.name if hasattr(csv_file, "name") else str(csv_file))
        info = (
            f"**Shape:** {len(df):,} rows × {len(df.columns)} columns  \n"
            f"**Columns:** {', '.join(df.columns.tolist()[:10])}"
            + (" ..." if len(df.columns) > 10 else "")
        )
        preview = df.head(5).to_markdown(index=False)
        return info, preview
    except Exception as exc:
        return f"Error reading CSV: {exc}", ""


# ── Gradio app ─────────────────────────────────────────────────────────────────


def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="Autonomous ML Pipeline Builder",
        css=BASE_CSS,
        theme=gr.themes.Soft(primary_hue="violet", neutral_hue="slate"),
    ) as demo:
        gr.Markdown(
            "# Autonomous ML Pipeline Builder\n"
            "Upload a CSV, describe your business problem — AI agents build, "
            "evaluate, and deploy a complete ML pipeline end to end."
        )

        with gr.Tabs():
            # ── Tab 1: Build Pipeline ──────────────────────────────────────────
            with gr.Tab("Build Pipeline"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 1. Upload your dataset")
                        csv_upload = gr.File(
                            label="CSV File",
                            file_types=[".csv"],
                            type="filepath",
                        )
                        csv_info = gr.Markdown("")
                        csv_preview = gr.Markdown("")

                        gr.Markdown("### 2. Describe the problem")
                        problem_tb = gr.Textbox(
                            label="Business Problem",
                            placeholder=(
                                "e.g. 'Predict which credit card transactions are fraudulent. "
                                "The target column is Class (0=normal, 1=fraud). "
                                "Minimize false negatives.'"
                            ),
                            lines=4,
                        )

                        gr.Markdown("### 3. LLM Provider")
                        provider_dd, model_dd, api_key_tb = make_provider_row()

                        run_btn = gr.Button(
                            "Run Pipeline", variant="primary", size="lg"
                        )

                    with gr.Column(scale=2):
                        gr.Markdown("### Live Agent Log")
                        log_output = make_log_panel(height=600)

                # Wire CSV preview
                csv_upload.change(
                    fn=_preview_csv,
                    inputs=csv_upload,
                    outputs=[csv_info, csv_preview],
                )

                # Wire pipeline run
                run_btn.click(
                    fn=_run_pipeline_direct,
                    inputs=[csv_upload, problem_tb, provider_dd, model_dd, api_key_tb],
                    outputs=log_output,
                )

            # ── Tab 2: Results ─────────────────────────────────────────────────
            with gr.Tab("Results"):
                gr.Markdown("### Pipeline Results")
                gr.Markdown(
                    "_Results appear here after the pipeline completes. "
                    "Refresh this tab when the log shows [PIPELINE COMPLETE]._"
                )
                results_md = gr.Markdown(_get_results_markdown())
                refresh_btn = gr.Button("Refresh Results")
                refresh_btn.click(fn=_get_results_markdown, outputs=results_md)

                shap_img = gr.Image(
                    label="SHAP Feature Importance",
                    value=_load_artifact("shap_summary.png"),
                    visible=Path("outputs/shap_summary.png").exists(),
                )

            # ── Tab 3: Download Artifacts ──────────────────────────────────────
            with gr.Tab("Download Artifacts"):
                gr.Markdown(
                    "### Generated Artifacts\n"
                    "After a pipeline run completes, download the generated files below."
                )
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**pipeline.py** — Complete ML pipeline code")
                        dl_pipeline = gr.DownloadButton(
                            label="Download pipeline.py",
                            value=_load_artifact("pipeline.py"),
                            visible=Path("outputs/pipeline.py").exists(),
                        )

                    with gr.Column():
                        gr.Markdown("**requirements.txt** — Dependencies")
                        dl_reqs = gr.DownloadButton(
                            label="Download requirements.txt",
                            value=_load_artifact("requirements.txt"),
                            visible=Path("outputs/requirements.txt").exists(),
                        )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**fastapi_endpoint.py** — Inference API")
                        dl_fastapi = gr.DownloadButton(
                            label="Download fastapi_endpoint.py",
                            value=_load_artifact("fastapi_endpoint.py"),
                            visible=Path("outputs/fastapi_endpoint.py").exists(),
                        )

                    with gr.Column():
                        gr.Markdown("**Dockerfile** — Container configuration")
                        dl_dockerfile = gr.DownloadButton(
                            label="Download Dockerfile",
                            value=_load_artifact("Dockerfile"),
                            visible=Path("outputs/Dockerfile").exists(),
                        )

                refresh_artifacts_btn = gr.Button("Refresh Artifact List")

                def _refresh_artifacts():
                    return [
                        gr.DownloadButton(
                            value=_load_artifact("pipeline.py"),
                            visible=Path("outputs/pipeline.py").exists(),
                        ),
                        gr.DownloadButton(
                            value=_load_artifact("requirements.txt"),
                            visible=Path("outputs/requirements.txt").exists(),
                        ),
                        gr.DownloadButton(
                            value=_load_artifact("fastapi_endpoint.py"),
                            visible=Path("outputs/fastapi_endpoint.py").exists(),
                        ),
                        gr.DownloadButton(
                            value=_load_artifact("Dockerfile"),
                            visible=Path("outputs/Dockerfile").exists(),
                        ),
                    ]

                refresh_artifacts_btn.click(
                    fn=_refresh_artifacts,
                    outputs=[dl_pipeline, dl_reqs, dl_fastapi, dl_dockerfile],
                )

            # ── Tab 4: About ───────────────────────────────────────────────────
            with gr.Tab("About"):
                gr.Markdown("""
## How It Works

This system uses a **7-agent LangGraph pipeline** to build a complete ML pipeline
from a raw CSV and a plain English problem description:

| Agent | What It Does |
|---|---|
| **Orchestrator** | Plans the pipeline: detects task type, picks models and metric |
| **Data Analyst** | Profiles the dataset: dtypes, missing values, class imbalance |
| **Feature Engineer** | Writes and executes preprocessing code in an E2B sandbox |
| **Model Trainer** | Trains 3-5 models **in parallel** using asyncio |
| **Evaluator** | Picks the winner, runs SHAP explainability, checks for bias |
| **Code Generator** | Writes clean, documented `pipeline.py` with all steps |
| **Deployment Agent** | Generates a FastAPI endpoint + Dockerfile + OpenAPI spec |

## Self-Correction Loop

When the Feature Engineer or Code Generator produces code that fails,
the system reads the traceback and asks the LLM to fix it — up to 3 times.
This is executed in a real isolated [E2B](https://e2b.dev) cloud sandbox.

## Tech Stack

`LangGraph` · `LangChain` · `E2B` · `FastAPI` · `Gradio 5` · `MLflow`  
`LightGBM` · `XGBoost` · `scikit-learn` · `SHAP` · `Pydantic v2` · `Docker`

## Supported Providers

- **Anthropic** Claude 3.5 Sonnet / Haiku / Opus
- **OpenAI** GPT-4o / GPT-4o-mini
- **Groq** Llama 3.3 70B / Llama 3.1 8B / Mixtral

## Demo Dataset

The built-in demo uses the [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
dataset — a real imbalanced classification problem from Kaggle.
""")

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=settings.gradio_port,
        share=False,
    )
