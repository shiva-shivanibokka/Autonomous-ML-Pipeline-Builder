"""
core.ui.components — reusable Gradio 5 component factories.

Replaces the copy-pasted provider selector, model dropdown, API key textbox,
and API helper functions that appear in 11 Gradio apps across this portfolio.

Usage:
    import gradio as gr
    from core.ui.components import make_provider_row, make_api_helpers, BASE_CSS

    with gr.Blocks(css=BASE_CSS) as demo:
        provider_dd, model_dd, api_key_tb = make_provider_row()
"""

from __future__ import annotations

import gradio as gr

from core.providers import PROVIDER_MODELS, PROVIDER_DEFAULTS

# ── Shared CSS ─────────────────────────────────────────────────────────────────
BASE_CSS = """
footer { display: none !important; }

.panel-box {
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 16px;
    background: #f8fafc;
}

.log-panel {
    font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
    font-size: 12px;
    line-height: 1.6;
    background: #0f172a;
    color: #e2e8f0;
    border-radius: 8px;
    padding: 12px;
    white-space: pre-wrap;
    overflow-y: auto;
}

.agent-step-running { color: #facc15; }
.agent-step-done    { color: #4ade80; }
.agent-step-error   { color: #f87171; }

.metric-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px 16px;
    text-align: center;
}

.winner-badge {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 6px;
    padding: 4px 10px;
    font-weight: 600;
    font-size: 12px;
}
"""


# ── Provider + Model row ───────────────────────────────────────────────────────


def make_provider_row(
    default_provider: str = "anthropic",
) -> tuple[gr.Dropdown, gr.Dropdown, gr.Textbox]:
    """
    Create a standardised LLM provider selection row.

    Renders three components side by side:
        [Provider Dropdown] [Model Dropdown] [API Key Textbox]

    The model dropdown updates automatically when the provider changes.

    Returns:
        (provider_dropdown, model_dropdown, api_key_textbox) — wire these
        into your button .click() inputs list.
    """
    providers = list(PROVIDER_MODELS.keys())

    with gr.Row():
        provider_dd = gr.Dropdown(
            choices=providers,
            value=default_provider,
            label="LLM Provider",
            scale=1,
        )
        model_dd = gr.Dropdown(
            choices=PROVIDER_MODELS[default_provider],
            value=PROVIDER_DEFAULTS[default_provider],
            label="Model",
            scale=2,
        )
        api_key_tb = gr.Textbox(
            label="API Key",
            placeholder="Paste your API key here (or set in .env)",
            type="password",
            scale=2,
        )

    def _update_models(provider: str) -> gr.Dropdown:
        models = PROVIDER_MODELS.get(provider, [])
        default = PROVIDER_DEFAULTS.get(provider, models[0] if models else "")
        return gr.Dropdown(choices=models, value=default)

    provider_dd.change(
        fn=_update_models,
        inputs=provider_dd,
        outputs=model_dd,
    )

    return provider_dd, model_dd, api_key_tb


# ── Streaming log panel ────────────────────────────────────────────────────────


def make_log_panel(label: str = "Pipeline Log", height: int = 500) -> gr.Textbox:
    """
    Create a styled monospace log panel for streaming agent output.

    The panel uses `elem_classes="log-panel"` which is styled by BASE_CSS
    to look like a terminal.
    """
    return gr.Textbox(
        label=label,
        lines=20,
        max_lines=50,
        interactive=False,
        elem_classes=["log-panel"],
        show_copy_button=True,
    )


# ── API helper factory ─────────────────────────────────────────────────────────


def make_api_helpers(api_base_url: str):
    """
    Return pre-wired _get, _post, _delete helper functions for a given API base.

    These wrap requests calls and raise gr.Error on failure, which the Gradio
    UI handles gracefully (shows a red banner instead of crashing).

    Usage:
        _get, _post, _delete = make_api_helpers("http://localhost:8000")
        result = _get("/health")
        data   = _post("/pipeline/run", json={"csv_path": "...", ...})
    """
    import requests

    def _get(path: str, **kwargs) -> dict:
        try:
            r = requests.get(f"{api_base_url}{path}", timeout=30, **kwargs)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.ConnectionError:
            raise gr.Error(f"Cannot connect to API at {api_base_url}. Is it running?")
        except requests.exceptions.HTTPError as e:
            raise gr.Error(
                f"API error {e.response.status_code}: {e.response.text[:200]}"
            )

    def _post(path: str, **kwargs) -> dict:
        try:
            r = requests.post(f"{api_base_url}{path}", timeout=120, **kwargs)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.ConnectionError:
            raise gr.Error(f"Cannot connect to API at {api_base_url}. Is it running?")
        except requests.exceptions.HTTPError as e:
            raise gr.Error(
                f"API error {e.response.status_code}: {e.response.text[:200]}"
            )

    def _delete(path: str, **kwargs) -> dict:
        try:
            r = requests.delete(f"{api_base_url}{path}", timeout=30, **kwargs)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.ConnectionError:
            raise gr.Error(f"Cannot connect to API at {api_base_url}. Is it running?")
        except requests.exceptions.HTTPError as e:
            raise gr.Error(
                f"API error {e.response.status_code}: {e.response.text[:200]}"
            )

    return _get, _post, _delete


# ── Status badge ───────────────────────────────────────────────────────────────


def status_badge(text: str, state: str = "running") -> str:
    """
    Return an HTML badge string for use in gr.HTML() components.

    state: "running" | "done" | "error" | "pending"
    """
    colours = {
        "running": "#facc15",
        "done": "#4ade80",
        "error": "#f87171",
        "pending": "#94a3b8",
    }
    colour = colours.get(state, "#94a3b8")
    return (
        f'<span style="background:{colour}; color:#0f172a; '
        f"border-radius:4px; padding:2px 8px; font-size:11px; "
        f'font-weight:600; font-family:monospace;">{text}</span>'
    )
