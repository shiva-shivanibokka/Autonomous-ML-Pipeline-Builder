"""
core — shared infrastructure library for Autonomous-ML-Pipeline-Builder.

This package consolidates every pattern that was previously copy-pasted across
15+ repos in this portfolio (LLM provider factory, Gradio components, MLflow
helpers, dotenv loading, structured LLM output parsing).

Other repos (Autonomous-SWE-Agent, etc.) can adopt the same structure by
copying this package and adjusting only the provider defaults.
"""
