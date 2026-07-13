export type Provider = "anthropic" | "openai" | "groq";

export interface UploadResponse {
  upload_id: string;
  filename: string;
  n_rows: number;
  n_cols: number;
  columns: string[];
  preview: Record<string, unknown>[];
}

export interface RunResponse {
  pipeline_id: string;
  status: string;
}

export type RunStatus = "pending" | "running" | "completed" | "failed";

export interface StatusResponse {
  pipeline_id: string;
  status: RunStatus;
  current_step: string;
  error: string | null;
  log_count: number;
}

export interface LogsResponse {
  logs: string[];
  total: number;
}

export interface ComparisonRow {
  model: string;
  failed?: boolean;
  cv_mean?: number;
  cv_std?: number;
  train_time_s?: number;
  [metric: string]: number | string | boolean | undefined;
}

export interface ResultResponse {
  pipeline_id: string;
  status: RunStatus;
  winner_model: string | null;
  primary_metric: string | null;
  metrics: Record<string, number> | null;
  justification: string | null;
  bias_warnings: string[];
  comparison_table: ComparisonRow[];
  has_shap_plot: boolean;
  has_pipeline_code: boolean;
  has_fastapi_endpoint: boolean;
  has_dockerfile: boolean;
  logs: string[];
}

export const AGENTS: { key: string; name: string; blurb: string }[] = [
  { key: "orchestrator", name: "Orchestrator", blurb: "Plans the pipeline & picks models" },
  { key: "data_analyst", name: "Data Analyst", blurb: "Profiles the dataset" },
  { key: "feature_engineer", name: "Feature Engineer", blurb: "Writes & runs preprocessing" },
  { key: "model_trainer", name: "Model Trainer", blurb: "Trains models in parallel" },
  { key: "evaluator", name: "Evaluator", blurb: "Selects winner, runs SHAP" },
  { key: "code_generator", name: "Code Generator", blurb: "Writes pipeline.py" },
  { key: "deployment_agent", name: "Deployment Agent", blurb: "Generates API + Dockerfile" },
];

export const ARTIFACTS: { file: string; label: string }[] = [
  { file: "pipeline.py", label: "pipeline.py" },
  { file: "model.pkl", label: "model.pkl" },
  { file: "feature_schema.json", label: "feature_schema.json" },
  { file: "fastapi_endpoint.py", label: "fastapi_endpoint.py" },
  { file: "Dockerfile", label: "Dockerfile" },
  { file: "requirements.txt", label: "requirements.txt" },
  { file: "openapi_spec.json", label: "openapi_spec.json" },
  { file: "shap_summary.png", label: "shap_summary.png" },
];
