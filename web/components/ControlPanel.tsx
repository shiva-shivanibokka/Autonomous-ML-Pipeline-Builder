"use client";

import { useRef, useState } from "react";
import InfoTip from "@/components/InfoTip";
import { uploadCsv } from "@/lib/api";
import type { Provider, UploadResponse } from "@/lib/types";

const MODELS: Record<Provider, string[]> = {
  anthropic: ["claude-sonnet-5", "claude-opus-4-8", "claude-haiku-4-5-20251001"],
  openai: ["gpt-4o", "gpt-4o-mini"],
  groq: ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
};

/** Render a cell without letting one long value stretch the whole table. */
function formatCell(v: unknown): string {
  if (v === null || v === undefined || v === "") return "—";
  const str = String(v);
  return str.length > 24 ? `${str.slice(0, 23)}…` : str;
}

export interface RunParams {
  upload_id: string;
  business_problem: string;
  provider: Provider;
  api_key: string;
  model_name: string;
}

export default function ControlPanel({
  running,
  onRun,
  replayOnly = false,
}: {
  running: boolean;
  onRun: (p: RunParams) => void;
  /** No backend is reachable — the controls would fail, so don't offer them. */
  replayOnly?: boolean;
}) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [upload, setUpload] = useState<UploadResponse | null>(null);
  const [uploading, setUploading] = useState(false);
  const [problem, setProblem] = useState("");
  const [provider, setProvider] = useState<Provider>("anthropic");
  const [model, setModel] = useState(MODELS.anthropic[0]);
  const [apiKey, setApiKey] = useState("");
  const [error, setError] = useState("");

  async function handleFile(file: File | undefined) {
    if (!file || replayOnly) return;
    setError("");
    setUploading(true);
    setUpload(null);
    try {
      setUpload(await uploadCsv(file));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  const canRun = !replayOnly && !!upload && problem.trim().length >= 10 && !running;

  function submit() {
    if (!upload) return;
    onRun({
      upload_id: upload.upload_id,
      business_problem: problem.trim(),
      provider,
      api_key: apiKey.trim(),
      model_name: model,
    });
  }

  return (
    <div className="panel" style={{ padding: 22 }}>
      {replayOnly && (
        <div
          style={{
            border: "1px solid var(--border)",
            borderLeft: "3px solid var(--running)",
            borderRadius: 10,
            padding: "14px 16px",
            marginBottom: 20,
            fontSize: 13.5,
            lineHeight: 1.65,
            color: "var(--muted)",
          }}
        >
          <b style={{ color: "var(--fg, var(--text))" }}>
            These controls need a backend, and this page does not have one.
          </b>
          <br />
          The pipeline trains real models and runs model-written code in a
          sandbox, so it cannot be a static site. Nothing here is disabled to
          hide it — running your own dataset genuinely works, it just needs the
          stack on your machine:
          <div
            className="mono"
            style={{
              background: "var(--bg)",
              border: "1px solid var(--border)",
              borderRadius: 8,
              padding: "9px 11px",
              margin: "11px 0 9px",
              fontSize: 12.5,
              color: "var(--done)",
              overflowX: "auto",
            }}
          >
            git clone https://github.com/shiva-shivanibokka/Autonomous-ML-Pipeline-Builder
            <br />
            cd Autonomous-ML-Pipeline-Builder &amp;&amp; cp .env.example .env
            <br />
            docker compose up --build
          </div>
          Add your own API key to <span className="mono">.env</span>, open{" "}
          <span className="mono">localhost:3000</span>, and any CSV with a target
          column will run end to end.
        </div>
      )}

      {/* 1. Dataset */}
      <label className="field-label">
        1 · Dataset
        <InfoTip text="A CSV where one column is what you want to predict and the rest are features. The header row is required. Up to 50 MB; the file is stored under a server-issued id, never a path you supply." />
      </label>
      <div
        onClick={() => fileRef.current?.click()}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          handleFile(e.dataTransfer.files?.[0]);
        }}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => (e.key === "Enter" ? fileRef.current?.click() : null)}
        className="mt-2 flex flex-col items-center justify-center rounded-[10px] text-center"
        style={{
          border: "1.5px dashed var(--border)",
          padding: "20px 16px",
          background: "var(--bg)",
          cursor: replayOnly ? "not-allowed" : "pointer",
          opacity: replayOnly ? 0.45 : 1,
        }}
      >
        <input
          ref={fileRef}
          type="file"
          accept=".csv"
          hidden
          onChange={(e) => handleFile(e.target.files?.[0])}
        />
        {uploading ? (
          <span style={{ color: "var(--running)", fontSize: 13 }}>Uploading…</span>
        ) : upload ? (
          <div>
            <div style={{ color: "var(--done)", fontSize: 15 }}>{upload.filename}</div>
            <div className="mono" style={{ color: "var(--faint)", fontSize: 13, marginTop: 4 }}>
              {upload.n_rows.toLocaleString()} rows × {upload.n_cols} cols
            </div>
          </div>
        ) : (
          <div>
            <div style={{ fontSize: 15 }}>Drop a CSV or click to browse</div>
            <div style={{ color: "var(--faint)", fontSize: 13, marginTop: 4 }}>
              Up to 50 MB
            </div>
          </div>
        )}
      </div>
      {upload && (
        <>
          <div
            className="field-label"
            style={{ display: "block", marginTop: 14 }}
          >
            First {upload.preview.length} rows
            <InfoTip text="Parsed by pandas on the server, exactly as the agents will see it. If a column looks wrong here — everything in one column, or a header row read as data — fix the CSV before spending a run on it." />
          </div>
          <div className="preview-wrap">
            <table className="preview">
              <thead>
                <tr>
                  {upload.columns.map((c) => (
                    <th key={c}>{c}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {upload.preview.map((row, i) => (
                  <tr key={i}>
                    {upload.columns.map((c) => (
                      <td key={c}>{formatCell(row[c])}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {/* 2. Problem */}
      <label className="field-label" style={{ display: "block", marginTop: 22 }}>
        2 · What should the model predict?
        <InfoTip text="Plain English. Name the target column if you can. The Orchestrator reads this to decide classification vs regression, which models to try, and which metric to optimise — so 'minimise false negatives' genuinely changes the outcome." />
      </label>
      <textarea
        value={problem}
        onChange={(e) => setProblem(e.target.value)}
        rows={4}
        disabled={replayOnly}
        placeholder="e.g. Predict which transactions are fraudulent. The target column is Class (0=normal, 1=fraud). Minimize false negatives."
        style={{ marginTop: 8, resize: "vertical" }}
      />

      {/* 3. Model provider */}
      <label className="field-label" style={{ display: "block", marginTop: 22 }}>
        3 · LLM provider
        <InfoTip text="Six of the seven agents call this model — it plans the run, profiles the data, writes the preprocessing code, and writes the pipeline and serving code. Only the Model Trainer is pure scikit-learn. Your key goes straight to the provider and is never stored." />
      </label>
      <div className="mt-2 grid grid-cols-2 gap-2">
        <select
          value={provider}
          onChange={(e) => {
            const p = e.target.value as Provider;
            setProvider(p);
            setModel(MODELS[p][0]);
          }}
        >
          <option value="anthropic">Anthropic</option>
          <option value="openai">OpenAI</option>
          <option value="groq">Groq</option>
        </select>
        <select value={model} onChange={(e) => setModel(e.target.value)}>
          {MODELS[provider].map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </select>
      </div>
      <input
        type="password"
        value={apiKey}
        onChange={(e) => setApiKey(e.target.value)}
        disabled={replayOnly}
        placeholder="API key — sent once over HTTPS, never stored"
        style={{ marginTop: 8 }}
      />

      <button
        className="btn-primary"
        style={{ width: "100%", marginTop: 20 }}
        disabled={!canRun}
        onClick={submit}
      >
        {replayOnly
          ? "Run it locally to build a pipeline"
          : running
            ? "Pipeline running…"
            : "Build the pipeline"}
      </button>

      {error && (
        <div style={{ color: "var(--error)", fontSize: 14, marginTop: 12 }}>{error}</div>
      )}
      {!upload && !error && !replayOnly && (
        <div style={{ color: "var(--faint)", fontSize: 13, marginTop: 12 }}>
          Bring your own API key — the agents call your chosen provider directly.
        </div>
      )}
    </div>
  );
}
