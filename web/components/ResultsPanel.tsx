"use client";

import Image from "next/image";
import { artifactUrl } from "@/lib/api";
import { ARTIFACTS, type ComparisonRow, type ResultResponse } from "@/lib/types";

const HIDE_COLS = new Set(["model", "failed", "memory_mb", "cv_std"]);

function metricColumns(rows: ComparisonRow[]): string[] {
  const seen = new Set<string>();
  for (const r of rows) {
    for (const k of Object.keys(r)) {
      if (!HIDE_COLS.has(k) && typeof r[k] === "number") seen.add(k);
    }
  }
  // Put cv_mean and train_time last.
  const cols = [...seen].filter((c) => c !== "cv_mean" && c !== "train_time_s");
  if (seen.has("cv_mean")) cols.push("cv_mean");
  if (seen.has("train_time_s")) cols.push("train_time_s");
  return cols;
}

const COL_LABEL: Record<string, string> = {
  cv_mean: "cv (f1/r²)",
  train_time_s: "time (s)",
};

export default function ResultsPanel({
  result,
  pipelineId,
}: {
  result: ResultResponse;
  pipelineId: string;
}) {
  const cols = metricColumns(result.comparison_table);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      {/* Winner banner */}
      <div
        className="panel"
        style={{
          padding: 20,
          borderColor: "var(--done)",
          boxShadow: "0 0 0 1px rgba(53,208,186,0.25), 0 0 40px rgba(53,208,186,0.08)",
        }}
      >
        <div className="eyebrow" style={{ color: "var(--done)" }}>
          Winning model
        </div>
        <div style={{ display: "flex", alignItems: "baseline", gap: 14, flexWrap: "wrap", marginTop: 6 }}>
          <span style={{ fontSize: 28, fontWeight: 600 }}>
            {result.winner_model || "—"}
          </span>
          {result.primary_metric && result.metrics?.[result.primary_metric] != null && (
            <span className="mono" style={{ color: "var(--done)", fontSize: 16 }}>
              {result.primary_metric} = {result.metrics[result.primary_metric]}
            </span>
          )}
        </div>
        {result.justification && (
          <p style={{ color: "var(--muted)", fontSize: 14, marginTop: 10, marginBottom: 0, lineHeight: 1.6 }}>
            {result.justification}
          </p>
        )}
        {/* metric tiles */}
        {result.metrics && (
          <div className="mt-4 grid gap-2" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(96px, 1fr))" }}>
            {Object.entries(result.metrics).map(([k, v]) => (
              <div
                key={k}
                style={{
                  background: "var(--bg)",
                  border: "1px solid var(--border)",
                  borderRadius: 9,
                  padding: "10px 12px",
                }}
              >
                <div className="mono" style={{ color: "var(--faint)", fontSize: 10.5, textTransform: "uppercase", letterSpacing: "0.08em" }}>
                  {k}
                </div>
                <div className="mono" style={{ fontSize: 17, marginTop: 2 }}>{v}</div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Bias warnings */}
      {result.bias_warnings.length > 0 && (
        <div
          className="panel"
          style={{ padding: 16, borderColor: "var(--running)" }}
        >
          <div className="eyebrow" style={{ color: "var(--running)" }}>Fairness check</div>
          <ul style={{ margin: "8px 0 0", paddingLeft: 18, color: "var(--muted)", fontSize: 13.5 }}>
            {result.bias_warnings.map((w, i) => (
              <li key={i} style={{ marginBottom: 4 }}>{w}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Comparison table */}
      {result.comparison_table.length > 0 && (
        <div className="panel" style={{ padding: 20 }}>
          <div className="eyebrow">Model comparison</div>
          <div style={{ overflowX: "auto", marginTop: 12 }}>
            <table className="mono" style={{ width: "100%", borderCollapse: "collapse", fontSize: 12.5 }}>
              <thead>
                <tr style={{ color: "var(--faint)", textAlign: "left" }}>
                  <th style={{ padding: "6px 10px" }}>model</th>
                  {cols.map((c) => (
                    <th key={c} style={{ padding: "6px 10px", whiteSpace: "nowrap" }}>
                      {COL_LABEL[c] ?? c}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {result.comparison_table.map((row) => {
                  const isWinner = row.model === result.winner_model;
                  return (
                    <tr
                      key={row.model}
                      style={{
                        borderTop: "1px solid var(--border-soft)",
                        background: isWinner ? "rgba(53,208,186,0.07)" : undefined,
                        color: row.failed ? "var(--error)" : "var(--text)",
                      }}
                    >
                      <td style={{ padding: "8px 10px", whiteSpace: "nowrap" }}>
                        {isWinner && <span style={{ color: "var(--done)" }}>▍ </span>}
                        {row.model}
                        {row.failed ? " (failed)" : ""}
                      </td>
                      {cols.map((c) => (
                        <td key={c} style={{ padding: "8px 10px" }}>
                          {typeof row[c] === "number" ? (row[c] as number) : "—"}
                        </td>
                      ))}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* SHAP */}
      {result.has_shap_plot && (
        <div className="panel" style={{ padding: 20 }}>
          <div className="eyebrow">Feature importance (SHAP · held-out test set)</div>
          <div style={{ marginTop: 12, background: "#fff", borderRadius: 8, padding: 8 }}>
            <Image
              src={artifactUrl(pipelineId, "shap_summary.png")}
              alt="SHAP feature importance summary"
              width={900}
              height={540}
              unoptimized
              style={{ width: "100%", height: "auto", borderRadius: 4 }}
            />
          </div>
        </div>
      )}

      {/* Artifacts */}
      <div className="panel" style={{ padding: 20 }}>
        <div className="eyebrow">Generated artifacts</div>
        <p style={{ color: "var(--muted)", fontSize: 13, margin: "8px 0 14px" }}>
          A runnable, self-contained deployment bundle — the saved pipeline predicts on
          raw input, no manual preprocessing to keep in sync.
        </p>
        <div className="grid gap-2" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))" }}>
          {ARTIFACTS.map((a) => (
            <a
              key={a.file}
              href={artifactUrl(pipelineId, a.file)}
              className="btn-ghost mono"
              style={{ display: "flex", alignItems: "center", justifyContent: "space-between", textDecoration: "none", fontSize: 12.5 }}
              download
            >
              {a.label}
              <span style={{ color: "var(--done)" }}>↓</span>
            </a>
          ))}
        </div>
      </div>
    </div>
  );
}
