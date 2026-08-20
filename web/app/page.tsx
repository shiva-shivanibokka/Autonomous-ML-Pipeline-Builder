"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import ControlPanel, { type RunParams } from "@/components/ControlPanel";
import AgentRail, { type AgentState, type RailAgent } from "@/components/AgentRail";
import LogConsole from "@/components/LogConsole";
import ResultsPanel from "@/components/ResultsPanel";
import { API_BASE, getLogs, getResult, getStatus, runPipeline } from "@/lib/api";
import {
  DEMO_MODE,
  DEMO_PIPELINE_ID,
  REPLAY_SECONDS,
  loadDemoRun,
  playDemoRun,
  type DemoRun,
} from "@/lib/demo";
import { AGENTS, type ResultResponse, type RunStatus } from "@/lib/types";

function deriveAgents(currentStep: string, status: RunStatus): RailAgent[] {
  const activeIdx = AGENTS.findIndex((a) => a.key === currentStep);
  return AGENTS.map((a, i) => {
    let state: AgentState = "pending";
    if (status === "completed") state = "done";
    else if (status === "failed") {
      if (activeIdx === -1) state = i === 0 ? "error" : "pending";
      else state = i < activeIdx ? "done" : i === activeIdx ? "error" : "pending";
    } else if (activeIdx !== -1) {
      state = i < activeIdx ? "done" : i === activeIdx ? "running" : "pending";
    }
    return { ...a, state };
  });
}

export default function Home() {
  const [pipelineId, setPipelineId] = useState<string | null>(null);
  const [status, setStatus] = useState<RunStatus>("pending");
  const [currentStep, setCurrentStep] = useState("orchestrator");
  const [logs, setLogs] = useState<string[]>([]);
  const [result, setResult] = useState<ResultResponse | null>(null);
  const [runError, setRunError] = useState("");
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  const started = pipelineId !== null;
  const running = started && (status === "pending" || status === "running");

  const stopPolling = useCallback(() => {
    if (timer.current) clearInterval(timer.current);
    timer.current = null;
  }, []);

  useEffect(() => stopPolling, [stopPolling]);

  // One dropped poll means nothing; a run of them means the backend is gone.
  // Swallowing every failure silently left the UI spinning "running" forever
  // with no clue that nothing was listening.
  const consecutiveFailures = useRef(0);
  const POLL_FAILURES_BEFORE_GIVING_UP = 8; // ~10s at the 1300ms interval

  const poll = useCallback(
    async (id: string) => {
      try {
        const [s, l] = await Promise.all([getStatus(id), getLogs(id, 0)]);
        consecutiveFailures.current = 0;
        setRunError("");
        setStatus(s.status);
        setCurrentStep(s.current_step || "orchestrator");
        setLogs(l.logs);
        if (s.status === "completed" || s.status === "failed") {
          stopPolling();
          try {
            setResult(await getResult(id));
          } catch {
            /* result unavailable on a hard failure — logs still show why */
          }
        }
      } catch (e) {
        consecutiveFailures.current += 1;
        if (consecutiveFailures.current >= POLL_FAILURES_BEFORE_GIVING_UP) {
          stopPolling();
          setStatus("failed");
          setRunError(
            `Lost contact with the backend at ${API_BASE}. ` +
              (e instanceof Error ? e.message : "The run may still be going."),
          );
        }
      }
    },
    [stopPolling],
  );

  async function onRun(params: RunParams) {
    setRunError("");
    setResult(null);
    setLogs([]);
    setStatus("pending");
    setCurrentStep("orchestrator");
    consecutiveFailures.current = 0;
    try {
      const { pipeline_id } = await runPipeline(params);
      setPipelineId(pipeline_id);
      poll(pipeline_id);
      timer.current = setInterval(() => poll(pipeline_id), 1300);
    } catch (e) {
      setRunError(e instanceof Error ? e.message : "Could not start the pipeline");
    }
  }

  // ── Replay of a recorded run ────────────────────────────────────────────
  const [demoRun, setDemoRun] = useState<DemoRun | null>(null);
  const [demoChecked, setDemoChecked] = useState(false);
  const stopReplay = useRef<(() => void) | null>(null);

  useEffect(() => {
    loadDemoRun().then((run) => {
      setDemoRun(run);
      setDemoChecked(true);
    });
  }, []);

  // Never leave an interval writing into unmounted state.
  useEffect(() => () => stopReplay.current?.(), []);

  const startReplay = useCallback(() => {
    if (!demoRun) return;
    stopPolling();
    stopReplay.current?.();
    setRunError("");
    setResult(null);
    setLogs([]);
    setStatus("pending");
    setCurrentStep("orchestrator");
    setPipelineId(DEMO_PIPELINE_ID);
    stopReplay.current = playDemoRun(demoRun, {
      onFrame: (s, step, lines) => {
        setStatus(s);
        setCurrentStep(step);
        setLogs(lines);
      },
      onDone: (r) => setResult(r),
    });
  }, [demoRun, stopPolling]);

  // On a hosted build there is no backend to talk to, so play automatically.
  useEffect(() => {
    if (DEMO_MODE && demoRun && !started) startReplay();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [demoRun]);

  const agents = started
    ? deriveAgents(currentStep, status)
    : AGENTS.map((a) => ({ ...a, state: "pending" as AgentState }));

  return (
    <main style={{ maxWidth: 1180, margin: "0 auto", padding: "0 22px 80px" }}>
      {/* Header */}
      <header style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "26px 0" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span
            aria-hidden
            style={{ width: 11, height: 11, borderRadius: 3, background: "var(--done)", boxShadow: "0 0 12px var(--done)" }}
          />
          <span className="mono" style={{ fontSize: 13, letterSpacing: "0.04em" }}>
            autonomous-ml-pipeline
          </span>
        </div>
        <a
          href="https://github.com/shiva-shivanibokka/Autonomous-ML-Pipeline-Builder"
          target="_blank"
          rel="noopener noreferrer"
          className="mono"
          style={{ color: "var(--muted)", fontSize: 12.5, textDecoration: "none" }}
        >
          view source ↗
        </a>
      </header>

      {/* Hero */}
      <section style={{ padding: "26px 0 34px", maxWidth: 780 }}>
        <div className="eyebrow">CSV in · deployable model out</div>
        <h1 style={{ fontSize: "clamp(34px, 5vw, 54px)", lineHeight: 1.05, margin: "14px 0 0", fontWeight: 600, letterSpacing: "-0.02em" }}>
          Describe the problem.
          <br />
          Watch seven agents{" "}
          <span style={{ color: "var(--done)" }}>build the pipeline.</span>
        </h1>
        <p style={{ color: "var(--muted)", fontSize: 16.5, lineHeight: 1.6, marginTop: 18 }}>
          Upload a dataset and a plain-English goal. A LangGraph crew profiles the data,
          engineers leakage-safe features, trains and cross-validates models in parallel,
          explains the winner with SHAP, and hands you a runnable FastAPI + Docker bundle.
        </p>
      </section>

      {/* What you are looking at — shown whenever a recording is available,
          so a replay is never mistaken for a live pipeline. */}
      {demoChecked && (demoRun || DEMO_MODE) && (
        <ReplayBanner run={demoRun} onReplay={startReplay} playing={running} />
      )}

      {/* Workspace */}
      <section
        style={{ display: "grid", gap: 22, gridTemplateColumns: "minmax(300px, 380px) 1fr", alignItems: "start" }}
        className="workspace"
      >
        <ControlPanel running={running} onRun={onRun} />

        <div className="panel" style={{ padding: 22 }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 18 }}>
            <div className="eyebrow">Pipeline · live</div>
            <StatusChip status={started ? status : "idle"} />
          </div>

          <div style={{ display: "grid", gap: 22, gridTemplateColumns: "minmax(200px, 240px) 1fr" }} className="livegrid">
            <AgentRail agents={agents} />
            <LogConsole
              logs={logs}
              emptyHint="Run a pipeline to stream the agents' work here…"
            />
          </div>

          {runError && (
            <div style={{ color: "var(--error)", fontSize: 13.5, marginTop: 14 }}>
              {runError}
            </div>
          )}
        </div>
      </section>

      {/* Results */}
      {result && (result.status === "completed" || result.winner_model) && (
        <section style={{ marginTop: 30 }}>
          <div className="eyebrow" style={{ marginBottom: 14 }}>Results</div>
          <ResultsPanel result={result} pipelineId={pipelineId!} />
        </section>
      )}

      <style>{`
        @media (max-width: 900px) {
          .workspace { grid-template-columns: 1fr !important; }
          .livegrid { grid-template-columns: 1fr !important; }
        }
      `}</style>
    </main>
  );
}

function StatusChip({ status }: { status: RunStatus | "idle" }) {
  const map: Record<string, { c: string; t: string }> = {
    idle: { c: "var(--faint)", t: "idle" },
    pending: { c: "var(--running)", t: "starting" },
    running: { c: "var(--running)", t: "running" },
    completed: { c: "var(--done)", t: "complete" },
    failed: { c: "var(--error)", t: "failed" },
  };
  const { c, t } = map[status] ?? map.idle;
  return (
    <span className="mono" style={{ display: "inline-flex", alignItems: "center", gap: 7, fontSize: 11.5, color: c, letterSpacing: "0.08em", textTransform: "uppercase" }}>
      <span style={{ width: 7, height: 7, borderRadius: "50%", background: c }} />
      {t}
    </span>
  );
}

function ReplayBanner({
  run,
  onReplay,
  playing,
}: {
  run: DemoRun | null;
  onReplay: () => void;
  playing: boolean;
}) {
  const box = {
    border: "1px solid var(--border)",
    borderRadius: 12,
    padding: "16px 18px",
    marginTop: 8,
    background: "var(--panel)",
    display: "flex",
    gap: 16,
    alignItems: "center",
    flexWrap: "wrap" as const,
    justifyContent: "space-between",
  };

  if (!run) {
    return (
      <div style={box}>
        <div style={{ fontSize: 13.5, color: "var(--muted)", lineHeight: 1.6 }}>
          <b style={{ color: "var(--fg)" }}>No recorded run is committed yet.</b>{" "}
          The backend is not hosted — it needs an LLM key and a sandbox — so this
          page can only replay a run captured locally. Capture one with{" "}
          <code className="mono">python scripts/record_demo_run.py</code>.
        </div>
      </div>
    );
  }

  const mins = Math.round(run.duration_seconds / 6) / 10;

  return (
    <div style={box}>
      <div style={{ fontSize: 13.5, color: "var(--muted)", lineHeight: 1.7, maxWidth: 780 }}>
        <b style={{ color: "var(--fg)" }}>This is a replay of one real run</b> —
        recorded {run.recorded_at.slice(0, 10)} on{" "}
        <span className="mono">{run.dataset.name}</span> (
        {run.dataset.n_rows.toLocaleString()} rows × {run.dataset.n_cols} cols) using{" "}
        <span className="mono">{run.model_name}</span>. Every log line, metric, SHAP
        plot and generated file below came out of that run; nothing is simulated.
        <br />
        The pipeline is not running now: the backend is a long-lived container that
        needs your own API key, so it is not hosted. It took{" "}
        <b style={{ color: "var(--fg)" }}>{mins} min</b> in reality, compressed to{" "}
        {REPLAY_SECONDS}s here. Clone the repo to run it for real.
      </div>
      <button className="btn-primary" onClick={onReplay} disabled={playing}>
        {playing ? "Replaying…" : "Replay the run"}
      </button>
    </div>
  );
}
