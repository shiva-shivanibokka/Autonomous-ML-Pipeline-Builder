"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import ControlPanel, { type RunParams } from "@/components/ControlPanel";
import AgentRail, { type AgentState, type RailAgent } from "@/components/AgentRail";
import LogConsole from "@/components/LogConsole";
import ResultsPanel from "@/components/ResultsPanel";
import InfoTip from "@/components/InfoTip";
import { API_BASE, getLogs, getResult, getStatus, runPipeline } from "@/lib/api";
import {
  DEMO_PIPELINE_ID,
  REPLAY_SECONDS,
  loadDemoRun,
  playDemoRun,
  shouldAutoplayReplay,
  type DemoRun,
} from "@/lib/demo";
import { AGENTS, type ResultResponse, type RunStatus } from "@/lib/types";

type TabKey = "setup" | "pipeline" | "results";

const TABS: { key: TabKey; label: string }[] = [
  { key: "setup", label: "Setup" },
  { key: "pipeline", label: "Pipeline" },
  { key: "results", label: "Results" },
];

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
  const lastAuto = useRef<TabKey | null>(null);

  const [tab, setTab] = useState<TabKey>("setup");

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
    lastAuto.current = null;
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
    lastAuto.current = null;
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

  // Hosted with no backend configured: play automatically, because there is
  // nothing else this page could do.
  useEffect(() => {
    if (demoRun && !started && shouldAutoplayReplay()) startReplay();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [demoRun]);

  const agents = started
    ? deriveAgents(currentStep, status)
    : AGENTS.map((a) => ({ ...a, state: "pending" as AgentState }));
  const doneCount = agents.filter((a) => a.state === "done").length;

  // Follow the work: starting a run shows the pipeline, finishing one shows the
  // results. Only ever moves forward, so it cannot yank a tab out from under
  // someone who navigated away deliberately.
  useEffect(() => {
    if (running && lastAuto.current !== "pipeline") {
      lastAuto.current = "pipeline";
      setTab("pipeline");
    }
  }, [running]);
  useEffect(() => {
    if (result && lastAuto.current !== "results") {
      lastAuto.current = "results";
      setTab("results");
    }
  }, [result]);

  return (
    <main className="shell">
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
      <section style={{ padding: "26px 0 30px" }}>
        <div className="eyebrow">CSV in · deployable model out</div>
        <h1 style={{ fontSize: "clamp(34px, 5vw, 54px)", lineHeight: 1.05, margin: "14px 0 0", fontWeight: 600, letterSpacing: "-0.02em" }}>
          Describe the problem.
          <br />
          Watch seven agents{" "}
          <span style={{ color: "var(--done)" }}>build the pipeline.</span>
        </h1>
        <p style={{ color: "var(--muted)", fontSize: 17.5, lineHeight: 1.65, marginTop: 18, maxWidth: "none" }}>
          Upload a dataset and a plain-English goal. A LangGraph crew profiles the data,
          engineers leakage-safe features, trains and cross-validates models in parallel,
          explains the winner with SHAP, and hands you a runnable FastAPI + Docker bundle.
        </p>
      </section>

      {/* What you are looking at — shown whenever a recording is available,
          so a replay is never mistaken for a live pipeline. */}
      {demoChecked && demoRun && (
        <ReplayBanner run={demoRun} onReplay={startReplay} playing={running} />
      )}

      {/* Tabs — setup, the live run, and results as separate views rather
          than one long scroll. */}
      <div className="tabbar" role="tablist" aria-label="Pipeline views">
        {TABS.map((t) => {
          const disabled = t.key === "results" && !result;
          return (
            <button
              key={t.key}
              role="tab"
              className="tab"
              aria-selected={tab === t.key}
              aria-controls={`panel-${t.key}`}
              id={`tab-${t.key}`}
              disabled={disabled}
              onClick={() => setTab(t.key)}
            >
              {t.label}
              {t.key === "pipeline" && started && (
                <span className="tab-badge" data-tone={running ? "running" : "done"}>
                  {running ? `${doneCount}/${AGENTS.length}` : status}
                </span>
              )}
              {t.key === "results" && result && (
                <span className="tab-badge" data-tone="done">
                  ready
                </span>
              )}
            </button>
          );
        })}
      </div>

      {tab === "setup" && (
        <section role="tabpanel" id="panel-setup" aria-labelledby="tab-setup">
          <div style={{ display: "grid", gap: 22, gridTemplateColumns: "minmax(340px, 460px) 1fr", alignItems: "start" }} className="workspace">
            <ControlPanel running={running} onRun={onRun} />
            <HowItWorks />
          </div>
        </section>
      )}

      {tab === "pipeline" && (
        <section role="tabpanel" id="panel-pipeline" aria-labelledby="tab-pipeline">
          <div className="panel" style={{ padding: 22 }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 18, gap: 12, flexWrap: "wrap" }}>
              <div className="eyebrow" style={{ display: "inline-flex", alignItems: "center" }}>
                Pipeline · live
                <InfoTip text="Seven agents run in order, each handing its state to the next. The rail on the left shows which one is working; the console on the right is the run's actual log stream, polled from the backend every 1.3 seconds." />
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
                {started && <Elapsed running={running} />}
                <StatusChip status={started ? status : "idle"} />
              </div>
            </div>

            <div style={{ display: "grid", gap: 22, gridTemplateColumns: "minmax(220px, 280px) 1fr" }} className="livegrid">
              <AgentRail agents={agents} />
              <LogConsole
                logs={logs}
                emptyHint="Run a pipeline to stream the agents' work here…"
              />
            </div>

            {runError && (
              <div style={{ color: "var(--error)", fontSize: 14, marginTop: 14 }}>
                {runError}
              </div>
            )}
          </div>
        </section>
      )}

      {tab === "results" && result && (
        <section role="tabpanel" id="panel-results" aria-labelledby="tab-results">
          <ResultsPanel result={result} pipelineId={pipelineId!} />
        </section>
      )}

      <style>{`
        @media (max-width: 1100px) {
          .workspace { grid-template-columns: 1fr !important; }
        }
        @media (max-width: 900px) {
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

/** Wall-clock time since the run started, so a long step never looks stalled. */
function Elapsed({ running }: { running: boolean }) {
  const [seconds, setSeconds] = useState(0);
  const startedAt = useRef(Date.now());

  useEffect(() => {
    if (!running) return;
    startedAt.current = Date.now();
    setSeconds(0);
    const t = setInterval(
      () => setSeconds(Math.floor((Date.now() - startedAt.current) / 1000)),
      1000,
    );
    return () => clearInterval(t);
  }, [running]);

  const mm = String(Math.floor(seconds / 60)).padStart(2, "0");
  const ss = String(seconds % 60).padStart(2, "0");
  return (
    <span className="mono" style={{ color: "var(--faint)", fontSize: 12.5 }}>
      {mm}:{ss}
    </span>
  );
}

/** The setup tab's right-hand column: what each agent will actually do. */
function HowItWorks() {
  return (
    <div className="panel" style={{ padding: 24 }}>
      <div className="eyebrow" style={{ display: "inline-flex", alignItems: "center" }}>
        What happens when you press go
        <InfoTip text="A LangGraph state machine. Each agent writes into a shared state object and hands it to the next; an error at any node short-circuits the graph and returns a message instead of a traceback." />
      </div>
      <ol style={{ listStyle: "none", padding: 0, margin: "18px 0 0", display: "grid", gap: 14 }}>
        {AGENTS.map((a, i) => (
          <li key={a.key} style={{ display: "flex", gap: 14, alignItems: "baseline" }}>
            <span
              className="mono"
              style={{ color: "var(--faint)", fontSize: 12.5, minWidth: 20 }}
            >
              {String(i + 1).padStart(2, "0")}
            </span>
            <span>
              <span style={{ fontSize: 15.5 }}>{a.name}</span>
              <span style={{ color: "var(--muted)", fontSize: 14.5, marginLeft: 9 }}>
                {a.blurb}
              </span>
            </span>
          </li>
        ))}
      </ol>
      <p style={{ color: "var(--muted)", fontSize: 14.5, lineHeight: 1.65, marginTop: 20 }}>
        Six of the seven call your chosen LLM — the Model Trainer is pure
        scikit-learn. Preprocessing is fit inside each model&apos;s pipeline on the
        training fold only, so nothing leaks from the test set, and the winner is
        picked by argmax on the metric rather than by asking the model which it
        liked.
      </p>
    </div>
  );
}
