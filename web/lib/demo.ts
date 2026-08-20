/**
 * Replay of a recorded pipeline run.
 *
 * The backend is a long-lived container that needs an LLM key and a paid
 * sandbox, so it is not hosted anywhere. This lets the console still be shown:
 * it replays one real run captured by `scripts/record_demo_run.py` from a JSON
 * file plus its artifacts, both served as static files.
 *
 * It is a recording, not a simulation — every log line, metric, SHAP plot and
 * generated file came out of an actual run. Nothing here is synthesised, and
 * the UI labels it as a replay so it is never mistaken for a live pipeline.
 */

import type { ResultResponse, RunStatus } from "./types";

/** Sentinel pipeline id. `artifactUrl` maps it to the static /demo assets. */
export const DEMO_PIPELINE_ID = "demo";

/** A snapshot of the run's progress, as the API reported it at that moment. */
export interface DemoFrame {
  /** Seconds after the run started. */
  t: number;
  current_step: string;
  status: RunStatus;
  /** How many log lines existed at this point. */
  n_logs: number;
}

export interface DemoRun {
  recorded_at: string;
  dataset: { name: string; n_rows: number; n_cols: number };
  business_problem: string;
  provider: string;
  model_name: string;
  /** Wall-clock seconds the real run took. */
  duration_seconds: number;
  frames: DemoFrame[];
  logs: string[];
  result: ResultResponse;
  artifacts: string[];
}

/**
 * How long the replay takes on screen. A real run is minutes; nobody watches
 * that. The banner states both this and the true duration so the compression
 * is visible rather than implied.
 */
export const REPLAY_SECONDS = 30;

export const DEMO_MODE =
  process.env.NEXT_PUBLIC_DEMO_MODE === "1" ||
  process.env.NEXT_PUBLIC_DEMO_MODE === "true";

let cached: DemoRun | null | undefined;

/**
 * Load the recorded run. Returns null when none is committed — the deploy is
 * still valid, the UI just says there is no recording rather than inventing one.
 */
export async function loadDemoRun(): Promise<DemoRun | null> {
  if (cached !== undefined) return cached;
  try {
    const res = await fetch("/demo/run.json", { cache: "force-cache" });
    if (!res.ok) {
      cached = null;
      return null;
    }
    const run = (await res.json()) as DemoRun;
    cached = run && Array.isArray(run.frames) && run.frames.length > 0 ? run : null;
  } catch {
    cached = null;
  }
  return cached;
}

export interface ReplayHandlers {
  onFrame: (status: RunStatus, currentStep: string, logs: string[]) => void;
  onDone: (result: ResultResponse) => void;
}

/**
 * Play a recorded run back on a timer.
 *
 * Returns a stop function; call it on unmount or when the user restarts, or
 * the interval outlives the component and keeps writing to dead state.
 */
export function playDemoRun(run: DemoRun, handlers: ReplayHandlers): () => void {
  const total = run.frames[run.frames.length - 1].t || 1;
  const speed = total / REPLAY_SECONDS; // recorded seconds per replayed second
  const started = Date.now();
  let i = 0;
  let stopped = false;

  const tick = () => {
    if (stopped) return;
    const elapsed = ((Date.now() - started) / 1000) * speed;

    // Advance past every frame this tick has caught up with, so a slow tab
    // fast-forwards instead of falling behind.
    let advanced = false;
    while (i < run.frames.length && run.frames[i].t <= elapsed) {
      i += 1;
      advanced = true;
    }
    const frame = run.frames[Math.max(0, i - 1)];
    if (advanced || i === 0) {
      handlers.onFrame(
        frame.status,
        frame.current_step,
        run.logs.slice(0, frame.n_logs),
      );
    }

    if (i >= run.frames.length) {
      stopped = true;
      clearInterval(timer);
      handlers.onFrame(run.result.status, frame.current_step, run.logs);
      handlers.onDone(run.result);
    }
  };

  const timer = setInterval(tick, 250);
  tick(); // paint the first frame immediately rather than after a beat

  return () => {
    stopped = true;
    clearInterval(timer);
  };
}
