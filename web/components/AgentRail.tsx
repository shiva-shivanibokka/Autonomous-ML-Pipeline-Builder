export type AgentState = "pending" | "running" | "done" | "error";

export interface RailAgent {
  key: string;
  name: string;
  blurb: string;
  state: AgentState;
}

const NODE: Record<AgentState, { ring: string; fill: string; label: string }> = {
  pending: { ring: "var(--border)", fill: "transparent", label: "var(--faint)" },
  running: { ring: "var(--running)", fill: "var(--running)", label: "var(--text)" },
  done: { ring: "var(--done)", fill: "var(--done)", label: "var(--text)" },
  error: { ring: "var(--error)", fill: "var(--error)", label: "var(--text)" },
};

export default function AgentRail({ agents }: { agents: RailAgent[] }) {
  return (
    <ol className="relative m-0 list-none p-0">
      {agents.map((a, i) => {
        const c = NODE[a.state];
        const isLast = i === agents.length - 1;
        return (
          <li key={a.key} className="relative flex gap-4 pb-5 last:pb-0">
            {/* connector line */}
            {!isLast && (
              <span
                aria-hidden
                className="absolute left-[13px] top-7 bottom-0 w-px"
                style={{
                  background:
                    a.state === "done" ? "var(--done)" : "var(--border-soft)",
                }}
              />
            )}
            {/* node */}
            <span
              className={`relative z-10 flex h-[27px] w-[27px] shrink-0 items-center justify-center rounded-full ${
                a.state === "running" ? "pulse" : ""
              }`}
              style={{
                border: `2px solid ${c.ring}`,
                background: a.state === "done" ? c.fill : "var(--bg)",
              }}
            >
              {a.state === "done" ? (
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none">
                  <path
                    d="M5 13l4 4L19 7"
                    stroke="#04120f"
                    strokeWidth="3"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              ) : a.state === "error" ? (
                <span style={{ color: "#2a0b0b", fontWeight: 700, fontSize: 14 }}>!</span>
              ) : (
                <span
                  className="mono"
                  style={{ color: c.label, fontSize: 11, fontWeight: 600 }}
                >
                  {i + 1}
                </span>
              )}
            </span>
            {/* label */}
            <div className="min-w-0 pt-0.5">
              <div className="flex items-center gap-2">
                <span
                  className="font-medium"
                  style={{ color: c.label, fontSize: 14 }}
                >
                  {a.name}
                </span>
                {a.state === "running" && (
                  <span
                    className="mono"
                    style={{ color: "var(--running)", fontSize: 10, letterSpacing: "0.1em" }}
                  >
                    RUNNING
                  </span>
                )}
              </div>
              <div style={{ color: "var(--faint)", fontSize: 12.5 }}>{a.blurb}</div>
            </div>
          </li>
        );
      })}
    </ol>
  );
}
