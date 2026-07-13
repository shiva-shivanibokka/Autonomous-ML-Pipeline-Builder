"use client";

import { useEffect, useRef } from "react";

function lineColor(line: string): string {
  if (/ERROR|FAILED|Refusing/i.test(line)) return "var(--error)";
  if (/Done\.|COMPLETE|Winner:|Saved|succeeded/i.test(line)) return "var(--done)";
  if (/RUNNING|Training|Generating|Retrieving|Running/i.test(line)) return "var(--running)";
  return "var(--muted)";
}

export default function LogConsole({
  logs,
  emptyHint,
}: {
  logs: string[];
  emptyHint: string;
}) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Keep the newest line in view as logs stream in.
    ref.current?.scrollTo({ top: ref.current.scrollHeight });
  }, [logs.length]);

  return (
    <div
      ref={ref}
      className="console mono"
      style={{
        height: 340,
        overflowY: "auto",
        background: "#080b12",
        border: "1px solid var(--border)",
        borderRadius: 10,
        padding: "12px 14px",
        fontSize: 12.5,
        lineHeight: 1.65,
      }}
    >
      {logs.length === 0 ? (
        <span style={{ color: "var(--faint)" }}>{emptyHint}</span>
      ) : (
        logs.map((line, i) => (
          <div key={i} style={{ color: lineColor(line), whiteSpace: "pre-wrap" }}>
            {line}
          </div>
        ))
      )}
    </div>
  );
}
