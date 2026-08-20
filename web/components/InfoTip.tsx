"use client";

import { useEffect, useRef, useState } from "react";

/**
 * A "?" that explains the control next to it.
 *
 * Opens on hover *and* on focus/click, so it is reachable by keyboard and on
 * touch, not just with a mouse. The text is also the button's accessible name,
 * which means a screen reader gets the explanation without the tooltip ever
 * being rendered.
 */
export default function InfoTip({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  const wrap = useRef<HTMLSpanElement>(null);

  // A tap opens it; a tap anywhere else should close it again.
  useEffect(() => {
    if (!open) return;
    const onDocPointerDown = (e: PointerEvent) => {
      if (!wrap.current?.contains(e.target as Node)) setOpen(false);
    };
    const onEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("pointerdown", onDocPointerDown);
    document.addEventListener("keydown", onEscape);
    return () => {
      document.removeEventListener("pointerdown", onDocPointerDown);
      document.removeEventListener("keydown", onEscape);
    };
  }, [open]);

  return (
    <span className="infotip-wrap" ref={wrap}>
      <button
        type="button"
        className="infotip-btn"
        aria-label={text}
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
      >
        ?
      </button>
      {open && (
        <span className="infotip" role="tooltip">
          {text}
        </span>
      )}
    </span>
  );
}
