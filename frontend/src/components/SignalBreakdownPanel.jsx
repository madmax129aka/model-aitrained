import React, { useState } from "react";

function SignalRow({ signal }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="border border-forensic-border rounded-lg overflow-hidden">
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center justify-between px-4 py-3 bg-white/[0.02] hover:bg-white/[0.04] transition-colors text-left"
      >
        <div className="flex items-center gap-3">
          <span className="font-mono text-sm text-slate-200">{signal.name}</span>
          <span className="text-[10px] uppercase tracking-wider text-slate-500 border border-forensic-border rounded px-1.5 py-0.5">
            weight {(signal.weight * 100).toFixed(0)}%
          </span>
        </div>
        <div className="flex items-center gap-3">
          <span className="font-mono text-sm text-forensic-cyan">
            {signal.score.toFixed(1)}%
          </span>
          <span className={`text-slate-500 transition-transform ${open ? "rotate-180" : ""}`}>
            ▾
          </span>
        </div>
      </button>
      {open && (
        <div className="px-4 py-3 text-sm text-slate-400 bg-black/20 border-t border-forensic-border animate-fade-in">
          <p>{signal.explanation}</p>
          {signal.details && Object.keys(signal.details).length > 0 && (
            <div className="mt-2 flex flex-wrap gap-2">
              {Object.entries(signal.details).map(([key, value]) => {
                if (key === "tags") return null;
                return (
                  <span
                    key={key}
                    className="text-[11px] font-mono px-2 py-1 rounded bg-white/5 border border-forensic-border text-slate-400"
                  >
                    {key}: {String(value)}
                  </span>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function SignalBreakdownPanel({ signals }) {
  if (!signals) return null;
  return (
    <div className="glass-panel rounded-xl p-6 animate-fade-in">
      <h3 className="text-sm font-mono uppercase tracking-wider text-slate-400 mb-4">
        Why we think this
      </h3>
      <div className="flex flex-col gap-3">
        {signals.map((s) => (
          <SignalRow key={s.name} signal={s} />
        ))}
      </div>
    </div>
  );
}
