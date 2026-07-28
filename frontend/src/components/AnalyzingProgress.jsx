import React, { useEffect, useState } from "react";

const STAGES = [
  "Reading image bytes...",
  "Resizing to 64x64 input tensor...",
  "Running custom CNN inference...",
  "Computing input-gradient saliency...",
  "Running FFT frequency analysis...",
  "Checking EXIF metadata...",
  "Combining weighted signals...",
];

export default function AnalyzingProgress() {
  const [stageIndex, setStageIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setStageIndex((i) => (i + 1 < STAGES.length ? i + 1 : i));
    }, 450);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="glass-panel rounded-xl p-8 animate-fade-in">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-3 h-3 rounded-full bg-forensic-cyan animate-pulse-slow" />
        <span className="font-mono text-sm text-forensic-cyan uppercase tracking-wider">
          Analyzing
        </span>
      </div>
      <div className="flex flex-col gap-2 mb-6">
        {STAGES.map((stage, idx) => (
          <div
            key={stage}
            className={`text-sm font-mono flex items-center gap-2 transition-colors ${
              idx <= stageIndex ? "text-slate-200" : "text-slate-600"
            }`}
          >
            <span>{idx < stageIndex ? "✓" : idx === stageIndex ? "▸" : "·"}</span>
            {stage}
          </div>
        ))}
      </div>
      <div className="space-y-3">
        {[1, 2, 3].map((i) => (
          <div key={i} className="skeleton h-4 rounded" style={{ width: `${100 - i * 12}%` }} />
        ))}
      </div>
    </div>
  );
}
