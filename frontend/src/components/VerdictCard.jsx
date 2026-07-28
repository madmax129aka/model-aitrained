import React from "react";
import ConfidenceGauge from "./ConfidenceGauge.jsx";

const VERDICT_STYLES = {
  "Likely AI-Generated": {
    badge: "bg-forensic-red/10 text-forensic-red border-forensic-red/40",
    label: "LIKELY AI-GENERATED",
  },
  "Likely Authentic": {
    badge: "bg-forensic-green/10 text-forensic-green border-forensic-green/40",
    label: "LIKELY AUTHENTIC",
  },
  Uncertain: {
    badge: "bg-forensic-amber/10 text-forensic-amber border-forensic-amber/40",
    label: "UNCERTAIN",
  },
};

export default function VerdictCard({ result, onDownloadReport, downloading }) {
  if (!result) return null;
  const style = VERDICT_STYLES[result.verdict] || VERDICT_STYLES.Uncertain;

  return (
    <div className="glass-panel rounded-xl p-6 flex flex-col sm:flex-row items-center gap-6 animate-fade-in">
      <ConfidenceGauge score={result.final_score} />
      <div className="flex-1 w-full">
        <span
          className={`inline-block px-3 py-1 rounded-full text-xs font-mono font-semibold border ${style.badge}`}
        >
          {style.label}
        </span>
        <p className="mt-3 text-sm text-slate-400">
          Score from our custom-trained CNN model.
        </p>
        <div className="mt-4 flex flex-wrap gap-2 text-xs font-mono text-slate-500">
          <span className="px-2 py-1 rounded bg-white/5 border border-forensic-border">
            Model verdict: {result.model_predicted_label}
          </span>
          <span className="px-2 py-1 rounded bg-white/5 border border-forensic-border">
            Raw output: {result.model_raw_output.toFixed(4)}
          </span>
        </div>
        <button
          onClick={onDownloadReport}
          disabled={downloading}
          className="mt-5 inline-flex items-center gap-2 px-4 py-2 rounded-md bg-forensic-cyan/10 text-forensic-cyan border border-forensic-cyan/40 hover:bg-forensic-cyan/20 transition-colors text-sm font-medium disabled:opacity-50"
        >
          {downloading ? "Generating PDF..." : "Download PDF Report"}
        </button>
      </div>
    </div>
  );
}
