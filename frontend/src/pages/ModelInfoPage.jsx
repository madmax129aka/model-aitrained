import React, { useEffect, useState } from "react";
import { fetchModelInfo } from "../api.js";

function MetricCard({ label, value, accent }) {
  return (
    <div className="glass-panel rounded-xl p-5 text-center">
      <p className="text-3xl font-mono font-bold" style={{ color: accent || "#22d3ee" }}>
        {value}
      </p>
      <p className="text-xs uppercase tracking-wider text-slate-500 mt-1">{label}</p>
    </div>
  );
}

function ClassMetricsTable({ perClass }) {
  const classes = Object.keys(perClass || {});
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-slate-500 border-b border-forensic-border">
            <th className="py-2 pr-4">Class</th>
            <th className="py-2 pr-4">Precision</th>
            <th className="py-2 pr-4">Recall</th>
            <th className="py-2 pr-4">F1-Score</th>
          </tr>
        </thead>
        <tbody>
          {classes.map((cls) => (
            <tr key={cls} className="border-b border-forensic-border/50">
              <td className="py-2 pr-4 font-mono text-slate-200">{cls}</td>
              <td className="py-2 pr-4 font-mono text-forensic-cyan">
                {(perClass[cls].precision * 100).toFixed(0)}%
              </td>
              <td className="py-2 pr-4 font-mono text-forensic-cyan">
                {(perClass[cls].recall * 100).toFixed(0)}%
              </td>
              <td className="py-2 pr-4 font-mono text-forensic-cyan">
                {(perClass[cls].f1_score * 100).toFixed(0)}%
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function ModelInfoPage() {
  const [info, setInfo] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchModelInfo()
      .then(setInfo)
      .catch((err) => setError(err.message));
  }, []);

  if (error) {
    return (
      <div className="glass-panel rounded-xl p-6 border border-forensic-red/40 text-forensic-red text-sm">
        Failed to load model info: {error}
      </div>
    );
  }

  if (!info) {
    return (
      <div className="flex flex-col gap-4">
        <div className="skeleton h-24 rounded-xl" />
        <div className="skeleton h-64 rounded-xl" />
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-8">
      <div>
        <h1 className="text-2xl font-bold text-slate-100">Model Info &amp; Proof of Training</h1>
        <p className="text-slate-500 mt-1 text-sm max-w-2xl">{info.model_source}</p>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <MetricCard label="Overall Accuracy" value={`${(info.overall_accuracy * 100).toFixed(0)}%`} />
        <MetricCard label="Dataset Size" value={info.dataset.total_images.toLocaleString()} accent="#34d399" />
        <MetricCard label="Test Set Size" value={info.test_set.total_images.toLocaleString()} accent="#f59e0b" />
        <MetricCard label="Train/Val/Test Split" value={info.dataset.train_val_test_split} accent="#a78bfa" />
      </div>

      <div className="glass-panel rounded-xl p-6">
        <h2 className="text-sm font-mono uppercase tracking-wider text-slate-400 mb-4">
          Per-Class Test Metrics ({info.test_set.total_images.toLocaleString()} images)
        </h2>
        <ClassMetricsTable perClass={info.per_class} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="glass-panel rounded-xl p-6">
          <h2 className="text-sm font-mono uppercase tracking-wider text-slate-400 mb-3">
            Confusion Matrix
          </h2>
          <img
            src={info.confusion_matrix_url}
            alt="Confusion matrix"
            className="w-full rounded-lg border border-forensic-border bg-black"
          />
        </div>
        <div className="glass-panel rounded-xl p-6">
          <h2 className="text-sm font-mono uppercase tracking-wider text-slate-400 mb-3">
            Training Curves
          </h2>
          <img
            src={info.training_curves_url}
            alt="Training accuracy/loss curves"
            className="w-full rounded-lg border border-forensic-border bg-black"
          />
        </div>
      </div>

      <div className="glass-panel rounded-xl p-6">
        <h2 className="text-sm font-mono uppercase tracking-wider text-slate-400 mb-3">
          Architecture Summary
        </h2>
        <ol className="flex flex-col gap-2">
          {info.architecture_summary.map((line, idx) => (
            <li key={idx} className="text-sm font-mono text-slate-300 flex gap-3">
              <span className="text-slate-600">{String(idx + 1).padStart(2, "0")}</span>
              {line}
            </li>
          ))}
        </ol>
      </div>

      <div className="glass-panel rounded-xl p-6">
        <h2 className="text-sm font-mono uppercase tracking-wider text-slate-400 mb-3">
          Training Details
        </h2>
        <div className="flex flex-wrap gap-2">
          {Object.entries(info.training_details).map(([key, value]) => (
            <span
              key={key}
              className="text-xs font-mono px-2.5 py-1 rounded bg-white/5 border border-forensic-border text-slate-400"
            >
              {key}: {String(value)}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
