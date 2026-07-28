import React, { useEffect, useState } from "react";
import { getHistory, clearHistory } from "../history.js";

const VERDICT_COLORS = {
  "Likely AI-Generated": "text-forensic-red border-forensic-red/40 bg-forensic-red/10",
  "Likely Authentic": "text-forensic-green border-forensic-green/40 bg-forensic-green/10",
  Uncertain: "text-forensic-amber border-forensic-amber/40 bg-forensic-amber/10",
};

export default function HistoryPage() {
  const [history, setHistory] = useState([]);

  useEffect(() => {
    setHistory(getHistory());
  }, []);

  function handleClear() {
    clearHistory();
    setHistory([]);
  }

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-2xl font-bold text-slate-100">History</h1>
          <p className="text-slate-500 mt-1 text-sm">
            Stored locally in your browser (localStorage) &mdash; never sent to a server.
          </p>
        </div>
        {history.length > 0 && (
          <button
            onClick={handleClear}
            className="px-3 py-1.5 rounded-md text-sm border border-forensic-border text-slate-400 hover:text-forensic-red hover:border-forensic-red/40 transition-colors"
          >
            Clear History
          </button>
        )}
      </div>

      {history.length === 0 ? (
        <div className="glass-panel rounded-xl p-10 text-center text-slate-500 text-sm">
          No analyses yet. Run an analysis on the Analyzer page to see it here.
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {history.map((entry) => (
            <div key={entry.id} className="glass-panel rounded-xl p-4 flex gap-3 animate-fade-in">
              {entry.thumbnail && (
                <img
                  src={entry.thumbnail}
                  alt={entry.fileName}
                  className="w-16 h-16 object-cover rounded-lg border border-forensic-border shrink-0"
                />
              )}
              <div className="flex-1 min-w-0">
                <p className="text-xs font-mono text-slate-300 truncate">{entry.fileName}</p>
                <p className="text-[11px] text-slate-500 mt-0.5">
                  {new Date(entry.timestamp).toLocaleString()}
                </p>
                <div className="flex items-center gap-2 mt-2">
                  <span
                    className={`text-[10px] font-mono px-2 py-0.5 rounded-full border ${
                      VERDICT_COLORS[entry.verdict] || VERDICT_COLORS.Uncertain
                    }`}
                  >
                    {entry.verdict}
                  </span>
                  <span className="text-[11px] font-mono text-slate-400">
                    {entry.finalScore?.toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
