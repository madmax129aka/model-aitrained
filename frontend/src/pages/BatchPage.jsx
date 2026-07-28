import React, { useState } from "react";
import UploadZone from "../components/UploadZone.jsx";
import ConfidenceGauge from "../components/ConfidenceGauge.jsx";
import { analyzeImage } from "../api.js";
import { addHistoryEntry } from "../history.js";

const VERDICT_COLORS = {
  "Likely AI-Generated": "text-forensic-red border-forensic-red/40 bg-forensic-red/10",
  "Likely Authentic": "text-forensic-green border-forensic-green/40 bg-forensic-green/10",
  Uncertain: "text-forensic-amber border-forensic-amber/40 bg-forensic-amber/10",
};

export default function BatchPage() {
  const [items, setItems] = useState([]); // { id, file, status, result, error }

  async function handleFiles(files) {
    const newItems = files.map((file) => ({
      id: `${file.name}-${file.size}-${Math.random().toString(36).slice(2)}`,
      file,
      status: "analyzing",
      result: null,
      error: null,
    }));
    setItems((prev) => [...newItems, ...prev]);

    newItems.forEach(async (item) => {
      try {
        const data = await analyzeImage(item.file);
        setItems((prev) =>
          prev.map((i) => (i.id === item.id ? { ...i, status: "done", result: data } : i))
        );
        addHistoryEntry({ ...data, fileName: item.file.name });
      } catch (err) {
        setItems((prev) =>
          prev.map((i) =>
            i.id === item.id ? { ...i, status: "error", error: err.message } : i
          )
        );
      }
    });
  }

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-100">Batch Mode</h1>
        <p className="text-slate-500 mt-1 text-sm">
          Upload multiple images at once and see results side-by-side.
        </p>
      </div>

      <UploadZone onFiles={handleFiles} multiple />

      {items.length > 0 && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {items.map((item) => (
            <div key={item.id} className="glass-panel rounded-xl p-4 animate-fade-in flex flex-col gap-3">
              <p className="text-xs font-mono text-slate-400 truncate">{item.file.name}</p>

              {item.status === "analyzing" && (
                <div className="flex flex-col gap-2">
                  <div className="skeleton h-32 rounded-lg" />
                  <p className="text-xs text-forensic-cyan animate-pulse-slow font-mono">
                    Analyzing...
                  </p>
                </div>
              )}

              {item.status === "error" && (
                <p className="text-xs text-forensic-red">{item.error}</p>
              )}

              {item.status === "done" && item.result && (
                <div className="flex flex-col items-center gap-2">
                  {item.result.original_image && (
                    <img
                      src={item.result.original_image}
                      alt={item.file.name}
                      className="w-full h-32 object-cover rounded-lg border border-forensic-border"
                    />
                  )}
                  <ConfidenceGauge score={item.result.final_score} size={100} />
                  <span
                    className={`text-[11px] font-mono px-2 py-0.5 rounded-full border ${
                      VERDICT_COLORS[item.result.verdict] || VERDICT_COLORS.Uncertain
                    }`}
                  >
                    {item.result.verdict}
                  </span>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
