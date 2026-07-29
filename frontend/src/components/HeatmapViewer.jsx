import React, { useState } from "react";

/**
 * Toggle between Original / Heatmap / Slider (before-after) views of the
 * genuine input-gradient saliency heatmap produced by our own CNN's
 * gradients (see backend app/model_service.py::predict_with_saliency).
 */
export default function HeatmapViewer({ originalImage, heatmapImage }) {
  const [mode, setMode] = useState("slider"); // "original" | "heatmap" | "slider"
  const [sliderPos, setSliderPos] = useState(50);

  if (!originalImage || !heatmapImage) return null;

  return (
    <div className="glass-panel rounded-xl p-6 animate-fade-in">
      <div className="flex items-center justify-between mb-4 flex-wrap gap-2">
        <h3 className="text-sm font-mono uppercase tracking-wider text-slate-400">
          Explainability Heatmap
        </h3>
        <div className="flex gap-1 bg-white/5 rounded-md p-1 border border-forensic-border">
          {["original", "heatmap", "slider"].map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={`px-3 py-1 rounded text-xs font-medium capitalize transition-colors ${
                mode === m
                  ? "bg-forensic-cyan/20 text-forensic-cyan"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              {m}
            </button>
          ))}
        </div>
      </div>

      <p className="text-xs text-slate-500 mb-3">
        Brighter regions influenced the CNN's prediction most, computed via
        real gradient backpropagation through our own model's weights.
      </p>

      {mode === "original" && (
        <img
          src={originalImage}
          alt="Original upload"
          className="w-full max-h-96 object-contain rounded-lg border border-forensic-border"
        />
      )}

      {mode === "heatmap" && (
        <img
          src={heatmapImage}
          alt="Saliency heatmap overlay"
          className="w-full max-h-96 object-contain rounded-lg border border-forensic-border"
        />
      )}

      {mode === "slider" && (
        // Fixed-height box (matches the Original/Heatmap tabs' max-h-96) with
        // both layers using object-contain so portrait or unusual-aspect-ratio
        // images are scaled to fit, never cropped/zoomed. Both <img> layers
        // share the exact same box so the divider lines up correctly.
        <div className="relative w-full h-96 rounded-lg overflow-hidden border border-forensic-border bg-black select-none">
          <img
            src={originalImage}
            alt="Original"
            className="absolute inset-0 w-full h-full object-contain block"
          />
          <div
            className="absolute inset-0 overflow-hidden"
            style={{ clipPath: `inset(0 ${100 - sliderPos}% 0 0)` }}
          >
            <img
              src={heatmapImage}
              alt="Heatmap"
              className="absolute inset-0 w-full h-full object-contain block"
            />
          </div>
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-forensic-cyan shadow-glow-sm"
            style={{ left: `${sliderPos}%` }}
          />
          <input
            type="range"
            min={0}
            max={100}
            value={sliderPos}
            onChange={(e) => setSliderPos(Number(e.target.value))}
            className="absolute inset-x-0 bottom-2 w-[90%] mx-[5%] accent-cyan-400"
          />
        </div>
      )}
    </div>
  );
}
