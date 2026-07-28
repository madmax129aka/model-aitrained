import React from "react";

export default function FFTViewer({ fftImage, details }) {
  if (!fftImage) return null;
  return (
    <div className="glass-panel rounded-xl p-6 animate-fade-in">
      <h3 className="text-sm font-mono uppercase tracking-wider text-slate-400 mb-3">
        Frequency Spectrum (FFT)
      </h3>
      <p className="text-xs text-slate-500 mb-3">
        Log-magnitude 2D FFT spectrum. Sharp periodic spikes can indicate
        repeated upsampling artifacts common in AI image generators.
      </p>
      <img
        src={fftImage}
        alt="FFT magnitude spectrum"
        className="w-full max-h-72 object-contain rounded-lg border border-forensic-border bg-black"
      />
      {details && (
        <p className="mt-3 text-[11px] font-mono text-slate-500">
          anomaly_ratio: {details.anomaly_ratio}
        </p>
      )}
    </div>
  );
}
