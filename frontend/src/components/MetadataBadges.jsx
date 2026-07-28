import React from "react";

export default function MetadataBadges({ exifSignal }) {
  if (!exifSignal) return null;
  const { details } = exifSignal;
  const tags = details?.tags || {};
  const hasTags = Object.keys(tags).length > 0;

  return (
    <div className="glass-panel rounded-xl p-6 animate-fade-in">
      <h3 className="text-sm font-mono uppercase tracking-wider text-slate-400 mb-3">
        Metadata / EXIF
      </h3>
      <div className="flex flex-wrap gap-2 mb-3">
        <span
          className={`px-2.5 py-1 rounded-full text-xs font-medium border ${
            details?.has_exif
              ? "bg-forensic-green/10 text-forensic-green border-forensic-green/40"
              : "bg-forensic-red/10 text-forensic-red border-forensic-red/40"
          }`}
        >
          {details?.has_exif ? "EXIF Present" : "No EXIF Data"}
        </span>
        <span
          className={`px-2.5 py-1 rounded-full text-xs font-medium border ${
            details?.has_camera_info
              ? "bg-forensic-green/10 text-forensic-green border-forensic-green/40"
              : "bg-white/5 text-slate-400 border-forensic-border"
          }`}
        >
          {details?.has_camera_info ? "Camera Info Found" : "No Camera Info"}
        </span>
      </div>
      {hasTags && (
        <div className="flex flex-col gap-1">
          {Object.entries(tags).map(([key, value]) => (
            <div key={key} className="text-xs font-mono flex justify-between border-b border-forensic-border/50 py-1">
              <span className="text-slate-500">{key}</span>
              <span className="text-slate-300">{value}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
