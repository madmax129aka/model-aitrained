import React, { useCallback, useRef, useState } from "react";

export default function UploadZone({ onFiles, multiple = false, disabled = false }) {
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef(null);

  const handleFiles = useCallback(
    (fileList) => {
      const files = Array.from(fileList).filter((f) => f.type.startsWith("image/"));
      if (files.length > 0) onFiles(multiple ? files : [files[0]]);
    },
    [onFiles, multiple]
  );

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDragActive(true);
      }}
      onDragLeave={() => setDragActive(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragActive(false);
        if (!disabled) handleFiles(e.dataTransfer.files);
      }}
      onClick={() => !disabled && inputRef.current?.click()}
      className={`relative rounded-xl border-2 border-dashed p-10 text-center cursor-pointer transition-all
        ${dragActive ? "border-forensic-cyan bg-forensic-cyan/5 shadow-glow" : "border-forensic-border hover:border-forensic-cyan/50"}
        ${disabled ? "opacity-50 cursor-not-allowed" : ""}
      `}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        multiple={multiple}
        className="hidden"
        disabled={disabled}
        onChange={(e) => e.target.files && handleFiles(e.target.files)}
      />
      <div className="flex flex-col items-center gap-3">
        <div className="w-14 h-14 rounded-full bg-forensic-cyan/10 border border-forensic-cyan/30 flex items-center justify-center text-forensic-cyan text-2xl">
          ⬆
        </div>
        <p className="text-slate-200 font-medium">
          Drop {multiple ? "images" : "an image"} here, or click to browse
        </p>
        <p className="text-xs text-slate-500">
          JPG, PNG, or WebP {multiple ? "· multiple files supported" : ""}
        </p>
      </div>
    </div>
  );
}
