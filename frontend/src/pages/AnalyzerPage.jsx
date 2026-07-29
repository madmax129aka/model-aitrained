import React, { useState } from "react";
import UploadZone from "../components/UploadZone.jsx";
import AnalyzingProgress from "../components/AnalyzingProgress.jsx";
import VerdictCard from "../components/VerdictCard.jsx";
import SignalBreakdownPanel from "../components/SignalBreakdownPanel.jsx";
import HeatmapViewer from "../components/HeatmapViewer.jsx";
import FFTViewer from "../components/FFTViewer.jsx";
import MetadataBadges from "../components/MetadataBadges.jsx";
import { analyzeImage, downloadReport } from "../api.js";
import { addHistoryEntry } from "../history.js";

export default function AnalyzerPage() {
  const [status, setStatus] = useState("idle"); // idle | analyzing | done | error
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [downloading, setDownloading] = useState(false);
  const [fileName, setFileName] = useState(null);

  async function handleFiles(files) {
    const file = files[0];
    setFileName(file.name);
    setStatus("analyzing");
    setError(null);
    try {
      const data = await analyzeImage(file);
      setResult(data);
      setStatus("done");
      addHistoryEntry({ ...data, fileName: file.name });
    } catch (err) {
      setError(err.message || "Analysis failed.");
      setStatus("error");
    }
  }

  async function handleDownload() {
    if (!result) return;
    setDownloading(true);
    try {
      await downloadReport(result.analysis_id);
    } catch (err) {
      setError(err.message || "Report generation failed.");
    } finally {
      setDownloading(false);
    }
  }

  const exifSignal = result?.signals?.find((s) => s.name === "Metadata / EXIF Check");
  const fftSignal = result?.signals?.find((s) => s.name === "Frequency Domain (FFT) Analysis");

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-100">Image Analyzer</h1>
        <p className="text-slate-500 mt-1 text-sm">
          Upload an image to run it through our custom-trained CNN, frequency
          analysis, and metadata forensics. Detection works best on photos
          containing a clear human face, since that's what our model was
          trained on.
        </p>
      </div>

      <UploadZone onFiles={handleFiles} disabled={status === "analyzing"} />

      {fileName && status !== "idle" && (
        <p className="text-xs font-mono text-slate-500">
          File: <span className="text-slate-300">{fileName}</span>
        </p>
      )}

      {status === "analyzing" && <AnalyzingProgress />}

      {status === "error" && (
        <div className="glass-panel rounded-xl p-6 border border-forensic-red/40 text-forensic-red text-sm animate-fade-in">
          {error}
        </div>
      )}

      {status === "done" && result && (
        <div className="flex flex-col gap-6">
          <VerdictCard
            result={result}
            onDownloadReport={handleDownload}
            downloading={downloading}
          />
          <SignalBreakdownPanel signals={result.signals} />
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <HeatmapViewer
              originalImage={result.original_image}
              heatmapImage={result.heatmap_image}
            />
            <div className="flex flex-col gap-6">
              <FFTViewer fftImage={result.fft_spectrum_image} details={fftSignal?.details} />
              <MetadataBadges exifSignal={exifSignal} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
