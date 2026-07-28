/**
 * Small fetch wrapper for the PixelTruth FastAPI backend.
 * All endpoints are same-origin in production (FastAPI serves this built
 * frontend directly). In dev, Vite proxies /api and /model-info to :8000.
 */

async function handleResponse(res) {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const data = await res.json();
      detail = data.detail || JSON.stringify(data);
    } catch {
      // ignore parse errors, fall back to statusText
    }
    throw new Error(detail || `Request failed with status ${res.status}`);
  }
  return res;
}

export async function analyzeImage(file) {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch("/api/analyze", { method: "POST", body: formData });
  await handleResponse(res);
  return res.json();
}

export async function fetchModelInfo() {
  const res = await fetch("/api/model-info");
  await handleResponse(res);
  return res.json();
}

export async function fetchHealth() {
  const res = await fetch("/api/health");
  await handleResponse(res);
  return res.json();
}

export async function downloadReport(analysisId) {
  const formData = new FormData();
  formData.append("analysis_id", analysisId);
  const res = await fetch("/api/report", { method: "POST", body: formData });
  await handleResponse(res);
  const blob = await res.blob();
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `pixeltruth_report_${analysisId.slice(0, 8)}.pdf`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.URL.revokeObjectURL(url);
}
