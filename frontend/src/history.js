/**
 * localStorage-based session history for past analyses. No server-side
 * persistence -- purely client-side, scoped to this browser.
 */
const STORAGE_KEY = "pixeltruth_history_v1";
const MAX_ENTRIES = 50;

export function getHistory() {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

export function addHistoryEntry(entry) {
  const history = getHistory();
  const record = {
    id: entry.analysis_id,
    timestamp: Date.now(),
    fileName: entry.fileName || "image",
    verdict: entry.verdict,
    finalScore: entry.final_score,
    thumbnail: entry.original_image,
  };
  const updated = [record, ...history.filter((h) => h.id !== record.id)].slice(
    0,
    MAX_ENTRIES
  );
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
  return updated;
}

export function clearHistory() {
  window.localStorage.removeItem(STORAGE_KEY);
}
