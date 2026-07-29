import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// PixelTruth frontend build config.
// In dev mode, /api requests are proxied to the FastAPI backend on :8000.
// In production, the backend serves this app's built `dist/` directly from
// the same process/port (see backend/app/main.py + run.sh).
export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      // NOTE: this is the *static image asset* path (confusion_matrix.png,
      // training_curves.png), served by FastAPI's StaticFiles mount. It is
      // deliberately different from "/model-info", which is the React
      // Router page route for the Model Info page (see src/pages/ModelInfoPage.jsx).
      "/static": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
  },
});
