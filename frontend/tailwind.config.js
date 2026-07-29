/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        forensic: {
          bg: "#070b12",
          panel: "#0d131f",
          panel2: "#101827",
          border: "#1e293b",
          cyan: "#22d3ee",
          cyan2: "#06b6d4",
          amber: "#f59e0b",
          red: "#f43f5e",
          green: "#34d399",
        },
      },
      fontFamily: {
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "monospace"],
        sans: ["Inter", "system-ui", "sans-serif"],
      },
      boxShadow: {
        glow: "0 0 20px rgba(34, 211, 238, 0.35)",
        "glow-sm": "0 0 10px rgba(34, 211, 238, 0.25)",
      },
      animation: {
        "fade-in": "fadeIn 0.5s ease-out forwards",
        "scan-line": "scanLine 2.2s linear infinite",
        "pulse-slow": "pulseSlow 2.4s ease-in-out infinite",
      },
      keyframes: {
        fadeIn: {
          "0%": { opacity: 0, transform: "translateY(6px)" },
          "100%": { opacity: 1, transform: "translateY(0)" },
        },
        scanLine: {
          "0%": { transform: "translateY(-100%)" },
          "100%": { transform: "translateY(100%)" },
        },
        pulseSlow: {
          "0%, 100%": { opacity: 0.5 },
          "50%": { opacity: 1 },
        },
      },
    },
  },
  plugins: [],
};
