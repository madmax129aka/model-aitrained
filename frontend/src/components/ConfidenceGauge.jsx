import React, { useEffect, useState } from "react";

/**
 * Animated circular gauge showing the combined AI-likelihood score (0-100).
 * Color ramps green (authentic) -> amber (uncertain) -> red (AI-generated).
 */
export default function ConfidenceGauge({ score = 0, size = 180 }) {
  const [animatedScore, setAnimatedScore] = useState(0);

  useEffect(() => {
    let frame;
    const start = performance.now();
    const duration = 900;
    const from = 0;
    const to = score;

    function tick(now) {
      const elapsed = now - start;
      const progress = Math.min(1, elapsed / duration);
      const eased = 1 - Math.pow(1 - progress, 3);
      setAnimatedScore(from + (to - from) * eased);
      if (progress < 1) frame = requestAnimationFrame(tick);
    }
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [score]);

  const radius = size / 2 - 12;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference * (1 - animatedScore / 100);

  let color = "#34d399"; // green
  if (score >= 65) color = "#f43f5e"; // red
  else if (score > 35) color = "#f59e0b"; // amber

  return (
    <div className="relative flex items-center justify-center" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke="#1e293b"
          strokeWidth={12}
          fill="none"
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke={color}
          strokeWidth={12}
          fill="none"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          style={{
            transition: "stroke 0.4s ease",
            filter: `drop-shadow(0 0 8px ${color}88)`,
          }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-3xl font-mono font-bold" style={{ color }}>
          {animatedScore.toFixed(0)}%
        </span>
        <span className="text-[10px] uppercase tracking-wider text-slate-500 mt-1">
          AI Likelihood
        </span>
      </div>
    </div>
  );
}
