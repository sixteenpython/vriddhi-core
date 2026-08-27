import { useEffect, useMemo, useState } from "react";
import type { MoveResult } from "./api";

const money = (paise: number) => paise >= 10_000_000 ? `₹${(paise / 10_000_000).toFixed(1)}L` : `₹${Math.round(paise / 100).toLocaleString("en-IN")}`;

export function BlitzRun({ result, onComplete }: { result: MoveResult; onComplete: () => void }) {
  const points = result.segment_series || [];
  const [visible, setVisible] = useState(1);
  useEffect(() => {
    setVisible(1);
    const interval = window.setInterval(() => setVisible((current) => {
      if (current >= points.length) {
        window.clearInterval(interval);
        window.setTimeout(onComplete, 450);
        return current;
      }
      return current + 1;
    }), Math.max(90, Math.floor(6500 / Math.max(points.length, 1))));
    return () => window.clearInterval(interval);
  }, [points.length, onComplete]);
  const shown = points.slice(0, visible);
  const scale = useMemo(() => {
    const values = shown.flatMap((point) => [point.portfolio_value_paise, point.benchmark_value_paise]);
    return { min: Math.min(...values, 0), max: Math.max(...values, 1) };
  }, [shown]);
  const path = (key: "portfolio_value_paise" | "benchmark_value_paise") => shown.map((point, index) => {
    const x = points.length > 1 ? (index / (points.length - 1)) * 100 : 0;
    const y = 48 - ((point[key] - scale.min) / (scale.max - scale.min || 1)) * 42;
    return `${x},${y}`;
  }).join(" ");
  const latest = shown.at(-1);
  return <section className="blitz-run-card">
    <header><span>BLITZ RUN · NO INTERVENTION</span><b>MONTH {latest?.month || 0} / {points.length}</b></header>
    <h2>The market is making its moves.</h2>
    <svg viewBox="0 0 100 52" preserveAspectRatio="none"><polyline className="nifty" points={path("benchmark_value_paise")} /><polyline className="player" points={path("portfolio_value_paise")} /></svg>
    <div><span>PLAYER <b>{money(latest?.portfolio_value_paise || 0)}</b></span><span>NIFTY <b>{money(latest?.benchmark_value_paise || 0)}</b></span><span>{latest?.regime.label || "OPENING"}</span></div>
    <button onClick={onComplete}>SKIP TO RESULT →</button>
  </section>;
}
