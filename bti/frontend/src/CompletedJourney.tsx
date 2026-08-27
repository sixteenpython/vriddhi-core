import { useMemo, useState } from "react";
import type { JourneyPoint, OHLCPoint } from "./api";

const money = (paise: number) => {
  const rupees = paise / 100;
  if (Math.abs(rupees) >= 10_000_000) return `₹${(rupees / 10_000_000).toFixed(2)}Cr`;
  if (Math.abs(rupees) >= 100_000) return `₹${(rupees / 100_000).toFixed(1)}L`;
  return `₹${Math.round(rupees).toLocaleString("en-IN")}`;
};
const signed = (value = 0) => `${value >= 0 ? "+" : ""}${value.toFixed(2)}%`;

function fallbackCandle(point: JourneyPoint, previous: number): OHLCPoint {
  const close = point.portfolio_value_paise;
  const open = previous || close;
  const excursion = 0.008 + ((point.month * 17) % 9) / 1000;
  return {
    open_paise: open,
    high_paise: Math.round(Math.max(open, close) * (1 + excursion)),
    low_paise: Math.round(Math.min(open, close) * (1 - excursion * 0.8)),
    close_paise: close,
  };
}

export function CompletedJourney({ points, compact = false }: { points: JourneyPoint[]; compact?: boolean }) {
  const [selected, setSelected] = useState<number | null>(null);
  const candles = useMemo(() => {
    let previous = points[0]?.portfolio_ohlc?.open_paise || points[0]?.portfolio_value_paise || 0;
    return points.map((point) => {
      const candle = point.portfolio_ohlc || fallbackCandle(point, previous);
      previous = candle.close_paise;
      return candle;
    });
  }, [points]);
  if (!points.length) return null;
  const values = [
    ...candles.flatMap((candle) => [candle.high_paise, candle.low_paise]),
    ...points.map((point) => point.benchmark_value_paise),
  ];
  const minimum = Math.min(...values) * 0.96;
  const maximum = Math.max(...values) * 1.04;
  const x = (index: number) => 34 + (index / Math.max(points.length - 1, 1)) * 832;
  const y = (value: number) => 270 - ((value - minimum) / (maximum - minimum || 1)) * 238;
  const width = Math.max(3, Math.min(12, 610 / Math.max(points.length, 1)));
  const benchmarkPath = points.map((point, index) => `${index ? "L" : "M"}${x(index)},${y(point.benchmark_value_paise)}`).join(" ");
  const focusIndex = Math.max(0, Math.min(points.length - 1, selected ?? points.length - 1));
  const focus = points[focusIndex];
  const candle = candles[focusIndex];
  return (
    <section className={`completed-journey ${compact ? "compact" : ""}`}>
      <header><div><span>COMPLETED MARKET JOURNEY</span><b>PLAYER OHLC · NIFTY CLOSE</b></div><strong>M1 — M{points.at(-1)?.month}</strong></header>
      <svg viewBox="0 0 900 292" preserveAspectRatio="none" aria-label="Completed simulated portfolio OHLC journey">
        <g className="market-grid">{[32, 92, 151, 211, 270].map((line) => <line key={line} x1="24" x2="876" y1={line} y2={line} />)}</g>
        <path className="journey-nifty" d={benchmarkPath} />
        {candles.map((item, index) => {
          const rising = item.close_paise >= item.open_paise;
          const top = y(Math.max(item.open_paise, item.close_paise));
          const bottom = y(Math.min(item.open_paise, item.close_paise));
          return <g key={points[index].month} className={`journey-candle ${rising ? "up" : "down"}`} onMouseEnter={() => setSelected(index)} onMouseLeave={() => setSelected(null)} onClick={() => setSelected(index)}>
            <line x1={x(index)} x2={x(index)} y1={y(item.high_paise)} y2={y(item.low_paise)} />
            <rect x={x(index) - width / 2} y={top} width={width} height={Math.max(2, bottom - top)} />
            <rect className="journey-hit" x={x(index) - Math.max(8, width)} y="18" width={Math.max(16, width * 2)} height="260" />
          </g>;
        })}
        <line className="journey-crosshair" x1={x(focusIndex)} x2={x(focusIndex)} y1="20" y2="278" />
      </svg>
      <div className="completed-journey-readout">
        <span><small>MONTH</small><b>{focus.month}</b></span>
        <span><small>OPEN</small><b>{money(candle.open_paise)}</b></span>
        <span><small>HIGH</small><b>{money(candle.high_paise)}</b></span>
        <span><small>LOW</small><b>{money(candle.low_paise)}</b></span>
        <span><small>CLOSE</small><b>{money(candle.close_paise)}</b></span>
        <span><small>NIFTY</small><b>{money(focus.benchmark_value_paise)}</b></span>
        <span><small>ALPHA</small><b className={(focus.alpha_pct || 0) >= 0 ? "positive" : "negative"}>{signed(focus.alpha_pct)}</b></span>
      </div>
    </section>
  );
}
