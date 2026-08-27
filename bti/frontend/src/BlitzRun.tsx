import { useEffect, useMemo, useState } from "react";
import type { MoveResult } from "./api";

const money = (paise: number) => paise >= 10_000_000 ? `₹${(paise / 10_000_000).toFixed(1)}L` : `₹${Math.round(paise / 100).toLocaleString("en-IN")}`;
const signed = (value: number) => `${value >= 0 ? "+" : ""}${value.toFixed(2)}%`;
type JourneyPoint = MoveResult["segment_series"][number];
type Candle = { open_paise: number; high_paise: number; low_paise: number; close_paise: number };

function candleFor(point: JourneyPoint, previous: number, key: "portfolio" | "benchmark"): Candle {
  const supplied = key === "portfolio" ? point.portfolio_ohlc : point.benchmark_ohlc;
  if (supplied) return supplied;
  const close = key === "portfolio" ? point.portfolio_value_paise : point.benchmark_value_paise;
  return { open_paise: previous || close, high_paise: Math.max(previous || close, close), low_paise: Math.min(previous || close, close), close_paise: close };
}

export function MarketRun({ result, onComplete }: { result: MoveResult; onComplete: () => void }) {
  const points = result.segment_series || [];
  const [visible, setVisible] = useState(1);
  const [paused, setPaused] = useState(false);
  const [speed, setSpeed] = useState<1 | 2>(1);
  const [selected, setSelected] = useState<number | null>(null);
  useEffect(() => { setVisible(1); setPaused(false); setSelected(null); }, [result.move]);
  useEffect(() => {
    if (paused || visible >= points.length) return;
    const journeyDuration = result.mode === "RAPID" ? 10_500 : 16_000;
    const maximumStep = result.mode === "RAPID" ? 900 : 600;
    const base = Math.max(220, Math.min(maximumStep, Math.floor(journeyDuration / Math.max(points.length, 1))));
    const timer = window.setTimeout(() => setVisible((current) => Math.min(points.length, current + 1)), base / speed);
    return () => window.clearTimeout(timer);
  }, [paused, points.length, speed, visible]);

  const shown = points.slice(0, Math.max(1, visible));
  const candles = useMemo(() => {
    let previous = result.portfolio_after_execution?.value_paise || shown[0]?.portfolio_value_paise || 0;
    return shown.map((point) => { const candle = candleFor(point, previous, "portfolio"); previous = candle.close_paise; return candle; });
  }, [result.portfolio_after_execution?.value_paise, shown]);
  const benchmark = useMemo(() => {
    let previous = shown[0]?.benchmark_ohlc?.open_paise || shown[0]?.benchmark_value_paise || 0;
    return shown.map((point) => { const candle = candleFor(point, previous, "benchmark"); previous = candle.close_paise; return candle; });
  }, [shown]);
  const values = [...candles.flatMap((c) => [c.high_paise, c.low_paise]), ...benchmark.flatMap((c) => [c.high_paise, c.low_paise])];
  const min = Math.min(...values, 0), max = Math.max(...values, 1);
  const y = (value: number) => 305 - ((value - min) / (max - min || 1)) * 270;
  const x = (index: number) => 32 + (index / Math.max(points.length - 1, 1)) * 836;
  const benchmarkPath = benchmark.map((candle, index) => `${index ? "L" : "M"}${x(index)},${y(candle.close_paise)}`).join(" ");
  const focusIndex = Math.max(0, Math.min(shown.length - 1, selected ?? shown.length - 1));
  const focus = shown[focusIndex], focusCandle = candles[focusIndex], focusBenchmark = benchmark[focusIndex];
  const isRapid = result.mode === "RAPID";

  const finished = visible >= points.length;
  return <section className={`market-run-card ${isRapid ? "rapid" : "blitz"}`}>
    <header><div><span>{result.mode} MARKET REPLAY · SIMULATION</span><b>MONTH {focus?.month || 0} / {points.at(-1)?.month || points.length}</b></div><div className="market-run-controls"><button onClick={() => setPaused((value) => !value)} disabled={finished}>{paused ? "▶ RESUME" : "Ⅱ PAUSE"}</button><button onClick={() => { setPaused(true); setVisible((value) => Math.max(1, value - 1)); setSelected(null); }}>← 1M</button><button onClick={() => { setVisible(1); setSelected(null); setPaused(false); }}>↶ REPLAY</button><button onClick={() => setSpeed((value) => value === 1 ? 2 : 1)}>{speed}×</button></div></header>
    <div className="market-run-title"><div><small>{isRapid ? "THE MARKET IS TRAVELLING TO YOUR NEXT STOP" : "ONE DECISION. THE ENTIRE MARKET ANSWERS."}</small><h2>{focus?.regime.label || "The market is making its moves."}</h2><p>{focus?.regime.narrative}</p></div><div className="market-run-score"><span>PLAYER <b>{money(focus?.portfolio_value_paise || 0)}</b></span><span>NIFTY <b>{money(focus?.benchmark_value_paise || 0)}</b></span><span>ALPHA <b className={(focus?.alpha_pct || 0) >= 0 ? "positive" : "negative"}>{signed(focus?.alpha_pct || 0)}</b></span></div></div>
    <div className="market-run-terminal">
      <svg viewBox="0 0 900 330" preserveAspectRatio="none" aria-label={`${result.mode} simulated portfolio OHLC journey`}>
        <g className="market-grid">{[35, 102, 170, 237, 305].map((line) => <line key={line} x1="24" x2="876" y1={line} y2={line} />)}</g><path className="journey-nifty" d={benchmarkPath} />
        {candles.map((candle, index) => { const rising = candle.close_paise >= candle.open_paise; const bodyTop = y(Math.max(candle.open_paise, candle.close_paise)); const bodyBottom = y(Math.min(candle.open_paise, candle.close_paise)); const width = Math.max(4, Math.min(14, 620 / Math.max(points.length, 1))); return <g key={shown[index].month} className={`journey-candle ${rising ? "up" : "down"}`} onMouseEnter={() => setSelected(index)} onMouseLeave={() => setSelected(null)} onClick={() => setSelected(index)}><line x1={x(index)} x2={x(index)} y1={y(candle.high_paise)} y2={y(candle.low_paise)} /><rect x={x(index) - width / 2} y={bodyTop} width={width} height={Math.max(2, bodyBottom - bodyTop)} /><rect className="journey-hit" x={x(index) - Math.max(10, width)} y="20" width={Math.max(20, width * 2)} height="295" /></g>; })}
        {focus && <line className="journey-crosshair" x1={x(focusIndex)} x2={x(focusIndex)} y1="20" y2="315" />}
      </svg>
      <div className="market-run-tooltip"><span>SIM MONTH {focus?.month || 0}</span><b>PORTFOLIO NAV OHLC</b><div><small>OPEN</small><strong>{money(focusCandle?.open_paise || 0)}</strong><small>HIGH</small><strong>{money(focusCandle?.high_paise || 0)}</strong><small>LOW</small><strong>{money(focusCandle?.low_paise || 0)}</strong><small>CLOSE</small><strong>{money(focusCandle?.close_paise || 0)}</strong></div><p>NIFTY O/H/L/C · {money(focusBenchmark?.open_paise || 0)} / {money(focusBenchmark?.high_paise || 0)} / {money(focusBenchmark?.low_paise || 0)} / {money(focusBenchmark?.close_paise || 0)}</p><em>Drawdown {signed(focus?.portfolio_drawdown_pct || 0)}</em></div>
    </div>
    <div className="market-run-events">{shown.slice(-3).reverse().map((point) => <article key={point.month} className={point.event?.tone || "neutral"}><span>{point.event?.desk || "MARKET DESK"} · {point.event?.time || `SIM M${point.month}`}</span><b>{point.event?.headline || point.regime.label}</b><p>{point.event?.detail || point.regime.narrative}</p></article>)}</div>
    <footer><span>{paused ? "REPLAY PAUSED" : finished ? "MARKET STOP REACHED · JOURNEY AVAILABLE FOR REVIEW" : "LIVE SIMULATION · FUTURE MONTHS SEALED"}</span><progress max={points.length} value={visible} />{finished ? <button className="primary" onClick={onComplete}>{isRapid ? "OPEN REBALANCE STOP →" : "OPEN FINAL RESULT →"}</button> : <button onClick={() => { setVisible(points.length); setPaused(true); }}>SKIP TO STOP →</button>}</footer>
  </section>;
}

export const BlitzRun = MarketRun;
