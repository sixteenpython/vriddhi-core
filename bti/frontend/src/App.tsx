import { useEffect, useMemo, useState } from "react";
import {
  api,
  APIRequestError,
  ensureSession,
  type Campaign,
  type Market,
  type MoveResult,
  type MoveReview,
  type Stock,
  type Trade,
} from "./api";
import { MarketTerminal } from "./Terminal";
import { StockResearch } from "./StockResearch";
import { GameBoard } from "./GameBoard";

type View =
  | "home"
  | "setup"
  | "market"
  | "stock"
  | "portfolio"
  | "review"
  | "result"
  | "progress"
  | "puzzles"
  | "lessons"
  | "leaderboard"
  | "profile"
  | "news"
  | "final";
const rupees = (p: number, compact = false) => {
  const v = p / 100;
  if (compact && v >= 1e7) return `₹${(v / 1e7).toFixed(2)}Cr`;
  if (compact && v >= 1e5) return `₹${(v / 1e5).toFixed(1)}L`;
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(v);
};
const pct = (v: number) => `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`;
const line = (a: number[], w = 640, h = 220) => {
  const lo = Math.min(...a),
    hi = Math.max(...a),
    s = hi - lo || 1;
  return a
    .map(
      (v, i) =>
        `${i ? "L" : "M"}${(i / (a.length - 1)) * w},${h - ((v - lo) / s) * h}`,
    )
    .join(" ");
};
const impact = (t: Trade, m: Market | null) => {
  const s = m?.stocks.find((x) => x.ticker === t.ticker);
  return (s?.close_paise || 0) * t.shares * (t.side === "SELL" ? 1 : -1);
};
const SIM = "SIMULATION MODE";
const NAV: [View, string, string][] = [["home", "⌂", "Home"]];
const TERMINAL_NAV: [View, string, string][] = [
  ["market", "▥", "Market Monitor"],
  ["review", "◫", "Game Board"],
];

function Badge() {
  return (
    <span className="sim-badge">
      <i /> {SIM}
    </span>
  );
}
function Spark({
  values,
  tone = "green",
}: {
  values: number[];
  tone?: "green" | "purple";
}) {
  return (
    <svg className="spark" viewBox="0 0 640 220" preserveAspectRatio="none">
      <defs>
        <linearGradient id={`fade-${tone}`} x1="0" y1="0" x2="0" y2="1">
          <stop
            stopColor={tone === "green" ? "#55e36a" : "#a758ff"}
            stopOpacity=".3"
          />
          <stop offset="1" stopColor="#081018" stopOpacity="0" />
        </linearGradient>
      </defs>
      <path
        d={`${line(values)} L640,220 L0,220Z`}
        fill={`url(#fade-${tone})`}
      />
      <path
        d={line(values)}
        fill="none"
        stroke={tone === "green" ? "#55e36a" : "#a758ff"}
        strokeWidth="4"
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  );
}
function Metric({
  label,
  value,
  tone = "",
}: {
  label: string;
  value: string;
  tone?: string;
}) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong className={tone}>{value}</strong>
    </div>
  );
}
function Header({
  campaign,
  terminalMode,
}: {
  campaign: Campaign | null;
  terminalMode: boolean;
}) {
  return (
    <header>
      <div>
        <span className="eyebrow">
          {terminalMode
            ? "BTI PORTFOLIO INTELLIGENCE DESK"
            : "BTI CAMPAIGN CONTROL"}
        </span>
        <h2>
          {campaign
            ? `${campaign.campaign_id} · Move ${campaign.current_move} / ${campaign.horizon_months}`
            : "Beat the Index"}
        </h2>
      </div>
      <div className="header-actions">
        <Badge />
        <button className="icon-btn" aria-label="Notifications">
          ◌
        </button>
        <button className="avatar">AT</button>
      </div>
    </header>
  );
}
function Shell({
  view,
  setView,
  campaign,
  children,
}: {
  view: View;
  setView: (v: View) => void;
  campaign: Campaign | null;
  children: React.ReactNode;
}) {
  const terminalMode = [
    "market",
    "stock",
    "portfolio",
    "review",
    "news",
  ].includes(view);
  const navigation = terminalMode ? TERMINAL_NAV : NAV;
  return (
    <div className="shell">
      <aside>
        <button className="brand" onClick={() => setView("home")}>
          <b>BTI</b>
          <small>BEAT THE INDEX</small>
        </button>
        <nav>
          {navigation.map(([id, icon, label]) => (
            <button
              key={id}
              className={
                view === id ||
                (id === "market" && ["stock", "news"].includes(view)) ||
                (id === "review" &&
                  ["result", "progress", "portfolio"].includes(view))
                  ? "active"
                  : ""
              }
              onClick={() => setView(id)}
            >
              <span>{icon}</span>
              {label}
            </button>
          ))}
        </nav>
        <div className="rail-foot">
          <Badge />
          <div className="player-identity">
            <span className="avatar">AT</span>
            <span>
              <b>{terminalMode ? "Arjun · Strategy Desk" : "Arjun Trader"}</b>
              <small>
                {terminalMode ? "Decision rating 1450" : "Rating 1450"}
              </small>
            </span>
          </div>
        </div>
      </aside>
      <main>
        <Header campaign={campaign} terminalMode={terminalMode} />
        <div className="simulation-strip">
          <b>
            The stocks are real. The market you are about to play in is not.
          </b>
          <span>
            Every price path and event is a deterministic simulation—
            {"not a live quote or investment recommendation"}.
          </span>
        </div>
        {children}
      </main>
    </div>
  );
}

function Home({
  campaign,
  setView,
}: {
  campaign: Campaign | null;
  setView: (v: View) => void;
}) {
  return (
    <section className="page home">
      <div className="hero">
        <div>
          <span className="eyebrow">VRIDDHI INTELLIGENCE · YOUR DECISIONS</span>
          <h1>
            Can you <em>beat the index?</em>
          </h1>
          <p>
            Read a realistic market. Build one complete portfolio move each
            month. Learn what strong investment decisions look like—without ever
            seeing the hidden answer.
          </p>
          <div className="hero-actions">
            {campaign ? (
              <button
                className="primary"
                onClick={() =>
                  setView(campaign.status === "COMPLETED" ? "review" : "market")
                }
              >
                {campaign.status === "COMPLETED"
                  ? "VIEW FINAL MATCH RESULT"
                  : `PLAY MOVE ${campaign.current_move}`} {" "}
                <b>→</b>
              </button>
            ) : (
              <button className="primary" onClick={() => setView("setup")}>
                START A CAMPAIGN <b>→</b>
              </button>
            )}
          </div>
        </div>
        <div className="orb">
          <span>BTI</span>
          <b>1450</b>
          <small>CLUB PLAYER</small>
        </div>
      </div>
      {campaign && (
        <div className="campaign-card">
          <div className="campaign-top">
            <div>
              <span className="eyebrow">
                {campaign.status === "COMPLETED" ? "COMPLETED RATED CAMPAIGN" : "ACTIVE CAMPAIGN"}
              </span>
              <h3>
                {rupees(campaign.monthly_amount_rupees * 100)} / month ·{" "}
                {campaign.horizon_months} moves
              </h3>
            </div>
            <span className="positive">{pct(campaign.alpha_pct)} ALPHA</span>
          </div>
          <div className="progress">
            <i
              style={{
                width: `${(campaign.moves_completed / campaign.horizon_months) * 100}%`,
              }}
            />
          </div>
          <div className="metric-row">
            <Metric
              label="CURRENT MOVE"
              value={`${campaign.current_move} / ${campaign.horizon_months}`}
            />
            <Metric
              label="PORTFOLIO"
              value={rupees(campaign.portfolio_value_paise, true)}
            />
            <Metric
              label="NIFTY 50"
              value={rupees(campaign.benchmark_value_paise, true)}
            />
            <Metric
              label="BTI RATING"
              value={String(campaign.rating)}
              tone="gold"
            />
          </div>
        </div>
      )}
      <div className="dashboard-grid gameplay-only-grid">
        <div className="feature-card">
          <span className="feature-icon green">▥</span>
          <b>Market Monitor</b>
          <p>
            Read the simulated market and construct one complete monthly move.
          </p>
        </div>
        <div className="feature-card">
          <span className="feature-icon purple">◫</span>
          <b>Rated Game Board</b>
          <p>
            Commit permanently, chase Nifty and study every move in your record.
          </p>
        </div>
      </div>
    </section>
  );
}

function Setup({ start }: { start: (a: number, h: number) => void }) {
  const [a, setA] = useState(50000),
    [h, setH] = useState(36);
  return (
    <section className="page narrow">
      <div className="title-block">
        <span className="eyebrow">NEW CAMPAIGN</span>
        <h1>Choose your challenge.</h1>
        <p>
          This decision is fixed once the campaign begins. One month becomes one
          complete portfolio move.
        </p>
      </div>
      <div className="panel setup-panel">
        <label>Monthly investment</label>
        <div className="choice-grid">
          {[10000, 25000, 50000, 75000, 100000].map((x) => (
            <button
              className={a === x ? "selected" : ""}
              onClick={() => setA(x)}
              key={x}
            >
              ₹{x / 1000}K
            </button>
          ))}
        </div>
        <label>Investment horizon</label>
        <div className="choice-grid horizons">
          {[
            [24, "2 years"],
            [36, "3 years"],
            [48, "4 years"],
            [60, "5 years"],
          ].map(([x, l]) => (
            <button
              className={h === x ? "selected" : ""}
              onClick={() => setH(x as number)}
              key={x}
            >
              <b>{l}</b>
              <small>{x} moves</small>
            </button>
          ))}
        </div>
        <div className="commitment">
          <Metric
            label="TOTAL CONTRIBUTION"
            value={rupees(a * h * 100, true)}
          />
          <Metric label="COMPLETE PORTFOLIO MOVES" value={String(h)} />
          <Metric
            label="MARKET"
            value="DETERMINISTIC SIMULATION"
            tone="positive"
          />
        </div>
        <ul className="rules">
          <li>One complete allocation decision per investment month</li>
          <li>Whole-share BUY and SELL instructions; no short selling</li>
          <li>Completed moves are immutable</li>
          <li>Decision quality is distinct from the next market outcome</li>
        </ul>
        <button className="primary full" onClick={() => start(a, h)}>
          BEGIN {h}-MOVE CAMPAIGN →
        </button>
      </div>
    </section>
  );
}

function MarketBoard({
  market,
  campaign,
  select,
  setView,
}: {
  market: Market;
  campaign: Campaign;
  select: (s: Stock) => void;
  setView: (v: View) => void;
}) {
  const [q, setQ] = useState("");
  const list = market.stocks.filter((s) =>
    `${s.ticker} ${s.sector}`.toLowerCase().includes(q.toLowerCase()),
  );
  return (
    <section className="page">
      <div className="market-head">
        <div>
          <span className="eyebrow">
            MOVE {campaign.current_move} / {campaign.horizon_months}
          </span>
          <h1>The market is yours to read.</h1>
          <p>{market.basis}</p>
        </div>
        <div className="pulse">
          <span>SIMULATED NIFTY 50</span>
          <b>24,812.45</b>
          <strong>+0.73%</strong>
          <Spark
            values={[
              1, 1.03, 1.01, 1.06, 1.04, 1.08, 1.12, 1.1, 1.17, 1.15, 1.22,
            ]}
          />
        </div>
      </div>
      <div className="toolbar">
        <div className="search">
          ⌕{" "}
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Search 50 stocks or sectors"
          />
        </div>
        <div className="filter-pills">
          <button className="selected">All 50</button>
          <button>Value</button>
          <button>Momentum</button>
          <button>Low risk</button>
        </div>
        <button
          className="primary compact"
          onClick={() => setView("portfolio")}
        >
          BUILD MOVE →
        </button>
      </div>
      <div className="market-layout">
        <div className="stock-table">
          <div className="stock-row table-head">
            <span>COMPANY</span>
            <span>PRICE</span>
            <span>MOVE</span>
            <span>PE</span>
            <span>PB</span>
            <span>PEG</span>
            <span>SHARPE</span>
            <span>DRAWDOWN</span>
            <span>FORECAST</span>
          </div>
          {list.map((s) => (
            <button
              className="stock-row"
              key={s.ticker}
              onClick={() => select(s)}
            >
              <span className="company">
                <i>{s.ticker.slice(0, 2)}</i>
                <span>
                  <b>{s.ticker}</b>
                  <small>{s.sector}</small>
                </span>
              </span>
              <b>{rupees(s.close_paise)}</b>
              <strong
                className={
                  s.close_paise >= s.open_paise ? "positive" : "negative"
                }
              >
                {pct((s.close_paise / s.open_paise - 1) * 100)}
              </strong>
              <span>{s.pe.toFixed(1)}</span>
              <span>{s.pb.toFixed(1)}</span>
              <span>{s.peg.toFixed(2)}</span>
              <span>{s.sharpe.toFixed(2)}</span>
              <span className="negative">{s.drawdown_pct.toFixed(1)}%</span>
              <strong className="positive">
                +{s.forecast_pct.toFixed(1)}%
              </strong>
            </button>
          ))}
        </div>
        <aside className="market-aside">
          <div className="panel">
            <span className="eyebrow">MARKET REGIME</span>
            <h3>Selective growth</h3>
            <p>
              Valuations are dispersed. Reward quality growth, but keep downside
              exposure deliberate.
            </p>
            <div className="regime">
              <span>Risk appetite</span>
              <i>
                <b style={{ width: "64%" }} />
              </i>
              <strong>64</strong>
            </div>
          </div>
          <div className="panel">
            <span className="eyebrow">SCENARIO EVENT</span>
            <h3>Rates remain unchanged</h3>
            <p>
              The simulated central bank signals a neutral stance.
              Rate-sensitive sectors may diverge.
            </p>
            <button className="text-btn" onClick={() => setView("news")}>
              READ EVENT CARDS →
            </button>
          </div>
        </aside>
      </div>
    </section>
  );
}

function StockDetail({
  stock,
  market,
  add,
  setView,
}: {
  stock: Stock;
  market: Market;
  add: (s: Stock) => void;
  setView: (v: View) => void;
}) {
  const change = (stock.close_paise / stock.open_paise - 1) * 100;
  return (
    <section className="page">
      <button className="back" onClick={() => setView("market")}>
        ← MARKET BOARD
      </button>
      <div className="stock-hero">
        <div>
          <span className="company-symbol">{stock.ticker.slice(0, 2)}</span>
          <div>
            <span className="eyebrow">{stock.sector} · SIMULATED QUOTE</span>
            <h1>{stock.ticker}</h1>
            <p>
              {rupees(stock.close_paise)}{" "}
              <strong className={change >= 0 ? "positive" : "negative"}>
                {pct(change)}
              </strong>
            </p>
          </div>
        </div>
        <button
          className="primary"
          onClick={() => {
            add(stock);
            setView("portfolio");
          }}
        >
          ADD TO MOVE +
        </button>
      </div>
      <div className="detail-grid">
        <div className="panel chart-panel">
          <div className="chart-tabs">
            <button>1M</button>
            <button>3M</button>
            <button className="selected">1Y</button>
            <button>3Y</button>
            <button>5Y</button>
          </div>
          <Spark values={stock.history_paise} />
          <div className="chart-axis">
            <span>Oldest simulated month</span>
            <span>Current move</span>
          </div>
        </div>
        <div className="panel signal-panel">
          <span className="eyebrow">PUBLIC SIGNALS</span>
          <div className="signal-score">
            <b>{Math.round(58 + stock.sharpe * 12 - stock.peg * 2)}</b>
            <span>
              Decision
              <br />
              evidence
            </span>
          </div>
          <p>These are observable inputs—not the hidden Vriddhi decision.</p>
          <div className="confidence">
            <i style={{ width: "78%" }} />
          </div>
        </div>
      </div>
      <div className="fundamentals">
        <Metric label="PE (TTM)" value={stock.pe.toFixed(1)} />
        <Metric label="PB" value={stock.pb.toFixed(1)} />
        <Metric label="PEG" value={stock.peg.toFixed(2)} />
        <Metric
          label="SHARPE"
          value={stock.sharpe.toFixed(2)}
          tone="positive"
        />
        <Metric
          label="VOLATILITY"
          value={`${stock.volatility_pct.toFixed(1)}%`}
        />
        <Metric
          label="MAX DRAWDOWN"
          value={`${stock.drawdown_pct.toFixed(1)}%`}
          tone="negative"
        />
        <Metric
          label="VaR (95%)"
          value={`${stock.var_95_pct.toFixed(1)}%`}
          tone="negative"
        />
        <Metric
          label="12M FORECAST"
          value={`+${stock.forecast_pct.toFixed(1)}%`}
          tone="positive"
        />
      </div>
      <div className="evidence-grid">
        <div className="panel">
          <span className="eyebrow">HOW TO READ IT</span>
          <h3>Growth has a price.</h3>
          <p>
            A PEG near 1 suggests expected growth and price are in balance. A
            higher number asks you to justify the premium with quality or
            portfolio fit.
          </p>
        </div>
        <div className="panel">
          <span className="eyebrow">PORTFOLIO QUESTION</span>
          <h3>What job does this stock perform?</h3>
          <p>
            Consider expected return, downside behaviour and what it adds—or
            duplicates—alongside your other holdings.
          </p>
        </div>
      </div>
    </section>
  );
}

function Portfolio({
  campaign,
  market,
  trades,
  setTrades,
  setView,
}: {
  campaign: Campaign;
  market: Market;
  trades: Trade[];
  setTrades: (t: Trade[]) => void;
  setView: (v: View) => void;
}) {
  const cash =
    campaign.cash_paise + trades.reduce((n, t) => n + impact(t, market), 0);
  const dep = Math.max(0, (1 - cash / campaign.cash_paise) * 100);
  const update = (i: number, k: keyof Trade, v: string) =>
    setTrades(
      trades.map((t, j) =>
        j === i
          ? { ...t, [k]: k === "shares" ? Math.max(0, Number(v)) : v }
          : t,
      ),
    );
  return (
    <section className="page">
      <div className="title-row">
        <div>
          <span className="eyebrow">
            MOVE {campaign.current_move} / {campaign.horizon_months}
          </span>
          <h1>Construct the complete move.</h1>
          <p>Sell first, then deploy this month’s capital into whole shares.</p>
        </div>
        <div className="cash-card">
          <span>CASH AVAILABLE</span>
          <b>{rupees(campaign.cash_paise)}</b>
        </div>
      </div>
      <div className="builder">
        <div className="panel orders">
          <div className="orders-head">
            <h3>Execution instructions</h3>
            <button
              className="text-btn"
              onClick={() => {
                const s = market.stocks[0];
                setTrades([
                  ...trades,
                  { side: "BUY", ticker: s.ticker, shares: 1 },
                ]);
              }}
            >
              + ADD ORDER
            </button>
          </div>
          {trades.map((t, i) => {
            const s = market.stocks.find((x) => x.ticker === t.ticker)!;
            return (
              <div className="order" key={`${t.ticker}-${i}`}>
                <select
                  value={t.side}
                  onChange={(e) => update(i, "side", e.target.value)}
                >
                  <option>BUY</option>
                  <option>SELL</option>
                </select>
                <select
                  value={t.ticker}
                  onChange={(e) => update(i, "ticker", e.target.value)}
                >
                  {market.stocks.map((x) => (
                    <option key={x.ticker}>{x.ticker}</option>
                  ))}
                </select>
                <div className="stepper">
                  <button
                    onClick={() => update(i, "shares", String(t.shares - 1))}
                  >
                    −
                  </button>
                  <input
                    value={t.shares}
                    onChange={(e) => update(i, "shares", e.target.value)}
                  />
                  <button
                    onClick={() => update(i, "shares", String(t.shares + 1))}
                  >
                    +
                  </button>
                </div>
                <span>{rupees(s.close_paise)}</span>
                <strong className={t.side === "BUY" ? "negative" : "positive"}>
                  {rupees(impact(t, market))}
                </strong>
                <button
                  className="remove"
                  onClick={() => setTrades(trades.filter((_, j) => j !== i))}
                >
                  ×
                </button>
              </div>
            );
          })}
          {!trades.length && (
            <div className="empty">
              No instructions yet. Explore the market, then build your thesis
              one order at a time.
            </div>
          )}
        </div>
        <aside className="execution-summary">
          <div className="panel">
            <span className="eyebrow">MOVE RECONCILIATION</span>
            <Metric label="Opening cash" value={rupees(campaign.cash_paise)} />
            <Metric
              label="Orders"
              value={rupees(cash - campaign.cash_paise)}
              tone={cash - campaign.cash_paise < 0 ? "negative" : "positive"}
            />
            <Metric label="Cash after move" value={rupees(cash)} />
            <div className="deployment">
              <span>
                <b>Capital deployed</b>
                <strong>{dep.toFixed(1)}%</strong>
              </span>
              <i>
                <b style={{ width: `${Math.min(100, dep)}%` }} />
              </i>
              <small>
                {dep >= 90
                  ? "Deployment gate passed"
                  : "Deploy at least 90% of available monthly capital"}
              </small>
            </div>
            <button
              disabled={!trades.length || cash < 0 || dep < 90}
              className="primary full"
              onClick={() => setView("review")}
            >
              REVIEW EXECUTION →
            </button>
          </div>
          <div className="panel coach-note">
            <span>DECISION DISCIPLINE</span>
            <p>
              A portfolio is one decision. BTI scores the combined risk,
              valuation, diversification and opportunity cost—not isolated stock
              picks.
            </p>
          </div>
        </aside>
      </div>
    </section>
  );
}

function Review({
  campaign,
  market,
  trades,
  execute,
  busy,
  setView,
}: {
  campaign: Campaign;
  market: Market;
  trades: Trade[];
  execute: () => void;
  busy: boolean;
  setView: (v: View) => void;
}) {
  let cash = campaign.cash_paise;
  return (
    <section className="page narrow-wide">
      <button className="back" onClick={() => setView("portfolio")}>
        ← EDIT MOVE
      </button>
      <div className="title-block">
        <span className="eyebrow">
          FINAL REVIEW · MOVE {campaign.current_move}
        </span>
        <h1>This decision cannot be undone.</h1>
        <p>Read the execution sheet as if you were handing it to a broker.</p>
      </div>
      <div className="panel sheet">
        <div className="sheet-row head">
          <span>STEP</span>
          <span>INSTRUCTION</span>
          <span>SIMULATED PRICE</span>
          <span>CASH MOVEMENT</span>
          <span>CASH AFTER</span>
        </div>
        {trades.map((t, i) => {
          const x = impact(t, market);
          cash += x;
          return (
            <div className="sheet-row" key={i}>
              <b>{i + 1}</b>
              <strong>
                {t.side} {t.shares} {t.shares === 1 ? "SHARE" : "SHARES"} OF{" "}
                {t.ticker}
              </strong>
              <span>
                {rupees(
                  market.stocks.find((s) => s.ticker === t.ticker)!.close_paise,
                )}
              </span>
              <span className={x > 0 ? "positive" : "negative"}>
                {rupees(x)}
              </span>
              <b>{rupees(cash)}</b>
            </div>
          );
        })}
      </div>
      <div className="review-footer">
        <div>
          <Badge />
          <p>
            Submitting locks the move, advances the simulated market and reveals
            an independent decision score.
          </p>
        </div>
        <button className="primary" disabled={busy} onClick={execute}>
          {busy ? "EVALUATING DECISION…" : "EXECUTE MOVE →"}
        </button>
      </div>
    </section>
  );
}

function Result({
  result,
  campaign,
  setView,
}: {
  result: MoveResult;
  campaign: Campaign;
  setView: (v: View) => void;
}) {
  return (
    <section className="page result-page">
      <div className="score-reveal">
        <div className="score-ring">
          <span>★</span>
          <b>{result.score}</b>
          <small>/ 100</small>
        </div>
        <div>
          <span className="eyebrow">MOVE {result.move} ANALYSIS</span>
          <h1>{result.classification}</h1>
          <p>{result.decision_quality.headline}</p>
        </div>
      </div>
      <div className="result-metrics">
        <Metric
          label="PORTFOLIO OUTCOME"
          value={pct(result.market_outcome.portfolio_return_pct)}
          tone={
            result.market_outcome.portfolio_return_pct >= 0
              ? "positive"
              : "negative"
          }
        />
        <Metric
          label="NIFTY 50 OUTCOME"
          value={pct(result.market_outcome.benchmark_return_pct)}
          tone={
            result.market_outcome.benchmark_return_pct >= 0
              ? "positive"
              : "negative"
          }
        />
        <Metric
          label="ALPHA THIS MOVE"
          value={pct(result.market_outcome.alpha_pct)}
          tone={result.market_outcome.alpha_pct >= 0 ? "positive" : "negative"}
        />
        <Metric
          label="CAMPAIGN RATING"
          value={String(campaign.rating)}
          tone="gold"
        />
      </div>
      <div className="analysis-grid">
        <div className="panel analysis good">
          <span className="eyebrow">WHAT YOU DID WELL</span>
          {result.decision_quality.did_well.map((x) => (
            <p key={x}>✓ {x}</p>
          ))}
        </div>
        <div className="panel analysis improve">
          <span className="eyebrow">YOUR NEXT EDGE</span>
          {result.decision_quality.improve.map((x) => (
            <p key={x}>↗ {x}</p>
          ))}
        </div>
      </div>
      <div className="process-note">
        <b>PROCESS ≠ OUTCOME</b>
        <p>
          {result.market_outcome.headline} BTI grades what was knowable when you
          decided; one lucky or unlucky month cannot define decision quality.
        </p>
      </div>
      <button className="primary result-cta" onClick={() => setView("market")}>
        CONTINUE TO MOVE {campaign.current_move} →
      </button>
    </section>
  );
}

function Progress({ campaign }: { campaign: Campaign }) {
  const a = [
      1, 1.03, 1.01, 1.08, 1.12, 1.2, 1.18, 1.31, 1.4, 1.38, 1.51, 1.66, 1.74,
      1.91, 2.03, 2.18, 2.34,
    ],
    b = [
      1, 1.01, 0.99, 1.04, 1.07, 1.1, 1.12, 1.17, 1.21, 1.2, 1.25, 1.29, 1.33,
      1.37, 1.39, 1.43, 1.46,
    ];
  return (
    <section className="page">
      <div className="title-row">
        <div>
          <span className="eyebrow">CAMPAIGN PROGRESS</span>
          <h1>Your decisions are compounding.</h1>
        </div>
        <strong className="positive big">
          {pct(campaign.alpha_pct)} ALPHA
        </strong>
      </div>
      <div className="panel big-chart">
        <div className="legend">
          <span className="green">● YOUR PORTFOLIO</span>
          <span className="purple">● NIFTY 50</span>
        </div>
        <svg viewBox="0 0 640 220" preserveAspectRatio="none">
          <path d={line(a)} className="plot-green" />
          <path d={line(b)} className="plot-purple" />
        </svg>
      </div>
      <div className="progress-grid">
        <div className="panel">
          <h3>Move quality</h3>
          {[
            ["Excellent", 8, 60],
            ["Good", 5, 38],
            ["Inaccuracy", 2, 16],
            ["Miss", 1, 8],
            ["Blunder", 1, 8],
          ].map(([x, n, w]) => (
            <div className="quality" key={x}>
              <span>{x}</span>
              <i>
                <b style={{ width: `${w}%` }} />
              </i>
              <strong>{n}</strong>
            </div>
          ))}
        </div>
        <div className="panel">
          <h3>Your emerging style</h3>
          <div className="skill">
            <span>Valuation discipline</span>
            <b>84</b>
          </div>
          <div className="skill">
            <span>Diversification</span>
            <b>78</b>
          </div>
          <div className="skill">
            <span>Downside awareness</span>
            <b>66</b>
          </div>
          <p className="hint">
            Your strongest edge is buying growth without overpaying. Your next
            lesson should focus on protecting against correlated sector risk.
          </p>
        </div>
      </div>
    </section>
  );
}

const lessons = [
  ["Understanding PE, PB & PEG", "Intermediate", "75%"],
  ["Sharpe ratio explained", "Beginner", "60%"],
  ["Drawdown & risk management", "Intermediate", "40%"],
  ["Portfolio diversification", "Beginner", "20%"],
  ["Valuation vs growth", "Advanced", "0%"],
];
function Explore({
  view,
  setView,
  campaign,
}: {
  view: View;
  setView: (v: View) => void;
  campaign: Campaign | null;
}) {
  const [puzzleAnswer, setPuzzleAnswer] = useState("");
  const [puzzleSubmitted, setPuzzleSubmitted] = useState(false);
  const [activeLesson, setActiveLesson] = useState<number | null>(null);
  if (view === "puzzles")
    return (
      <section className="page">
        <div className="title-row">
          <div>
            <span className="eyebrow">DAILY DECISION PUZZLE</span>
            <h1>One market. One move.</h1>
          </div>
          <span className="rating">
            PUZZLE RATING <b>1260</b>
          </span>
        </div>
        <div className="puzzle-layout">
          <div className="panel puzzle">
            <span className="difficulty">MEDIUM · #1287</span>
            <h2>Momentum or value trap?</h2>
            <p>
              Market volatility has increased. Two holdings have risen sharply,
              but their fundamentals diverged. You have ₹1,00,000 and one move.
            </p>
            <div className="scenario-stats">
              <Metric label="MARKET" value="−12.4%" tone="negative" />
              <Metric label="VALUATION SPREAD" value="HIGH" />
              <Metric label="YOUR CASH" value="₹1,00,000" />
            </div>
            <div className="puzzle-choice">
              {["BUY THE DIP", "REBALANCE", "HOLD CASH"].map((answer) => (
                <button
                  className={puzzleAnswer === answer ? "selected" : ""}
                  onClick={() => {
                    setPuzzleAnswer(answer);
                    setPuzzleSubmitted(false);
                  }}
                  key={answer}
                >
                  {answer}
                </button>
              ))}
            </div>
            <button
              className="primary full"
              disabled={!puzzleAnswer}
              onClick={() => setPuzzleSubmitted(true)}
            >
              SOLVE PUZZLE →
            </button>
            {puzzleSubmitted && (
              <div
                className={`puzzle-verdict ${puzzleAnswer === "REBALANCE" ? "correct" : "review"}`}
              >
                <b>
                  {puzzleAnswer === "REBALANCE"
                    ? "EXCELLENT · 92"
                    : "INACCURACY · 54"}
                </b>
                <p>
                  {puzzleAnswer === "REBALANCE"
                    ? "Rebalancing responds to deteriorating fundamentals while preserving exposure to the genuinely undervalued security. It improves the portfolio rather than treating every decline as a bargain."
                    : "The choice reacts to the market headline but does not separate valuation opportunity from fundamental deterioration. Re-open the risk and valuation evidence before deciding."}
                </p>
              </div>
            )}
          </div>
          <div className="panel">
            <h3>Why puzzles matter</h3>
            <p>
              Short, frozen scenarios isolate one investment skill. The same
              deterministic engine evaluates puzzles and campaign moves.
            </p>
            <div className="streak">
              <b>7</b>
              <span>
                DAY
                <br />
                STREAK
              </span>
            </div>
          </div>
        </div>
      </section>
    );
  if (view === "lessons")
    return (
      <section className="page">
        <div className="title-block">
          <span className="eyebrow">SKILL LAB</span>
          <h1>Learn from the moves you make.</h1>
          <p>Short lessons, recommended from your actual decision history.</p>
          {campaign?.last_result && (
            <div className="adaptive-recommendation">
              <b>RECOMMENDED FROM MOVE {campaign.last_result.move}</b>
              <span>
                {campaign.last_result.decision_quality.improve[0] ||
                  "Strengthen portfolio construction discipline."}
              </span>
            </div>
          )}
        </div>
        <div className="lesson-list">
          {lessons.map((x, i) => (
            <button
              className={`lesson ${activeLesson === i ? "active" : ""}`}
              key={x[0]}
              onClick={() => setActiveLesson(activeLesson === i ? null : i)}
            >
              <span className={`lesson-icon n${i}`}>▤</span>
              <span>
                <b>{x[0]}</b>
                <small>
                  {x[1]} · {i === 0 ? "Recommended for you" : "8 min lesson"}
                </small>
              </span>
              <i>
                <b style={{ width: x[2] }} />
              </i>
              <strong>{x[2]}</strong>
              <em>→</em>
            </button>
          ))}
        </div>
        {activeLesson !== null && (
          <div className="panel lesson-workspace">
            <span className="eyebrow">
              INTERACTIVE LESSON · {lessons[activeLesson][1]}
            </span>
            <h2>{lessons[activeLesson][0]}</h2>
            <p>
              {activeLesson === 0
                ? "PE tells you how much the market asks for each rupee of earnings. PB compares price with accounting net worth. PEG tests whether the price paid is proportionate to expected growth. None is a verdict alone: compare the company with its sector, quality, balance sheet and portfolio role."
                : activeLesson === 1
                  ? "Sharpe measures excess return per unit of volatility. A higher value can indicate more efficient risk-taking, but unstable history, changing regimes and concentrated exposures still matter."
                  : activeLesson === 2
                    ? "Drawdown measures the fall from a previous peak. Risk management asks whether the portfolio can survive that path—not merely whether the final forecast is attractive."
                    : "Diversification is about independent economic drivers, not the number of ticker symbols. Holdings in different sectors can still respond to the same macro shock."}
            </p>
            <button
              className="primary"
              onClick={() => setView(campaign ? "market" : "home")}
            >
              {campaign ? "OPEN MARKET DESK →" : "START A CAMPAIGN →"}
            </button>
          </div>
        )}
      </section>
    );
  if (view === "leaderboard")
    return (
      <section className="page">
        <div className="title-block">
          <span className="eyebrow">SCENARIO-NORMALISED LEAGUES</span>
          <h1>Skill—not capital—sets the rank.</h1>
          <p>
            Showcase leaderboard. Production rankings compare the same horizon,
            capital band and scenario family.
          </p>
        </div>
        <div className="podium">
          <div>
            <span>2</span>
            <i>NS</i>
            <b>Neha Trader</b>
            <strong>1680</strong>
          </div>
          <div className="winner">
            <span>1</span>
            <i>RI</i>
            <b>Rohit Investo</b>
            <strong>1824</strong>
          </div>
          <div>
            <span>3</span>
            <i>VK</i>
            <b>Vikram Kumar</b>
            <strong>1650</strong>
          </div>
        </div>
        <div className="panel rankings">
          {[
            [4, "Arjun Trader", 1450],
            [5, "Meera Shah", 1421],
            [6, "Karan Malhotra", 1398],
          ].map((x) => (
            <div>
              <b>#{x[0]}</b>
              <span className="avatar">{String(x[1]).slice(0, 2)}</span>
              <strong>{x[1]}</strong>
              <em>{x[2]}</em>
            </div>
          ))}
        </div>
      </section>
    );
  if (view === "news")
    return (
      <section className="page">
        <div className="title-block">
          <span className="eyebrow">SIMULATED MARKET EVENTS</span>
          <h1>The world moved. What matters?</h1>
          <p>
            Scenario cards are generated for this campaign and are not live
            news.
          </p>
        </div>
        <div className="news-grid">
          {[
            [
              "Rates remain unchanged",
              "Neutral stance keeps borrowing expectations steady",
              "Macro",
            ],
            [
              "Coal India beats estimates",
              "Operating performance improves in this simulated quarter",
              "Energy",
            ],
            [
              "Oil prices fall on demand concerns",
              "Upstream and downstream exposures may diverge",
              "Global",
            ],
            [
              "US technology rallies",
              "Indian technology sentiment receives a tailwind",
              "Technology",
            ],
            [
              "Large-cap buyback announced",
              "Capital allocation signal enters the market",
              "Corporate",
            ],
          ].map((x, i) => (
            <button className="news-card">
              <span className={`news-icon n${i}`}>◈</span>
              <span>
                <small>{x[2]} · SIMULATED</small>
                <b>{x[0]}</b>
                <p>{x[1]}</p>
              </span>
              <em>→</em>
            </button>
          ))}
        </div>
      </section>
    );
  if (view === "profile")
    return (
      <section className="page">
        <div className="profile-hero">
          <span className="avatar large">AT</span>
          <div>
            <span className="eyebrow">PLAYER PROFILE</span>
            <h1>Arjun Trader</h1>
            <p>Club player · member since August 2026</p>
          </div>
          <strong>1450</strong>
        </div>
        <div className="fundamentals">
          <Metric label="CAMPAIGNS" value="12" />
          <Metric label="COMPLETED" value="9" />
          <Metric label="WIN RATE" value="62%" />
          <Metric label="BEST RATING" value="1528" tone="gold" />
          <Metric label="BEST ALPHA" value="+18.7%" tone="positive" />
          <Metric label="PUZZLES" value="25" />
          <Metric label="LESSONS" value="8" />
          <Metric label="CURRENT STREAK" value="3" />
        </div>
      </section>
    );
  return null;
}

export default function App() {
  const [view, setView] = useState<View>("home"),
    [campaign, setCampaign] = useState<Campaign | null>(null),
    [market, setMarket] = useState<Market | null>(null),
    [stock, setStock] = useState<Stock | null>(null),
    [trades, setTrades] = useState<Trade[]>([]),
    [result, setResult] = useState<MoveResult | null>(null),
    [reviewData, setReviewData] = useState<MoveReview | null>(null),
    [busy, setBusy] = useState(true),
    [error, setError] = useState("");
  useEffect(() => {
    (async () => {
      try {
        await ensureSession();
        const cs = await api.campaigns();
        if (cs[0]) {
          setCampaign(cs[0]);
          setMarket(await api.market(cs[0].campaign_id));
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : "BTI server unavailable");
      } finally {
        setBusy(false);
      }
    })();
  }, []);
  const start = async (a: number, h: number) => {
    setBusy(true);
    try {
      const c = await api.create(a, h);
      setCampaign(c);
      setMarket(await api.market(c.campaign_id));
      setTrades([]);
      setResult(null);
      setReviewData(null);
      setView("market");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not create campaign");
    } finally {
      setBusy(false);
    }
  };
  const select = (s: Stock) => {
    setStock(s);
    setView("stock");
  };
  const add = (s: Stock) => {
    if (!trades.some((t) => t.ticker === s.ticker))
      setTrades([...trades, { side: "BUY", ticker: s.ticker, shares: 1 }]);
  };
  const execute = async () => {
    if (!campaign) return;
    setBusy(true);
    try {
      const x = await api.commit(
        campaign.campaign_id,
        campaign.current_move,
        trades,
      );
      setCampaign(x.campaign);
      setResult(x.result);
      setReviewData(null);
      setMarket(await api.market(x.campaign.campaign_id));
      setTrades([]);
      setView("review");
    } catch (e) {
      if (e instanceof APIRequestError && e.code === "SESSION_RESET") {
        setCampaign(null);
        setMarket(null);
        setStock(null);
        setTrades([]);
        setView("home");
      }
      setError(e instanceof Error ? e.message : "Move failed");
    } finally {
      setBusy(false);
    }
  };
  const reviewMove = async (move: number) => {
    if (!campaign) return;
    setBusy(true);
    try {
      setReviewData(await api.reviewMove(campaign.campaign_id, move));
      setView("review");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Move history unavailable");
    } finally {
      setBusy(false);
    }
  };
  const continueGame = () => {
    setResult(null);
    setReviewData(null);
    setView("market");
  };
  const content = useMemo(() => {
    if (busy && !campaign)
      return (
        <div className="loading">
          <i />
          <b>Preparing the decision universe…</b>
          <span>Vriddhi intelligence stays server-side.</span>
        </div>
      );
    if (error && !campaign)
      return (
        <div className="connection">
          <Badge />
          <h1>The game engine is offline.</h1>
          <p>{error}</p>
          <code>
            Reload the page. If this persists, the simulation service is
            restarting.
          </code>
        </div>
      );
    if (view === "home") return <Home campaign={campaign} setView={setView} />;
    if (view === "setup") return <Setup start={start} />;
    if (view === "market" && campaign && market)
      return (
        <MarketTerminal
          market={market}
          campaign={campaign}
          select={select}
          buildMove={() => {
            setResult(null);
            setReviewData(null);
            setView("review");
          }}
          trades={trades}
          setTrades={setTrades}
        />
      );
    if (view === "stock" && stock && market && campaign)
      return (
        <StockResearch
          stock={stock}
          market={market}
          campaign={campaign}
          add={add}
          back={() => setView("market")}
          openWorkbench={() => setView("market")}
        />
      );
    if (view === "review" && campaign && market)
      return (
        <GameBoard
          campaign={campaign}
          market={market}
          trades={trades}
          result={result}
          reviewData={reviewData}
          execute={execute}
          busy={busy}
          editMove={() => setView("market")}
          continueGame={continueGame}
          reviewMove={reviewMove}
          returnLive={() => setReviewData(null)}
        />
      );
    if (view === "result" && result && campaign)
      return <Result result={result} campaign={campaign} setView={setView} />;
    if (view === "progress" && campaign)
      return <Progress campaign={campaign} />;
    return <Explore view={view} setView={setView} campaign={campaign} />;
  }, [view, campaign, market, stock, trades, result, reviewData, busy, error]);
  return (
    <Shell view={view} setView={setView} campaign={campaign}>
      {error && campaign && (
        <div className="toast">
          {error}
          <button onClick={() => setError("")}>×</button>
        </div>
      )}
      {content}
    </Shell>
  );
}
