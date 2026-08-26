import { useMemo, useState } from "react";
import type { Campaign, Market, Stock, Trade } from "./api";
import {
  buildDraftPortfolio,
  draftDeltaFor,
  setDraftDelta,
} from "./portfolioDraft";

type Lens =
  | "ALL"
  | "VALUE"
  | "QUALITY"
  | "QUANT"
  | "TECHNICAL"
  | "NEWS"
  | "HELD";

type Props = {
  market: Market;
  campaign: Campaign;
  select: (stock: Stock) => void;
  buildMove: () => void;
  trades: Trade[];
  setTrades: (trades: Trade[]) => void;
};

const money = (paise: number, compact = false) => {
  const value = paise / 100;
  if (compact && Math.abs(value) >= 100_000)
    return `₹${(value / 100_000).toFixed(1)}L`;
  return `₹${Math.round(value).toLocaleString("en-IN")}`;
};
const signed = (value: number) =>
  `${value >= 0 ? "+" : ""}${value.toFixed(1)}%`;
const move = (stock: Stock) => {
  const monthlyWindow = stock.history_paise.slice(-22);
  const first = monthlyWindow[0] || stock.open_paise || 1;
  const last = monthlyWindow.at(-1) || stock.close_paise;
  return (last / first - 1) * 100;
};

function Sparkline({
  values,
  positive,
}: {
  values: number[];
  positive: boolean;
}) {
  if (values.length < 2) return null;
  const sample = values.slice(-32);
  const minimum = Math.min(...sample);
  const maximum = Math.max(...sample);
  const points = sample
    .map((value, index) => {
      const x = (index / (sample.length - 1)) * 100;
      const y = 30 - ((value - minimum) / (maximum - minimum || 1)) * 26;
      return `${x},${y}`;
    })
    .join(" ");
  return (
    <svg
      className="mobile-spark"
      viewBox="0 0 100 32"
      preserveAspectRatio="none"
      aria-hidden="true"
    >
      <polyline points={points} className={positive ? "up" : "down"} />
    </svg>
  );
}

function lensMetrics(stock: Stock, lens: Lens) {
  if (lens === "VALUE")
    return [
      ["PEG", stock.peg.toFixed(2)],
      ["PE", stock.pe.toFixed(1)],
      ["PB", stock.pb.toFixed(1)],
    ];
  if (lens === "QUALITY")
    return [
      ["ROE", `${stock.roe_pct.toFixed(1)}%`],
      ["MARGIN", `${stock.profit_margin_pct.toFixed(1)}%`],
      ["GROWTH", `${stock.earnings_growth_pct.toFixed(1)}%`],
    ];
  if (lens === "QUANT")
    return [
      ["SHARPE", stock.sharpe.toFixed(2)],
      ["VOL", `${stock.volatility_pct.toFixed(1)}%`],
      ["VaR", `${stock.var_95_pct.toFixed(1)}%`],
    ];
  if (lens === "TECHNICAL")
    return [
      ["RSI", stock.rsi_14.toFixed(0)],
      ["90D", signed(stock.momentum_90d_pct)],
      ["BETA", stock.beta.toFixed(2)],
    ];
  if (lens === "NEWS")
    return [
      ["SENTIMENT", `${stock.sentiment_score.toFixed(0)}/100`],
      ["VOLUME", stock.volume_index.toFixed(0)],
      ["SIM 12M", signed(stock.forecast_pct)],
    ];
  return [
    ["PEG", stock.peg.toFixed(2)],
    ["SHARPE", stock.sharpe.toFixed(2)],
    ["SIM 12M", signed(stock.forecast_pct)],
  ];
}

export function MobileMarket({
  market,
  campaign,
  select,
  buildMove,
  trades,
  setTrades,
}: Props) {
  const [lens, setLens] = useState<Lens>("ALL");
  const [query, setQuery] = useState("");
  const [moveSheet, setMoveSheet] = useState(false);
  const [intelligenceOpen, setIntelligenceOpen] = useState(false);
  const [repeatNotice, setRepeatNotice] = useState("");
  const draft = buildDraftPortfolio(campaign, market, trades);
  const analytics = useMemo(() => {
    const gainers = [...market.stocks]
      .sort((a, b) => move(b) - move(a))
      .slice(0, 5);
    const losers = [...market.stocks]
      .sort((a, b) => move(a) - move(b))
      .slice(0, 5);
    const marketPulse =
      market.stocks[0]?.history_paise.map((_, index) => {
        const values = market.stocks.map((stock) => {
          const start = stock.history_paise[0] || 1;
          return ((stock.history_paise[index] || start) / start) * 100;
        });
        return values.reduce((sum, value) => sum + value, 0) / values.length;
      }) || [];
    const median = (values: number[]) => {
      const sorted = [...values].sort((a, b) => a - b);
      return sorted[Math.floor(sorted.length / 2)] || 0;
    };
    return {
      gainers,
      losers,
      marketPulse,
      medianVar: median(market.stocks.map((stock) => stock.var_95_pct)),
      medianSharpe: median(market.stocks.map((stock) => stock.sharpe)),
      meanVolatility:
        market.stocks.reduce((sum, stock) => sum + stock.volatility_pct, 0) /
        market.stocks.length,
    };
  }, [market]);
  const stocks = useMemo(() => {
    const filtered = market.stocks.filter((stock) =>
      `${stock.ticker} ${stock.sector}`
        .toLowerCase()
        .includes(query.toLowerCase()),
    );
    const held =
      lens === "HELD"
        ? filtered.filter((stock) => campaign.holdings[stock.ticker])
        : filtered;
    return [...held].sort((a, b) => {
      if (lens === "VALUE") return (a.peg || 999) - (b.peg || 999);
      if (lens === "QUALITY")
        return b.roe_pct + b.sharpe * 10 - (a.roe_pct + a.sharpe * 10);
      if (lens === "QUANT")
        return (
          b.sharpe - b.volatility_pct / 30 - (a.sharpe - a.volatility_pct / 30)
        );
      if (lens === "TECHNICAL") return b.momentum_90d_pct - a.momentum_90d_pct;
      if (lens === "NEWS") return b.sentiment_score - a.sentiment_score;
      return a.overall_rank - b.overall_rank;
    });
  }, [market, campaign.holdings, lens, query]);
  const adjust = (ticker: string, increment: number) => {
    const held = campaign.holdings[ticker] || 0;
    setTrades(
      setDraftDelta(
        trades,
        ticker,
        draftDeltaFor(trades, ticker) + increment,
        held,
      ),
    );
  };
  const leading = campaign.match_summary.position === "LEADING";
  return (
    <section className="mobile-market-page">
      <header className="mobile-match-card">
        <div>
          <small>
            RATED CAMPAIGN · MOVE {campaign.current_move}/
            {campaign.horizon_months}
          </small>
          <h1>
            {leading ? "You are ahead of Nifty." : "Nifty sets the target."}
          </h1>
          <p>
            {market.regime?.label || campaign.current_regime.label} ·{" "}
            {market.regime?.narrative || campaign.current_regime.narrative}
          </p>
        </div>
        <div className="mobile-match-numbers">
          <span>
            <small>PLAYER</small>
            <b>{money(campaign.portfolio_value_paise, true)}</b>
          </span>
          <span>
            <small>NIFTY</small>
            <b>{money(campaign.benchmark_value_paise, true)}</b>
          </span>
          <span>
            <small>GAP</small>
            <b className={campaign.alpha_pct >= 0 ? "positive" : "negative"}>
              {signed(campaign.alpha_pct)}
            </b>
          </span>
        </div>
      </header>

      <section className="mobile-market-brief">
        <div>
          <span className="live-dot" /> MARKET BRIEFING
        </div>
        <h2>{market.regime?.label || "Simulated market in motion"}</h2>
        <p>
          Every signal below was visible before your move. Momentum attracts
          attention; portfolio quality wins the match.
        </p>
        {campaign.can_repeat_last_move && (
          <button
            className="mobile-repeat-move"
            onClick={() => {
              setTrades(
                campaign.last_move_instructions.map((trade) => ({ ...trade })),
              );
              setRepeatNotice(
                "Last month’s whole-share move is staged. This market is different—recheck every signal before committing.",
              );
            }}
          >
            ↻ REPEAT LAST MOVE
          </button>
        )}
      </section>
      {repeatNotice && (
        <div className="mobile-repeat-warning">
          <span>{repeatNotice}</span>
          <button onClick={() => setRepeatNotice("")}>×</button>
        </div>
      )}

      <button
        className={`mobile-intelligence-card ${intelligenceOpen ? "open" : ""}`}
        onClick={() => setIntelligenceOpen((open) => !open)}
      >
        <div>
          <span>
            <small>MARKET PULSE</small>
            <b>SIMULATED NIFTY 50</b>
          </span>
          <strong>
            {intelligenceOpen ? "COLLAPSE −" : "OPEN RISK DESK +"}
          </strong>
        </div>
        <Sparkline values={analytics.marketPulse} positive />
        <section>
          <span>
            <small>MEDIAN VaR</small>
            <b>{analytics.medianVar.toFixed(1)}%</b>
          </span>
          <span>
            <small>MARKET VOL</small>
            <b>{analytics.meanVolatility.toFixed(1)}%</b>
          </span>
          <span>
            <small>MEDIAN SHARPE</small>
            <b>{analytics.medianSharpe.toFixed(2)}</b>
          </span>
        </section>
        {intelligenceOpen && (
          <p>
            The pulse aggregates all 50 generated price paths. Risk metrics
            describe the current decision universe—not the hidden optimal move.
          </p>
        )}
      </button>

      <section className="mobile-movers">
        <div className="mobile-section-title">
          <b>MONTHLY MOVERS</b>
          <small>SIMULATED</small>
        </div>
        <div className="mobile-mover-strip">
          {analytics.gainers.map((stock) => (
            <button key={`g-${stock.ticker}`} onClick={() => select(stock)}>
              <small>GAINER</small>
              <b>{stock.ticker}</b>
              <strong className="positive">{signed(move(stock))}</strong>
            </button>
          ))}
          {analytics.losers.map((stock) => (
            <button key={`l-${stock.ticker}`} onClick={() => select(stock)}>
              <small>LOSER</small>
              <b>{stock.ticker}</b>
              <strong className="negative">{signed(move(stock))}</strong>
            </button>
          ))}
        </div>
      </section>

      <section
        className="mobile-headline-card"
        onClick={() => select(analytics.gainers[0])}
        role="button"
        tabIndex={0}
      >
        <small>SIMULATED NEWS · ATTENTION IS NOT ALPHA</small>
        <h2>
          {analytics.gainers[0]?.ticker} leads the tape as the market rewards
          visible momentum.
        </h2>
        <p>
          Open the evidence before deciding whether the headline improves the
          complete portfolio.
        </p>
      </section>

      <div className="mobile-stock-tools">
        <label>
          ⌕
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search stocks or sectors"
          />
        </label>
        <div className="mobile-lenses">
          {(
            [
              "ALL",
              "VALUE",
              "QUALITY",
              "QUANT",
              "TECHNICAL",
              "NEWS",
              "HELD",
            ] as Lens[]
          ).map((item) => (
            <button
              key={item}
              className={lens === item ? "active" : ""}
              onClick={() => setLens(item)}
            >
              {item}
            </button>
          ))}
        </div>
        <p>
          <b>{lens === "ALL" ? "DECISION FEED" : `${lens} LENS`}</b> ·{" "}
          {stocks.length} securities · tap a stock for the complete research
          desk
        </p>
      </div>

      <div className="mobile-stock-feed">
        {stocks.map((stock) => {
          const change = move(stock);
          const held = campaign.holdings[stock.ticker] || 0;
          const delta = draftDeltaFor(trades, stock.ticker);
          const after = held + delta;
          return (
            <article className="mobile-stock-card" key={stock.ticker}>
              <button
                className="mobile-stock-open"
                onClick={() => select(stock)}
              >
                <div>
                  <i>{stock.ticker.slice(0, 2)}</i>
                  <span>
                    <b>{stock.ticker}</b>
                    <small>
                      {stock.sector} · {held ? `HELD ${held}` : "NOT HELD"}
                    </small>
                  </span>
                </div>
                <div>
                  <b>{money(stock.close_paise)}</b>
                  <strong className={change >= 0 ? "positive" : "negative"}>
                    {signed(change)}
                  </strong>
                </div>
              </button>
              <button
                className="mobile-chart-open"
                onClick={() => select(stock)}
                aria-label={`Open ${stock.ticker} research`}
              >
                <Sparkline
                  values={stock.history_paise}
                  positive={change >= 0}
                />
              </button>
              <div className="mobile-card-metrics">
                {lensMetrics(stock, lens).map(([label, value]) => (
                  <span key={label}>
                    <small>{label}</small>
                    <b>{value}</b>
                  </span>
                ))}
              </div>
              <div className="mobile-position-stepper">
                <span>
                  <small>HELD</small>
                  <b>{held}</b>
                </span>
                <button
                  onClick={() => adjust(stock.ticker, -1)}
                  disabled={after <= 0}
                >
                  −
                </button>
                <div>
                  <small>AFTER MOVE</small>
                  <b>{after}</b>
                  {delta !== 0 && <em>{delta > 0 ? `+${delta}` : delta}</em>}
                </div>
                <button onClick={() => adjust(stock.ticker, 1)}>+</button>
                {delta !== 0 && (
                  <button
                    className="mobile-hold"
                    onClick={() =>
                      setTrades(setDraftDelta(trades, stock.ticker, 0, held))
                    }
                  >
                    HOLD
                  </button>
                )}
              </div>
            </article>
          );
        })}
      </div>

      <button className="mobile-move-dock" onClick={() => setMoveSheet(true)}>
        <span>
          <small>LIVE MOVE</small>
          <b>
            {trades.length} ORDERS · {money(draft.buyTotalPaise, true)} DEPLOYED
          </b>
        </span>
        <strong>{money(draft.cashAfterPaise)} →</strong>
      </button>

      {moveSheet && (
        <div className="mobile-sheet-backdrop">
          <section className="mobile-move-sheet">
            <header>
              <div>
                <small>MOVE {campaign.current_move}</small>
                <h2>Your complete portfolio move</h2>
              </div>
              <button onClick={() => setMoveSheet(false)}>×</button>
            </header>
            <div className="mobile-xray-title">
              <b>PORTFOLIO X-RAY</b>
              <small>PROJECTED AFTER THIS MOVE</small>
            </div>
            <div className="mobile-health-strip">
              <span>
                <small>SHARPE</small>
                <b>{draft.weighted.sharpe.toFixed(2)}</b>
              </span>
              <span>
                <small>PEG</small>
                <b>{draft.weighted.peg.toFixed(2)}</b>
              </span>
              <span>
                <small>TOP WEIGHT</small>
                <b>{draft.concentrationPct.toFixed(0)}%</b>
              </span>
              <span>
                <small>SECTORS</small>
                <b>{draft.sectors.length}</b>
              </span>
            </div>
            <div className="mobile-sheet-orders">
              {trades.map((trade) => (
                <div key={trade.ticker}>
                  <b className={trade.side === "BUY" ? "positive" : "negative"}>
                    {trade.side}
                  </b>
                  <span>
                    {trade.shares} {trade.ticker}
                  </span>
                  <strong>
                    {money(
                      (market.stocks.find(
                        (stock) => stock.ticker === trade.ticker,
                      )?.close_paise || 0) * trade.shares,
                    )}
                  </strong>
                  <button
                    onClick={() =>
                      setTrades(
                        setDraftDelta(
                          trades,
                          trade.ticker,
                          0,
                          campaign.holdings[trade.ticker] || 0,
                        ),
                      )
                    }
                  >
                    ×
                  </button>
                </div>
              ))}
              {!trades.length && (
                <p>
                  Research the market, then use − / + to construct this month’s
                  move.
                </p>
              )}
            </div>
            <div className="mobile-sector-list">
              {draft.sectors.slice(0, 6).map(([sector, value]) => (
                <div key={sector}>
                  <span>{sector}</span>
                  <i>
                    <b
                      style={{
                        width: `${draft.investedPaise ? (value / draft.investedPaise) * 100 : 0}%`,
                      }}
                    />
                  </i>
                  <strong>
                    {draft.investedPaise
                      ? ((value / draft.investedPaise) * 100).toFixed(0)
                      : 0}
                    %
                  </strong>
                </div>
              ))}
            </div>
            <div className="mobile-reconcile">
              <span>
                <small>DEPLOYMENT</small>
                <b>{draft.deploymentPct.toFixed(1)}%</b>
              </span>
              <span>
                <small>CASH AFTER</small>
                <b>{money(draft.cashAfterPaise)}</b>
              </span>
            </div>
            <button
              className="primary full"
              disabled={!trades.length}
              onClick={buildMove}
            >
              REVIEW RATED MOVE →
            </button>
          </section>
        </div>
      )}
    </section>
  );
}
