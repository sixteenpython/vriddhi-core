import { useEffect, useMemo, useRef, useState } from "react";
import type { Campaign, Market, Stock, Trade } from "./api";
import {
  buildDraftPortfolio,
  draftDeltaFor,
  normalizeTrades,
  setDraftDelta,
} from "./portfolioDraft";
import { IntelligenceDeck, PortfolioRibbon } from "./Cockpit";

type BaseProps = { market: Market; campaign: Campaign };
type MarketProps = BaseProps & {
  select: (stock: Stock) => void;
  buildMove: () => void;
  trades: Trade[];
  setTrades: (trades: Trade[]) => void;
};
type PortfolioProps = BaseProps & {
  trades: Trade[];
  setTrades: (trades: Trade[]) => void;
  review: () => void;
  backToMarket: () => void;
};
type SortKey =
  | "ticker"
  | "price"
  | "change"
  | "position"
  | "pe"
  | "pb"
  | "peg"
  | "sharpe"
  | "volatility"
  | "drawdown"
  | "var"
  | "forecast";
type ColumnFilters = {
  sector: string;
  position: string;
  peMax: string;
  pbMax: string;
  pegMax: string;
  sharpeMin: string;
  volatilityMax: string;
  drawdownMax: string;
  forecastMin: string;
};

const rupees = (paise: number, compact = false) => {
  const value = paise / 100;
  if (compact && value >= 10_000_000)
    return `₹${(value / 10_000_000).toFixed(2)}Cr`;
  if (compact && value >= 100_000) return `₹${(value / 100_000).toFixed(1)}L`;
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(value);
};
const signed = (value: number, digits = 2) =>
  `${value >= 0 ? "+" : ""}${value.toFixed(digits)}%`;
const change = (stock: Stock) =>
  (stock.close_paise / stock.open_paise - 1) * 100;
const median = (values: number[]) => {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted.length ? sorted[Math.floor(sorted.length / 2)] : 0;
};
const path = (raw: number[], width = 160, height = 42) => {
  const values = raw.length > 1 ? raw : [raw[0] || 1, raw[0] || 1];
  const low = Math.min(...values);
  const high = Math.max(...values);
  const spread = high - low || 1;
  return values
    .map(
      (value, index) =>
        `${index ? "L" : "M"}${(index / (values.length - 1)) * width},${height - ((value - low) / spread) * height}`,
    )
    .join(" ");
};

function MiniChart({
  values,
  tone = "green",
}: {
  values: number[];
  tone?: "green" | "red" | "purple" | "cyan";
}) {
  return (
    <svg
      className={`terminal-spark ${tone}`}
      viewBox="0 0 160 42"
      preserveAspectRatio="none"
      aria-hidden="true"
    >
      <path d={path(values)} />
    </svg>
  );
}

const stories = [
  {
    tag: "MONETARY POLICY",
    title: "Rates held steady as the simulated policy path stays neutral",
    body: "Financials keep their funding-cost tailwind, while expensive duration-sensitive names still need earnings support.",
    impact: "Banks · NBFC · Realty",
    tone: "purple",
    age: "SIM 08:42",
  },
  {
    tag: "ENERGY",
    title: "Crude eases; refiners gain a margin cushion",
    body: "Lower input costs improve the near-term setup for downstream energy while upstream sensitivity softens.",
    impact: "BPCL · ONGC · RELIANCE",
    tone: "amber",
    age: "SIM 08:18",
  },
  {
    tag: "EARNINGS",
    title: "Capital-goods order books remain resilient",
    body: "Visibility improves, but premium multiples leave little room for execution misses.",
    impact: "BEL · LT · ADANIPORTS",
    tone: "cyan",
    age: "SIM 07:55",
  },
  {
    tag: "GLOBAL TECH",
    title: "Risk appetite returns to technology after a weak quarter",
    body: "The simulated global cue helps sentiment; domestic valuation and growth quality remain the deciding filters.",
    impact: "TCS · INFY · HCLTECH",
    tone: "green",
    age: "SIM 07:31",
  },
  {
    tag: "CONSUMER",
    title: "Rural demand improves while urban premium demand cools",
    body: "Volume recovery helps staples, but high PEG names still carry expectation risk.",
    impact: "ITC · NESTLEIND · TITAN",
    tone: "red",
    age: "SIM 06:50",
  },
];

export function MarketTerminal({
  market,
  campaign,
  select,
  buildMove,
  trades,
  setTrades,
}: MarketProps) {
  const [query, setQuery] = useState("");
  const [repeatNotice, setRepeatNotice] = useState("");
  const [filter, setFilter] = useState("ALL");
  const [sort, setSort] = useState<{ key: SortKey; direction: "asc" | "desc" }>(
    {
      key: "ticker",
      direction: "asc",
    },
  );
  const [columns, setColumns] = useState<ColumnFilters>({
    sector: "ALL",
    position: "ALL",
    peMax: "",
    pbMax: "",
    pegMax: "",
    sharpeMin: "",
    volatilityMax: "",
    drawdownMax: "",
    forecastMin: "",
  });
  const topScrollRef = useRef<HTMLDivElement>(null);
  const tableScrollRef = useRef<HTMLDivElement>(null);
  const analytics = useMemo(() => {
    const advances = market.stocks.filter((stock) => change(stock) > 0).length;
    const declines = market.stocks.filter((stock) => change(stock) < 0).length;
    const sectors = Object.entries(
      market.stocks.reduce<Record<string, Stock[]>>((groups, stock) => {
        (groups[stock.sector] ||= []).push(stock);
        return groups;
      }, {}),
    )
      .map(([sector, stocks]) => ({
        sector,
        move:
          stocks.reduce((sum, stock) => sum + change(stock), 0) / stocks.length,
        forecast:
          stocks.reduce((sum, stock) => sum + stock.forecast_pct, 0) /
          stocks.length,
        sharpe:
          stocks.reduce((sum, stock) => sum + stock.sharpe, 0) / stocks.length,
      }))
      .sort((a, b) => b.forecast - a.forecast);
    const ranked = [...market.stocks].sort(
      (a, b) =>
        b.forecast_pct +
        b.sharpe * 4 -
        b.peg * 0.6 -
        (a.forecast_pct + a.sharpe * 4 - a.peg * 0.6),
    );
    const gainers = [...market.stocks]
      .filter((stock) => change(stock) > 0)
      .sort((a, b) => change(b) - change(a));
    const losers = [...market.stocks]
      .filter((stock) => change(stock) < 0)
      .sort((a, b) => change(a) - change(b));
    const attention = [...market.stocks].sort(
      (a, b) =>
        Math.abs(change(b)) * 2.2 +
        b.sentiment_score / 10 +
        b.volume_index / 35 -
        (Math.abs(change(a)) * 2.2 +
          a.sentiment_score / 10 +
          a.volume_index / 35),
    );
    return {
      advances,
      declines,
      medianPeg: median(
        market.stocks.map((stock) => stock.peg).filter((peg) => peg > 0),
      ),
      meanForecast:
        market.stocks.reduce((sum, stock) => sum + stock.forecast_pct, 0) /
        market.stocks.length,
      meanVolatility:
        market.stocks.reduce((sum, stock) => sum + stock.volatility_pct, 0) /
        market.stocks.length,
      sectors,
      ranked,
      gainers,
      losers,
      attention,
    };
  }, [market]);
  const sectors = [
    ...new Set(market.stocks.map((stock) => stock.sector)),
  ].sort();
  const numberPass = (value: number, raw: string, mode: "min" | "max") =>
    raw === "" ||
    (mode === "min" ? value >= Number(raw) : value <= Number(raw));
  const sortValue = (stock: Stock, key: SortKey): number | string => {
    if (key === "ticker") return stock.ticker;
    if (key === "price") return stock.close_paise;
    if (key === "change") return change(stock);
    if (key === "position")
      return (
        (campaign.holdings[stock.ticker] ? 2 : 0) +
        (draftDeltaFor(trades, stock.ticker) !== 0 ? 1 : 0)
      );
    if (key === "volatility") return stock.volatility_pct;
    if (key === "drawdown") return stock.drawdown_pct;
    if (key === "var") return stock.var_95_pct;
    if (key === "forecast") return stock.forecast_pct;
    return stock[key];
  };
  const chooseSort = (key: SortKey) =>
    setSort((current) => ({
      key,
      direction:
        current.key === key && current.direction === "desc" ? "asc" : "desc",
    }));
  const sortMark = (key: SortKey) =>
    sort.key === key ? (sort.direction === "asc" ? "▲" : "▼") : "↕";
  const updateColumn = (key: keyof ColumnFilters, value: string) =>
    setColumns((current) => ({ ...current, [key]: value }));
  const clearColumns = () =>
    setColumns({
      sector: "ALL",
      position: "ALL",
      peMax: "",
      pbMax: "",
      pegMax: "",
      sharpeMin: "",
      volatilityMax: "",
      drawdownMax: "",
      forecastMin: "",
    });
  const draft = buildDraftPortfolio(campaign, market, trades);
  const visible = [...market.stocks]
    .filter((stock) =>
      `${stock.ticker} ${stock.sector}`
        .toLowerCase()
        .includes(query.toLowerCase()),
    )
    .filter((stock) => {
      if (filter === "VALUE") return stock.peg > 0 && stock.peg <= 1.2;
      if (filter === "QUALITY") return stock.sharpe >= 1;
      if (filter === "LOW RISK")
        return stock.volatility_pct <= analytics.meanVolatility;
      if (filter === "HELD") return Boolean(campaign.holdings[stock.ticker]);
      return true;
    })
    .filter(
      (stock) =>
        (columns.sector === "ALL" || stock.sector === columns.sector) &&
        (columns.position === "ALL" ||
          (columns.position === "HELD" &&
            Boolean(campaign.holdings[stock.ticker])) ||
          (columns.position === "DRAFTED" &&
            draftDeltaFor(trades, stock.ticker) !== 0) ||
          (columns.position === "HELD + DRAFTED" &&
            (Boolean(campaign.holdings[stock.ticker]) ||
              draftDeltaFor(trades, stock.ticker) !== 0)) ||
          (columns.position === "NOT HELD" &&
            !campaign.holdings[stock.ticker])) &&
        numberPass(stock.pe, columns.peMax, "max") &&
        numberPass(stock.pb, columns.pbMax, "max") &&
        numberPass(stock.peg, columns.pegMax, "max") &&
        numberPass(stock.sharpe, columns.sharpeMin, "min") &&
        numberPass(stock.volatility_pct, columns.volatilityMax, "max") &&
        numberPass(Math.abs(stock.drawdown_pct), columns.drawdownMax, "max") &&
        numberPass(stock.forecast_pct, columns.forecastMin, "min"),
    )
    .sort((a, b) => {
      const left = sortValue(a, sort.key);
      const right = sortValue(b, sort.key);
      const comparison =
        typeof left === "string"
          ? left.localeCompare(String(right))
          : left - Number(right);
      return sort.direction === "asc" ? comparison : -comparison;
    });
  const syncScroll = (
    source: HTMLDivElement,
    target: HTMLDivElement | null,
  ) => {
    if (target && Math.abs(target.scrollLeft - source.scrollLeft) > 1)
      target.scrollLeft = source.scrollLeft;
  };
  const adjustDraft = (ticker: string, increment: number) => {
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
  return (
    <section className="terminal-page">
      <div className="terminal-commandbar">
        <div>
          <span className="live-dot" /> MARKET WORKSTATION{" "}
          <b>BTI-SIM/{campaign.campaign_id}</b>
        </div>
        <div className="terminal-clock">
          SIM SESSION · M{campaign.current_move}/{campaign.horizon_months} ·
          FEED HEALTH 100%
        </div>
        <div className="market-turn-actions">
          {campaign.can_repeat_last_move && (
            <button
              className="repeat-move"
              onClick={() => {
                setTrades(
                  campaign.last_move_instructions.map((trade) => ({
                    ...trade,
                  })),
                );
                setRepeatNotice(
                  "Previous whole-share instructions copied exactly. This market is different—review affordability, holdings and portfolio health before committing.",
                );
              }}
            >
              ↻ REPEAT LAST MOVE
            </button>
          )}
          <button onClick={buildMove}>
            REVIEW MOVE <b>F9</b>
          </button>
        </div>
      </div>
      <div className="market-regime-strip">
        <span>MARKET COUNTERMOVE</span>
        <b>{market.regime?.label || campaign.current_regime.label}</b>
        <p>{market.regime?.narrative || campaign.current_regime.narrative}</p>
        <em>
          DIFFICULTY{" "}
          {(
            market.regime?.difficulty || campaign.current_regime.difficulty
          ).toFixed(2)}
          ×
        </em>
      </div>
      {repeatNotice && (
        <div className="repeat-warning">
          <b>REPEATED MOVE · NOT RECOMMENDED BLINDLY</b>
          <span>{repeatNotice}</span>
          <button onClick={() => setRepeatNotice("")}>×</button>
        </div>
      )}
      <PortfolioRibbon
        campaign={campaign}
        draft={draft}
        trades={trades}
        openWorkbench={buildMove}
      />
      <IntelligenceDeck market={market} />
      <div className="terminal-grid">
        <div className="terminal-main-shell">
          <div className="terminal-toolbar">
            <div className="terminal-search">
              ⌕{" "}
              <input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Command / ticker / sector"
              />
            </div>
            <div className="terminal-filters">
              {["ALL", "VALUE", "QUALITY", "LOW RISK", "HELD"].map((item) => (
                <button
                  key={item}
                  className={filter === item ? "active" : ""}
                  onClick={() => setFilter(item)}
                >
                  {item}
                </button>
              ))}
            </div>
            <span className="filter-status">
              {visible.length}/{market.stocks.length} ROWS · SORT{" "}
              {sort.key.toUpperCase()} {sort.direction === "asc" ? "▲" : "▼"}
            </span>
            <button className="clear-filters" onClick={clearColumns}>
              CLEAR COLUMN FILTERS
            </button>
          </div>
          <div
            className="table-scroll-top"
            ref={topScrollRef}
            onScroll={(event) =>
              syncScroll(event.currentTarget, tableScrollRef.current)
            }
            aria-label="Top horizontal market-table navigation"
          >
            <div className="table-scroll-spacer" />
          </div>
          <div
            className="terminal-main"
            ref={tableScrollRef}
            onScroll={(event) =>
              syncScroll(event.currentTarget, topScrollRef.current)
            }
          >
            <div className="pro-table">
              <div className="pro-row pro-head">
                <button onClick={() => chooseSort("ticker")}>
                  SECURITY {sortMark("ticker")}
                </button>
                <button onClick={() => chooseSort("price")}>
                  LAST {sortMark("price")}
                </button>
                <button onClick={() => chooseSort("change")}>
                  SIM Δ {sortMark("change")}
                </button>
                <button onClick={() => chooseSort("position")}>
                  POSITION / DRAFT {sortMark("position")}
                </button>
                <span>INTRAMONTH</span>
                <button onClick={() => chooseSort("pe")}>
                  PE {sortMark("pe")}
                </button>
                <button onClick={() => chooseSort("pb")}>
                  PB {sortMark("pb")}
                </button>
                <button onClick={() => chooseSort("peg")}>
                  PEG {sortMark("peg")}
                </button>
                <button onClick={() => chooseSort("sharpe")}>
                  SHARPE {sortMark("sharpe")}
                </button>
                <button onClick={() => chooseSort("volatility")}>
                  VOL {sortMark("volatility")}
                </button>
                <button onClick={() => chooseSort("drawdown")}>
                  DD {sortMark("drawdown")}
                </button>
                <button onClick={() => chooseSort("var")}>
                  VaR 95 {sortMark("var")}
                </button>
                <button onClick={() => chooseSort("forecast")}>
                  SIM 12M {sortMark("forecast")}
                </button>
              </div>
              <div className="pro-row pro-filter-row">
                <select
                  value={columns.sector}
                  onChange={(event) =>
                    updateColumn("sector", event.target.value)
                  }
                  aria-label="Filter sector"
                >
                  <option>ALL</option>
                  {sectors.map((sector) => (
                    <option key={sector}>{sector}</option>
                  ))}
                </select>
                <span />
                <span />
                <select
                  value={columns.position}
                  onChange={(event) =>
                    updateColumn("position", event.target.value)
                  }
                  aria-label="Filter portfolio position"
                >
                  <option>ALL</option>
                  <option>HELD</option>
                  <option>DRAFTED</option>
                  <option>HELD + DRAFTED</option>
                  <option>NOT HELD</option>
                </select>
                <small>
                  CLICK HEADERS
                  <br />
                  TO SORT
                </small>
                <input
                  value={columns.peMax}
                  onChange={(event) =>
                    updateColumn("peMax", event.target.value)
                  }
                  placeholder="≤ max"
                  inputMode="decimal"
                  aria-label="Maximum PE"
                />
                <input
                  value={columns.pbMax}
                  onChange={(event) =>
                    updateColumn("pbMax", event.target.value)
                  }
                  placeholder="≤ max"
                  inputMode="decimal"
                  aria-label="Maximum PB"
                />
                <input
                  value={columns.pegMax}
                  onChange={(event) =>
                    updateColumn("pegMax", event.target.value)
                  }
                  placeholder="≤ max"
                  inputMode="decimal"
                  aria-label="Maximum PEG"
                />
                <input
                  value={columns.sharpeMin}
                  onChange={(event) =>
                    updateColumn("sharpeMin", event.target.value)
                  }
                  placeholder="≥ min"
                  inputMode="decimal"
                  aria-label="Minimum Sharpe"
                />
                <input
                  value={columns.volatilityMax}
                  onChange={(event) =>
                    updateColumn("volatilityMax", event.target.value)
                  }
                  placeholder="≤ max"
                  inputMode="decimal"
                  aria-label="Maximum volatility"
                />
                <input
                  value={columns.drawdownMax}
                  onChange={(event) =>
                    updateColumn("drawdownMax", event.target.value)
                  }
                  placeholder="≤ abs"
                  inputMode="decimal"
                  aria-label="Maximum drawdown"
                />
                <span />
                <input
                  value={columns.forecastMin}
                  onChange={(event) =>
                    updateColumn("forecastMin", event.target.value)
                  }
                  placeholder="≥ min"
                  inputMode="decimal"
                  aria-label="Minimum forecast"
                />
              </div>
              {visible.map((stock) => {
                const move = change(stock);
                const held = campaign.holdings[stock.ticker] || 0;
                const delta = draftDeltaFor(trades, stock.ticker);
                const projected = held + delta;
                return (
                  <div className="pro-row" key={stock.ticker}>
                    <button
                      className="pro-security security-open"
                      onClick={() => select(stock)}
                    >
                      <i>{stock.ticker.slice(0, 2)}</i>
                      <span>
                        <b>{stock.ticker}</b>
                        <small>
                          {stock.sector}
                          {held ? ` · HELD ${held}` : " · NOT HELD"}
                        </small>
                      </span>
                    </button>
                    <strong>{rupees(stock.close_paise)}</strong>
                    <strong className={move >= 0 ? "positive" : "negative"}>
                      {signed(move)}
                    </strong>
                    <div
                      className="inline-position"
                      aria-label={`${stock.ticker} portfolio position`}
                    >
                      <span>
                        <small>HELD</small>
                        <b>{held}</b>
                      </span>
                      <button
                        onClick={() => adjustDraft(stock.ticker, -1)}
                        disabled={projected <= 0}
                      >
                        −
                      </button>
                      <input
                        aria-label={`${stock.ticker} draft share change`}
                        className={
                          delta > 0 ? "positive" : delta < 0 ? "negative" : ""
                        }
                        value={delta}
                        inputMode="numeric"
                        onChange={(event) =>
                          setTrades(
                            setDraftDelta(
                              trades,
                              stock.ticker,
                              Number(event.target.value) || 0,
                              held,
                            ),
                          )
                        }
                      />
                      <button onClick={() => adjustDraft(stock.ticker, 1)}>
                        +
                      </button>
                      <span>
                        <small>AFTER</small>
                        <b>{projected}</b>
                      </span>
                      {held > 0 && projected > 0 && (
                        <button
                          className="exit-position"
                          onClick={() =>
                            setTrades(
                              setDraftDelta(trades, stock.ticker, -held, held),
                            )
                          }
                        >
                          EXIT
                        </button>
                      )}
                      {delta !== 0 && (
                        <button
                          className="hold-position"
                          onClick={() =>
                            setTrades(
                              setDraftDelta(trades, stock.ticker, 0, held),
                            )
                          }
                        >
                          HOLD
                        </button>
                      )}
                    </div>
                    <MiniChart
                      values={stock.history_paise}
                      tone={move >= 0 ? "green" : "red"}
                    />
                    <span>{stock.pe.toFixed(1)}</span>
                    <span>{stock.pb.toFixed(1)}</span>
                    <span
                      className={
                        stock.peg > 0 && stock.peg <= 1 ? "positive" : ""
                      }
                    >
                      {stock.peg.toFixed(2)}
                    </span>
                    <span>{stock.sharpe.toFixed(2)}</span>
                    <span>{stock.volatility_pct.toFixed(1)}%</span>
                    <span className="negative">
                      {stock.drawdown_pct.toFixed(1)}%
                    </span>
                    <span>{stock.var_95_pct.toFixed(1)}%</span>
                    <strong className="positive">
                      +{stock.forecast_pct.toFixed(1)}%
                    </strong>
                  </div>
                );
              })}
              {!visible.length && (
                <div className="market-empty">
                  <b>NO SECURITIES MATCH THESE FILTERS</b>
                  <span>
                    Clear one or more column conditions to restore the decision
                    universe.
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>
        <aside className="terminal-side">
          <div className="terminal-panel draft-tray">
            <div className="panel-label">
              <span>LIVE MOVE TRAY</span>
              <small>{trades.length} NET ORDERS</small>
            </div>
            {trades.map((trade) => (
              <div key={trade.ticker}>
                <b className={trade.side === "BUY" ? "positive" : "negative"}>
                  {trade.side}
                </b>
                <span>
                  {trade.shares} {trade.ticker}
                </span>
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
                Use − / + in the market table to construct this month’s
                portfolio decision.
              </p>
            )}
            <button className="tray-review" onClick={buildMove}>
              RECONCILE IN WORKBENCH →
            </button>
          </div>
          <div className="terminal-panel opportunity-monitor">
            <div className="panel-label">
              <span>QUANT RADAR</span>
              <small>PUBLIC SIGNALS</small>
            </div>
            {analytics.ranked.slice(0, 5).map((stock, index) => (
              <button key={stock.ticker} onClick={() => select(stock)}>
                <b>{index + 1}</b>
                <span>
                  {stock.ticker}
                  <small>
                    PEG {stock.peg.toFixed(2)} · S {stock.sharpe.toFixed(2)}
                  </small>
                </span>
                <strong>+{stock.forecast_pct.toFixed(1)}%</strong>
              </button>
            ))}
          </div>
          <div className="terminal-panel portfolio-xray">
            <div className="panel-label">
              <span>PORTFOLIO X-RAY</span>
              <small>PROJECTED</small>
            </div>
            <div className="xray-grid">
              <span>
                PE <b>{draft.weighted.pe.toFixed(1)}</b>
              </span>
              <span>
                PEG <b>{draft.weighted.peg.toFixed(2)}</b>
              </span>
              <span>
                SHARPE <b>{draft.weighted.sharpe.toFixed(2)}</b>
              </span>
              <span>
                SIM 12M{" "}
                <b className="positive">{signed(draft.weighted.forecast)}</b>
              </span>
              <span>
                VOL <b>{draft.weighted.volatility.toFixed(1)}%</b>
              </span>
              <span>
                TOP WEIGHT{" "}
                <b
                  className={
                    draft.concentrationPct > 25 ? "negative" : "positive"
                  }
                >
                  {draft.concentrationPct.toFixed(1)}%
                </b>
              </span>
            </div>
            {draft.sectors.slice(0, 5).map(([sector, value]) => (
              <div className="xray-sector" key={sector}>
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
                    ? ((value / draft.investedPaise) * 100).toFixed(1)
                    : "0.0"}
                  %
                </strong>
              </div>
            ))}
          </div>
          <div className="terminal-panel flash-newswire">
            <div className="panel-label">
              <span>
                <i className="wire-live" /> BTI NEWSWIRE
              </span>
              <small>SIMULATED · LIVE DESK</small>
            </div>
            <div className="wire-ticker">
              <span>BREAKING</span>
              <b>
                {analytics.gainers[0]?.ticker || "MARKET"} LEADS ·{" "}
                {analytics.losers[0]?.ticker || "RISK"} LAGS · VOLATILITY{" "}
                {analytics.meanVolatility.toFixed(1)}%
              </b>
            </div>
            {stories.map((story, index) => (
              <article
                className={`wire-story macro ${story.tone}`}
                key={story.title}
              >
                <div>
                  <small>
                    {story.age} · {story.tag}
                  </small>
                  <em>{index === 0 ? "TOP STORY" : "MARKET DESK"}</em>
                </div>
                <h4>{story.title}</h4>
                <p>{story.body}</p>
                <strong>{story.impact}</strong>
              </article>
            ))}
            {analytics.attention.slice(0, 18).map((stock, index) => (
              <button key={stock.ticker} onClick={() => select(stock)}>
                <div>
                  <small>
                    SIM {String(9 + Math.floor(index / 6)).padStart(2, "0")}:
                    {String((index * 7) % 60).padStart(2, "0")} ·{" "}
                    {stock.sector.toUpperCase()}
                  </small>
                  <em>
                    {index < 3
                      ? "TRENDING"
                      : stock.sentiment_score >= 60
                        ? "BULLISH"
                        : "WATCH"}
                  </em>
                </div>
                <h4>
                  {change(stock) > 0
                    ? `${stock.ticker} catches momentum screens after a ${signed(change(stock))} simulated move`
                    : change(stock) < 0
                      ? `${stock.ticker} enters the red as traders reassess the monthly setup`
                      : `${stock.ticker} features on desk screens as valuation and growth signals diverge`}
                </h4>
                <p>
                  Sentiment {stock.sentiment_score.toFixed(0)}/100 · volume{" "}
                  {stock.volume_index.toFixed(0)} · PEG {stock.peg.toFixed(2)} ·
                  Sharpe {stock.sharpe.toFixed(2)}
                </p>
                <strong
                  className={change(stock) >= 0 ? "positive" : "negative"}
                >
                  {signed(change(stock))}
                </strong>
              </button>
            ))}
            <footer>
              ATTENTION IS NOT ALPHA · EVERY STORY AND PRICE IS SIMULATED
            </footer>
          </div>
        </aside>
      </div>
      <div className="terminal-statusbar">
        <span>SIMULATION MODE · GENERATED MARKET FEED</span>
        <span>
          50 SECURITIES · {analytics.sectors.length} SECTORS · {visible.length}{" "}
          DISPLAYED
        </span>
        <span>F1 HELP · F9 ORDERS · ENTER SECURITY</span>
      </div>
    </section>
  );
}

export function PortfolioWorkbench({
  market,
  campaign,
  trades,
  setTrades,
  review,
  backToMarket,
}: PortfolioProps) {
  const [ticket, setTicket] = useState(
    market.stocks.find((stock) => stock.close_paise <= campaign.cash_paise)
      ?.ticker || market.stocks[0].ticker,
  );
  const draft = buildDraftPortfolio(campaign, market, trades);
  const cash = draft.cashAfterPaise;
  const deployment = draft.deploymentPct;
  const positions = draft.positions.map((position) => ({
    stock: position.stock,
    shares: position.projectedShares,
    value: position.valuePaise,
  }));
  const invested = draft.investedPaise;
  const sectors = draft.sectors;
  const weighted = (
    field: keyof Pick<
      Stock,
      "pe" | "peg" | "sharpe" | "forecast_pct" | "volatility_pct"
    >,
  ) =>
    ({
      pe: draft.weighted.pe,
      peg: draft.weighted.peg,
      sharpe: draft.weighted.sharpe,
      forecast_pct: draft.weighted.forecast,
      volatility_pct: draft.weighted.volatility,
    })[field];
  const concentration = draft.concentrationPct;
  const update = (index: number, field: keyof Trade, value: string) =>
    setTrades(
      normalizeTrades(
        trades.map((trade, row) =>
          row === index
            ? {
                ...trade,
                [field]:
                  field === "shares" ? Math.max(0, Number(value)) : value,
              }
            : trade,
        ),
        campaign.holdings,
      ),
    );
  const addOrder = () =>
    setTrades(
      setDraftDelta(
        trades,
        ticket,
        draftDeltaFor(trades, ticket) + 1,
        campaign.holdings[ticket] || 0,
      ),
    );
  return (
    <section className="terminal-page workbench-page">
      <div className="terminal-commandbar">
        <div>
          <span className="live-dot" /> PORTFOLIO CONSTRUCTION{" "}
          <b>BTI-OMS/{campaign.campaign_id}</b>
        </div>
        <div className="terminal-clock">
          MOVE {campaign.current_move}/{campaign.horizon_months} · PRE-TRADE
          RISK ONLINE
        </div>
        <button onClick={backToMarket}>
          ← MARKET TERMINAL <b>ESC</b>
        </button>
      </div>
      <div className="workbench-kpis">
        <div>
          <span>BUYING POWER</span>
          <b>{rupees(draft.buyingPowerPaise)}</b>
          <small>CONTRIBUTION + SELL PROCEEDS</small>
        </div>
        <div>
          <span>ORDERS</span>
          <b>{trades.length}</b>
          <small>
            {trades.filter((trade) => trade.side === "BUY").length} BUY ·{" "}
            {trades.filter((trade) => trade.side === "SELL").length} SELL
          </small>
        </div>
        <div>
          <span>DEPLOYMENT</span>
          <b className={deployment >= 90 ? "positive" : "gold"}>
            {deployment.toFixed(1)}%
          </b>
          <small>MINIMUM 90%</small>
        </div>
        <div>
          <span>CASH AFTER</span>
          <b className={cash < 0 ? "negative" : ""}>{rupees(cash)}</b>
          <small>{cash < 0 ? "OVERDRAWN" : "ESTIMATED"}</small>
        </div>
        <div>
          <span>POSITIONS AFTER</span>
          <b>{positions.length}</b>
          <small>{sectors.length} SECTORS</small>
        </div>
        <div>
          <span>TOP WEIGHT</span>
          <b className={concentration > 25 ? "negative" : "positive"}>
            {concentration.toFixed(1)}%
          </b>
          <small>CONCENTRATION</small>
        </div>
      </div>
      <div className="workbench-grid">
        <div className="workbench-center">
          <div className="terminal-panel order-entry">
            <div className="panel-label">
              <span>ORDER ENTRY</span>
              <small>WHOLE SHARES · SIMULATED EXECUTION</small>
            </div>
            <div className="quick-ticket">
              <select
                value={ticket}
                onChange={(event) => setTicket(event.target.value)}
              >
                {market.stocks.map((stock) => (
                  <option key={stock.ticker}>{stock.ticker}</option>
                ))}
              </select>
              <div>
                <span>LAST</span>
                <b>
                  {rupees(
                    market.stocks.find((stock) => stock.ticker === ticket)!
                      .close_paise,
                  )}
                </b>
              </div>
              <div>
                <span>PEG</span>
                <b>
                  {market.stocks
                    .find((stock) => stock.ticker === ticket)!
                    .peg.toFixed(2)}
                </b>
              </div>
              <div>
                <span>SHARPE</span>
                <b>
                  {market.stocks
                    .find((stock) => stock.ticker === ticket)!
                    .sharpe.toFixed(2)}
                </b>
              </div>
              <div>
                <span>SIM 12M</span>
                <b className="positive">
                  +
                  {market.stocks
                    .find((stock) => stock.ticker === ticket)!
                    .forecast_pct.toFixed(1)}
                  %
                </b>
              </div>
              <button onClick={addOrder}>ADD TICKET +</button>
            </div>
            <div className="blotter">
              <div className="blotter-row blotter-head">
                <span>#</span>
                <span>SIDE</span>
                <span>SECURITY</span>
                <span>QTY</span>
                <span>SIM LAST</span>
                <span>NOTIONAL</span>
                <span>PORTFOLIO EFFECT</span>
                <span />
              </div>
              {trades.map((trade, index) => {
                const stock = market.stocks.find(
                  (item) => item.ticker === trade.ticker,
                )!;
                const notional = stock.close_paise * trade.shares;
                return (
                  <div className="blotter-row" key={`${trade.ticker}-${index}`}>
                    <b>{String(index + 1).padStart(2, "0")}</b>
                    <select
                      value={trade.side}
                      onChange={(event) =>
                        update(index, "side", event.target.value)
                      }
                    >
                      <option>BUY</option>
                      <option>SELL</option>
                    </select>
                    <select
                      value={trade.ticker}
                      onChange={(event) =>
                        update(index, "ticker", event.target.value)
                      }
                    >
                      {market.stocks.map((item) => (
                        <option key={item.ticker}>{item.ticker}</option>
                      ))}
                    </select>
                    <div className="terminal-stepper">
                      <button
                        onClick={() =>
                          update(index, "shares", String(trade.shares - 1))
                        }
                      >
                        −
                      </button>
                      <input
                        value={trade.shares}
                        onChange={(event) =>
                          update(index, "shares", event.target.value)
                        }
                      />
                      <button
                        onClick={() =>
                          update(index, "shares", String(trade.shares + 1))
                        }
                      >
                        +
                      </button>
                    </div>
                    <span>{rupees(stock.close_paise)}</span>
                    <strong
                      className={trade.side === "BUY" ? "negative" : "positive"}
                    >
                      {trade.side === "BUY" ? "−" : "+"}
                      {rupees(notional)}
                    </strong>
                    <span>
                      {trade.side === "BUY"
                        ? `Adds ${stock.sector}`
                        : `Cuts ${stock.sector}`}
                    </span>
                    <button
                      className="remove"
                      onClick={() =>
                        setTrades(trades.filter((_, row) => row !== index))
                      }
                    >
                      ×
                    </button>
                  </div>
                );
              })}
              {!trades.length && (
                <div className="blotter-empty">
                  <b>ORDER BLOTTER READY</b>
                  <span>
                    Select a security above or return to the market terminal.
                  </span>
                </div>
              )}
            </div>
          </div>
          <div className="workbench-lower">
            <div className="terminal-panel allocation-monitor">
              <div className="panel-label">
                <span>PROJECTED HOLDINGS</span>
                <small>POST-TRADE</small>
              </div>
              {positions.length ? (
                positions
                  .sort((a, b) => b.value - a.value)
                  .map((position) => {
                    const weight = invested
                      ? (position.value / invested) * 100
                      : 0;
                    return (
                      <div
                        className="allocation-row"
                        key={position.stock.ticker}
                      >
                        <b>{position.stock.ticker}</b>
                        <span>{position.shares} sh</span>
                        <i>
                          <b style={{ width: `${weight}%` }} />
                        </i>
                        <strong>{weight.toFixed(1)}%</strong>
                        <em>{rupees(position.value)}</em>
                      </div>
                    );
                  })
              ) : (
                <div className="blotter-empty">No projected holdings yet.</div>
              )}
            </div>
            <div className="terminal-panel exposure-monitor">
              <div className="panel-label">
                <span>SECTOR EXPOSURE</span>
                <small>POST-TRADE</small>
              </div>
              {sectors.map(([sector, value]) => (
                <div key={sector}>
                  <span>{sector}</span>
                  <i>
                    <b
                      style={{
                        width: `${invested ? (value / invested) * 100 : 0}%`,
                      }}
                    />
                  </i>
                  <strong>
                    {invested ? ((value / invested) * 100).toFixed(1) : "0.0"}%
                  </strong>
                </div>
              ))}
            </div>
          </div>
        </div>
        <aside className="workbench-side">
          <div className="terminal-panel risk-monitor">
            <div className="panel-label">
              <span>PRE-TRADE ANALYTICS</span>
              <small>ESTIMATE</small>
            </div>
            <div className="risk-grid">
              <div>
                <span>WEIGHTED PE</span>
                <b>{weighted("pe").toFixed(1)}</b>
              </div>
              <div>
                <span>WEIGHTED PEG</span>
                <b>{weighted("peg").toFixed(2)}</b>
              </div>
              <div>
                <span>PORTFOLIO SHARPE</span>
                <b className="positive">{weighted("sharpe").toFixed(2)}</b>
              </div>
              <div>
                <span>SIM FORECAST</span>
                <b className="positive">
                  +{weighted("forecast_pct").toFixed(1)}%
                </b>
              </div>
              <div>
                <span>VOLATILITY</span>
                <b>{weighted("volatility_pct").toFixed(1)}%</b>
              </div>
              <div>
                <span>TOP WEIGHT</span>
                <b className={concentration > 25 ? "negative" : ""}>
                  {concentration.toFixed(1)}%
                </b>
              </div>
            </div>
          </div>
          <div className="terminal-panel scenario-monitor">
            <div className="panel-label">
              <span>SCENARIO LENS</span>
              <small>SIMULATED</small>
            </div>
            <div>
              <span>Growth surprise</span>
              <b className="positive">
                {weighted("forecast_pct") >= 10 ? "FAVOURABLE" : "NEUTRAL"}
              </b>
            </div>
            <div>
              <span>Volatility shock</span>
              <b
                className={
                  weighted("volatility_pct") > 18 ? "negative" : "gold"
                }
              >
                {weighted("volatility_pct") > 18 ? "EXPOSED" : "MANAGEABLE"}
              </b>
            </div>
            <div>
              <span>Valuation compression</span>
              <b className={weighted("peg") > 2 ? "negative" : "positive"}>
                {weighted("peg") > 2 ? "SENSITIVE" : "RESILIENT"}
              </b>
            </div>
            <div>
              <span>Concentration</span>
              <b className={concentration > 25 ? "negative" : "positive"}>
                {concentration > 25 ? "HIGH" : "CONTROLLED"}
              </b>
            </div>
          </div>
          <div className="terminal-panel reconcile-monitor">
            <div className="panel-label">
              <span>CASH RECONCILIATION</span>
              <small>SIM OMS</small>
            </div>
            <div>
              <span>Opening cash</span>
              <b>{rupees(campaign.cash_paise)}</b>
            </div>
            <div>
              <span>Net orders</span>
              <b
                className={
                  cash - campaign.cash_paise < 0 ? "negative" : "positive"
                }
              >
                {rupees(cash - campaign.cash_paise)}
              </b>
            </div>
            <div>
              <span>Cash remaining</span>
              <b className={cash < 0 ? "negative" : ""}>{rupees(cash)}</b>
            </div>
            <div className="utilisation">
              <span>
                <b>Deployment gate</b>
                <strong>{deployment.toFixed(1)}%</strong>
              </span>
              <i>
                <b style={{ width: `${Math.min(100, deployment)}%` }} />
              </i>
              <small>
                {deployment >= 90
                  ? "READY FOR REVIEW"
                  : `DEPLOY ${rupees(Math.max(0, cash - campaign.cash_paise * 0.1))} MORE`}
              </small>
            </div>
            <button
              className="primary full"
              disabled={!trades.length || cash < 0 || deployment < 90}
              onClick={review}
            >
              REVIEW EXECUTION <b>F10</b> →
            </button>
          </div>
        </aside>
      </div>
      <div className="terminal-statusbar">
        <span>SIMULATION MODE · NO REAL ORDERS</span>
        <span>
          PRE-TRADE CHECKS: {cash >= 0 ? "CASH ✓" : "CASH ✕"} ·{" "}
          {deployment >= 90 ? "DEPLOYMENT ✓" : "DEPLOYMENT ✕"} · WHOLE SHARES ✓
        </span>
        <span>DECISION ENGINE RUNS AFTER COMMIT</span>
      </div>
    </section>
  );
}

export function NewsTerminal({
  market,
  campaign,
  backToMarket,
}: BaseProps & { backToMarket: () => void }) {
  const movers = [...market.stocks]
    .sort((a, b) => Math.abs(change(b)) - Math.abs(change(a)))
    .slice(0, 6);
  const gainers = [...market.stocks]
    .filter((stock) => change(stock) > 0)
    .sort((a, b) => change(b) - change(a))
    .slice(0, 8);
  const losers = [...market.stocks]
    .filter((stock) => change(stock) < 0)
    .sort((a, b) => change(a) - change(b))
    .slice(0, 8);
  const lead = movers[0];
  const riskLead = [...market.stocks].sort(
    (a, b) =>
      b.volume_index + b.volatility_pct - (a.volume_index + a.volatility_pct),
  )[0];
  const sentiment =
    market.stocks.reduce((sum, stock) => sum + stock.sentiment_score, 0) /
    market.stocks.length;
  const contextualStories = [
    {
      tag: "PRICE DISCOVERY",
      title: `${lead.ticker} becomes the month’s strongest attention signal`,
      body: `${signed(change(lead))} price movement, ${lead.volume_index.toFixed(0)} volume index and ${lead.sentiment_score.toFixed(0)}/100 sentiment make this material—but not automatically investable.`,
      impact: `${lead.ticker} · ${lead.sector}`,
      tone: change(lead) >= 0 ? "green" : "red",
      age: "SIM NOW",
    },
    {
      tag: "RISK DESK",
      title: `${riskLead.ticker} volatility and activity demand position-size discipline`,
      body: `The security carries ${riskLead.volatility_pct.toFixed(1)}% annualised volatility, ${riskLead.var_95_pct.toFixed(1)}% VaR and a ${riskLead.volume_index.toFixed(0)} activity index in this generated information set.`,
      impact: `${riskLead.ticker} · ${riskLead.sector}`,
      tone: "amber",
      age: "SIM 08:31",
    },
    ...stories.slice(1, 3),
  ];
  return (
    <section className="terminal-page news-terminal">
      <div className="terminal-commandbar">
        <div>
          <span className="live-dot" /> BTI NEWSWIRE <b>SIMULATED EDITION</b>
        </div>
        <div className="terminal-clock">
          CURATED FOR CAMPAIGN {campaign.campaign_id} · MOVE{" "}
          {campaign.current_move}
        </div>
        <button onClick={backToMarket}>← MARKET TERMINAL</button>
      </div>
      <div className="news-tape">
        {movers.map((stock) => (
          <div key={stock.ticker}>
            <b>{stock.ticker}</b>
            <span>{rupees(stock.close_paise)}</span>
            <strong className={change(stock) >= 0 ? "positive" : "negative"}>
              {signed(change(stock))}
            </strong>
            <MiniChart
              values={stock.history_paise}
              tone={change(stock) >= 0 ? "green" : "red"}
            />
          </div>
        ))}
      </div>
      <div className="news-nav">
        <button className="active">TOP STORIES</button>
        <button>MARKETS</button>
        <button>ECONOMY</button>
        <button>EARNINGS</button>
        <button>SECTORS</button>
        <span>ALL CONTENT IS GENERATED FOR THIS SIMULATION</span>
      </div>
      <div className="news-movers-board">
        <div className="terminal-panel">
          <div className="panel-label">
            <span>TOP GAINERS · THIS SIMULATED MONTH</span>
            <small>RANKED BY OPEN-TO-LAST</small>
          </div>
          {gainers.length === 0 && (
            <p className="movers-empty">
              No positive moves yet in this campaign month.
            </p>
          )}
          {gainers.map((stock, index) => (
            <div className="news-mover-row" key={stock.ticker}>
              <b>{String(index + 1).padStart(2, "0")}</b>
              <span>
                {stock.ticker}
                <small>{stock.sector}</small>
              </span>
              <MiniChart values={stock.history_paise} tone="green" />
              <em>{rupees(stock.close_paise)}</em>
              <strong className="positive">{signed(change(stock))}</strong>
            </div>
          ))}
        </div>
        <div className="terminal-panel">
          <div className="panel-label">
            <span>TOP LOSERS · THIS SIMULATED MONTH</span>
            <small>RANKED BY OPEN-TO-LAST</small>
          </div>
          {losers.length === 0 && (
            <p className="movers-empty">
              No negative moves yet in this campaign month.
            </p>
          )}
          {losers.map((stock, index) => (
            <div className="news-mover-row" key={stock.ticker}>
              <b>{String(index + 1).padStart(2, "0")}</b>
              <span>
                {stock.ticker}
                <small>{stock.sector}</small>
              </span>
              <MiniChart values={stock.history_paise} tone="red" />
              <em>{rupees(stock.close_paise)}</em>
              <strong className={change(stock) < 0 ? "negative" : ""}>
                {signed(change(stock))}
              </strong>
            </div>
          ))}
        </div>
      </div>
      <div className="newsroom-grid">
        <main className="newsroom-main">
          <article className="lead-story">
            <div className="story-art policy-art">
              <span>RBI</span>
              <b>POLICY</b>
              <i>SIMULATION</i>
            </div>
            <div>
              <small>MACRO · SIM 08:42 · 6 MIN READ</small>
              <h1>
                Market sentiment sits at {sentiment.toFixed(0)}/100. The real
                contest remains evidence quality.
              </h1>
              <p>
                Price, volume, fundamentals and risk are moving together inside
                this calibrated synthetic market. The strongest headline is a
                signal to investigate—not a direction to trade.
              </p>
              <div className="story-tags">
                <span>HDFCBANK</span>
                <span>ICICIBANK</span>
                <span>SBIN</span>
                <span>BAJFINANCE</span>
              </div>
            </div>
          </article>
          <div className="story-columns">
            {contextualStories.map((story) => (
              <article key={story.title}>
                <div className={`story-art mini ${story.tone}`}>
                  <span>{story.tag}</span>
                  <b>{story.impact.split(" · ")[0]}</b>
                  <i>SIM</i>
                </div>
                <small>
                  {story.age} · {story.tag}
                </small>
                <h2>{story.title}</h2>
                <p>{story.body}</p>
                <div className="story-tags">
                  <span>{story.impact}</span>
                </div>
              </article>
            ))}
          </div>
        </main>
        <aside className="newsroom-side">
          <div className="terminal-panel">
            <div className="panel-label">
              <span>MARKET PULSE</span>
              <small>SIM NOW</small>
            </div>
            <div className="pulse-score">
              <b>{sentiment.toFixed(0)}</b>
              <span>
                SELECTIVE
                <br />
                RISK-ON
              </span>
            </div>
            <div className="breadth-bar">
              <i
                style={{
                  width: `${(market.stocks.filter((stock) => change(stock) >= 0).length / market.stocks.length) * 100}%`,
                }}
              />
              <span>ADVANCES</span>
            </div>
          </div>
          <div className="terminal-panel">
            <div className="panel-label">
              <span>TRENDING SECURITIES</span>
              <small>SIM ACTIVITY</small>
            </div>
            {movers.map((stock) => (
              <div className="trending-row" key={stock.ticker}>
                <b>{stock.ticker}</b>
                <MiniChart
                  values={stock.history_paise}
                  tone={change(stock) >= 0 ? "green" : "red"}
                />
                <span>{rupees(stock.close_paise)}</span>
                <strong
                  className={change(stock) >= 0 ? "positive" : "negative"}
                >
                  {signed(change(stock))}
                </strong>
              </div>
            ))}
          </div>
          <div className="terminal-panel editor-note">
            <div className="panel-label">
              <span>HOW TO USE THIS FEED</span>
            </div>
            <p>
              Events change the context, not the answer. Test each headline
              against valuation, quality, downside and portfolio fit before
              changing the move.
            </p>
          </div>
        </aside>
      </div>
      <div className="terminal-statusbar">
        <span>SIMULATION MODE · GENERATED NEWSWIRE</span>
        <span>NO LIVE NEWS · NO REAL-TIME PRICES · NO INVESTMENT ADVICE</span>
        <span>
          EVENT SET {campaign.current_move.toString().padStart(2, "0")}
        </span>
      </div>
    </section>
  );
}
