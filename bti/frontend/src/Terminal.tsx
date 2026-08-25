import { useMemo, useState } from "react";
import type { Campaign, Market, Stock, Trade } from "./api";

type BaseProps = { market: Market; campaign: Campaign };
type MarketProps = BaseProps & {
  select: (stock: Stock) => void;
  buildMove: () => void;
  openNews: () => void;
};
type PortfolioProps = BaseProps & {
  trades: Trade[];
  setTrades: (trades: Trade[]) => void;
  review: () => void;
  backToMarket: () => void;
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
  openNews,
}: MarketProps) {
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState("ALL");
  const [sort, setSort] = useState<"ticker" | "forecast" | "sharpe" | "peg">(
    "ticker",
  );
  const analytics = useMemo(() => {
    const advances = market.stocks.filter((stock) => change(stock) >= 0).length;
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
    return {
      advances,
      declines: market.stocks.length - advances,
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
    };
  }, [market]);
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
    .sort((a, b) => {
      if (sort === "forecast") return b.forecast_pct - a.forecast_pct;
      if (sort === "sharpe") return b.sharpe - a.sharpe;
      if (sort === "peg") return (a.peg || 999) - (b.peg || 999);
      return a.ticker.localeCompare(b.ticker);
    });
  const tape = [
    ["NIFTY 50", "24,812.45", "+0.73", [98, 101, 99, 103, 104, 108, 107, 112]],
    [
      "BANK NIFTY",
      "55,240.10",
      "+0.42",
      [101, 99, 102, 106, 105, 108, 110, 111],
    ],
    ["INDIA VIX", "13.82", "-2.18", [110, 108, 111, 105, 103, 101, 99, 97]],
    ["USD / INR", "83.74", "+0.06", [99, 100, 100, 101, 100, 102, 102, 103]],
    ["BRENT", "$81.20", "-0.91", [112, 109, 110, 105, 107, 102, 101, 99]],
    ["GOLD", "₹72,430", "+0.31", [100, 101, 103, 102, 105, 104, 106, 108]],
  ] as const;
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
        <button onClick={buildMove}>
          OPEN ORDER WORKBENCH <b>F9</b>
        </button>
      </div>
      <div className="market-tape">
        {tape.map(([name, value, move, history]) => (
          <div className="tape-quote" key={name}>
            <span>
              {name} <small>SIM</small>
            </span>
            <b>{value}</b>
            <strong className={Number(move) >= 0 ? "positive" : "negative"}>
              {Number(move) >= 0 ? "+" : ""}
              {move}%
            </strong>
            <MiniChart
              values={[...history]}
              tone={Number(move) >= 0 ? "green" : "red"}
            />
          </div>
        ))}
      </div>
      <div className="terminal-kpis">
        <div>
          <span>MARKET BREADTH</span>
          <b>
            {analytics.advances} : {analytics.declines}
          </b>
          <small>ADV / DEC</small>
        </div>
        <div>
          <span>MEDIAN PEG</span>
          <b>{analytics.medianPeg.toFixed(2)}</b>
          <small>PRICE / GROWTH</small>
        </div>
        <div>
          <span>MEAN FORECAST</span>
          <b className="positive">+{analytics.meanForecast.toFixed(1)}%</b>
          <small>SIM 12M</small>
        </div>
        <div>
          <span>CROSS-SECTION VOL</span>
          <b>{analytics.meanVolatility.toFixed(1)}%</b>
          <small>ANNUALISED</small>
        </div>
        <div>
          <span>PORTFOLIO CASH</span>
          <b>{rupees(campaign.cash_paise)}</b>
          <small>AVAILABLE</small>
        </div>
        <div>
          <span>ALPHA TO DATE</span>
          <b className={campaign.alpha_pct >= 0 ? "positive" : "negative"}>
            {signed(campaign.alpha_pct)}
          </b>
          <small>VS NIFTY 50</small>
        </div>
      </div>
      <div className="terminal-grid">
        <div className="terminal-main">
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
            <select
              value={sort}
              onChange={(event) => setSort(event.target.value as typeof sort)}
              aria-label="Sort market"
            >
              <option value="ticker">SORT: TICKER</option>
              <option value="forecast">SORT: FORECAST</option>
              <option value="sharpe">SORT: SHARPE</option>
              <option value="peg">SORT: PEG</option>
            </select>
          </div>
          <div className="pro-table">
            <div className="pro-row pro-head">
              <span>SECURITY</span>
              <span>LAST</span>
              <span>SIM Δ</span>
              <span>INTRAMONTH</span>
              <span>PE</span>
              <span>PB</span>
              <span>PEG</span>
              <span>SHARPE</span>
              <span>VOL</span>
              <span>DD</span>
              <span>VaR 95</span>
              <span>SIM 12M</span>
            </div>
            {visible.map((stock) => {
              const move = change(stock);
              return (
                <button
                  className="pro-row"
                  key={stock.ticker}
                  onClick={() => select(stock)}
                >
                  <span className="pro-security">
                    <i>{stock.ticker.slice(0, 2)}</i>
                    <span>
                      <b>{stock.ticker}</b>
                      <small>
                        {stock.sector}
                        {campaign.holdings[stock.ticker]
                          ? ` · HELD ${campaign.holdings[stock.ticker]}`
                          : ""}
                      </small>
                    </span>
                  </span>
                  <strong>{rupees(stock.close_paise)}</strong>
                  <strong className={move >= 0 ? "positive" : "negative"}>
                    {signed(move)}
                  </strong>
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
                </button>
              );
            })}
          </div>
        </div>
        <aside className="terminal-side">
          <div className="terminal-panel sector-monitor">
            <div className="panel-label">
              <span>SECTOR MONITOR</span>
              <small>SIM FLOW</small>
            </div>
            {analytics.sectors.map((sector) => (
              <button key={sector.sector}>
                <span>{sector.sector}</span>
                <i>
                  <b
                    style={{
                      width: `${Math.min(100, Math.max(8, sector.forecast * 4.5))}%`,
                    }}
                  />
                </i>
                <strong className={sector.move >= 0 ? "positive" : "negative"}>
                  {signed(sector.move)}
                </strong>
              </button>
            ))}
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
          <div className="terminal-panel event-monitor">
            <div className="panel-label">
              <span>NEWSWIRE</span>
              <button onClick={openNews}>ALL →</button>
            </div>
            {stories.slice(0, 3).map((story) => (
              <button key={story.title} onClick={openNews}>
                <small>
                  {story.age} · {story.tag}
                </small>
                <b>{story.title}</b>
                <span>{story.impact}</span>
              </button>
            ))}
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
  const cash =
    campaign.cash_paise +
    trades.reduce((sum, trade) => {
      const stock = market.stocks.find((item) => item.ticker === trade.ticker)!;
      return (
        sum +
        stock.close_paise * trade.shares * (trade.side === "SELL" ? 1 : -1)
      );
    }, 0);
  const deployment = Math.max(0, (1 - cash / campaign.cash_paise) * 100);
  const afterHoldings = { ...campaign.holdings };
  trades.forEach((trade) => {
    afterHoldings[trade.ticker] = Math.max(
      0,
      (afterHoldings[trade.ticker] || 0) +
        trade.shares * (trade.side === "BUY" ? 1 : -1),
    );
  });
  const positions = Object.entries(afterHoldings)
    .map(([ticker, shares]) => {
      const stock = market.stocks.find((item) => item.ticker === ticker)!;
      return { stock, shares, value: shares * stock.close_paise };
    })
    .filter((position) => position.shares > 0);
  const invested = positions.reduce((sum, position) => sum + position.value, 0);
  const sectors = Object.entries(
    positions.reduce<Record<string, number>>((all, position) => {
      all[position.stock.sector] =
        (all[position.stock.sector] || 0) + position.value;
      return all;
    }, {}),
  ).sort((a, b) => b[1] - a[1]);
  const weighted = (
    field: keyof Pick<
      Stock,
      "pe" | "peg" | "sharpe" | "forecast_pct" | "volatility_pct"
    >,
  ) =>
    invested
      ? positions.reduce(
          (sum, position) =>
            sum + Number(position.stock[field]) * position.value,
          0,
        ) / invested
      : 0;
  const concentration = invested
    ? Math.max(
        0,
        ...positions.map((position) => (position.value / invested) * 100),
      )
    : 0;
  const update = (index: number, field: keyof Trade, value: string) =>
    setTrades(
      trades.map((trade, row) =>
        row === index
          ? {
              ...trade,
              [field]: field === "shares" ? Math.max(0, Number(value)) : value,
            }
          : trade,
      ),
    );
  const addOrder = () =>
    setTrades([...trades, { side: "BUY", ticker: ticket, shares: 1 }]);
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
          <b>{rupees(campaign.cash_paise)}</b>
          <small>CONTRIBUTION + CASH</small>
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
                Rates stay unchanged. The real contest shifts to earnings
                quality.
              </h1>
              <p>
                The simulated policy committee keeps borrowing conditions
                stable. Financials retain support, but valuation discipline
                decides which balance sheets deserve capital.
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
            {stories.slice(1, 5).map((story) => (
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
              <b>64</b>
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
