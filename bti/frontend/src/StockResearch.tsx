import type { Campaign, Market, Stock } from "./api";

type Props = {
  stock: Stock;
  market: Market;
  campaign: Campaign;
  add: (stock: Stock) => void;
  back: () => void;
  openWorkbench: () => void;
};

const money = (paise: number) =>
  new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(paise / 100);
const signed = (value: number) =>
  `${value >= 0 ? "+" : ""}${value.toFixed(2)}%`;
const move = (stock: Stock) => (stock.close_paise / stock.open_paise - 1) * 100;

function Candles({ stock }: { stock: Stock }) {
  const candles = stock.ohlc_history.length
    ? stock.ohlc_history
    : [
        {
          month: 0,
          open_paise: stock.open_paise,
          high_paise: stock.high_paise,
          low_paise: stock.low_paise,
          close_paise: stock.close_paise,
        },
      ];
  const low = Math.min(...candles.map((item) => item.low_paise));
  const high = Math.max(...candles.map((item) => item.high_paise));
  const spread = high - low || Math.max(1, high * 0.02);
  const y = (value: number) => 245 - ((value - low) / spread) * 205;
  const step = 720 / candles.length;
  const width = Math.max(4, Math.min(18, step * 0.52));
  const ticks = Array.from(
    { length: 5 },
    (_, index) => high - (spread * index) / 4,
  );
  return (
    <svg
      className="research-chart"
      viewBox="0 0 800 280"
      preserveAspectRatio="none"
      role="img"
      aria-label={`${stock.ticker} simulated OHLC chart`}
    >
      {ticks.map((tick) => (
        <g key={tick}>
          <line
            x1="0"
            x2="720"
            y1={y(tick)}
            y2={y(tick)}
            className="chart-gridline"
          />
          <text x="732" y={y(tick) + 4} className="chart-label">
            {money(tick)}
          </text>
        </g>
      ))}
      {candles.map((item, index) => {
        const x = step * index + step / 2;
        const rising = item.close_paise >= item.open_paise;
        const bodyTop = Math.min(y(item.open_paise), y(item.close_paise));
        const bodyHeight = Math.max(
          2,
          Math.abs(y(item.open_paise) - y(item.close_paise)),
        );
        return (
          <g
            key={`${item.month}-${index}`}
            className={rising ? "candle-up" : "candle-down"}
          >
            <line
              x1={x}
              x2={x}
              y1={y(item.high_paise)}
              y2={y(item.low_paise)}
            />
            <rect
              x={x - width / 2}
              y={bodyTop}
              width={width}
              height={bodyHeight}
            />
          </g>
        );
      })}
      <text x="0" y="274" className="chart-label">
        GENERATED LOOKBACK
      </text>
      <text x="655" y="274" className="chart-label">
        CURRENT MOVE
      </text>
    </svg>
  );
}

function ForecastChart({ stock }: { stock: Stock }) {
  const curve = stock.forecast_curve || [];
  const values = curve.map((point) => point.cumulative_pct);
  const low = Math.min(0, ...values);
  const high = Math.max(1, ...values);
  const spread = high - low || 1;
  const x = (index: number) =>
    55 + (index / Math.max(1, curve.length - 1)) * 660;
  const y = (value: number) => 235 - ((value - low) / spread) * 185;
  const path = curve
    .map(
      (point, index) =>
        `${index ? "L" : "M"}${x(index)},${y(point.cumulative_pct)}`,
    )
    .join(" ");
  return (
    <svg
      className="research-chart forecast"
      viewBox="0 0 800 280"
      preserveAspectRatio="none"
      role="img"
      aria-label={`${stock.ticker} simulated forecast curve`}
    >
      <line
        x1="45"
        x2="730"
        y1={y(0)}
        y2={y(0)}
        className="chart-gridline zero"
      />
      <path
        d={`${path} L${x(curve.length - 1)},245 L${x(0)},245 Z`}
        className="forecast-area"
      />
      <path d={path} className="forecast-line" />
      {curve.map((point, index) => (
        <g key={point.months}>
          <circle cx={x(index)} cy={y(point.cumulative_pct)} r="5" />
          <text
            x={x(index)}
            y="267"
            textAnchor="middle"
            className="chart-label"
          >
            {point.months}M
          </text>
          <text
            x={x(index)}
            y={y(point.cumulative_pct) - 12}
            textAnchor="middle"
            className="forecast-label"
          >
            {signed(point.cumulative_pct)}
          </text>
        </g>
      ))}
    </svg>
  );
}

const sectorContext: Record<string, string> = {
  Energy:
    "Crude sensitivity and cash-flow durability remain the main simulated catalysts.",
  Financials:
    "Funding costs, asset quality and credit growth drive the current simulated debate.",
  IT: "Global technology spending and currency movement shape the generated outlook.",
  Healthcare:
    "Product mix, regulatory execution and defensive demand support the current scenario.",
  Automobile:
    "Volume, input costs and premiumisation determine operating leverage in this scenario.",
  Consumer:
    "Demand resilience must justify the valuation premium embedded in the simulated quote.",
  Infrastructure:
    "Order visibility is supportive, while execution and valuation remain the key risks.",
  Materials:
    "The generated cycle is sensitive to commodity prices and utilisation.",
  Utilities:
    "Stable demand helps resilience; capital intensity limits upside without execution.",
  Telecom:
    "Subscriber economics and capital discipline define the simulated investment case.",
};

export function StockResearch({
  stock,
  market,
  campaign,
  add,
  back,
  openWorkbench,
}: Props) {
  const change = move(stock);
  const holding = campaign.holdings[stock.ticker] || 0;
  const peers = market.stocks
    .filter(
      (item) => item.sector === stock.sector && item.ticker !== stock.ticker,
    )
    .sort((a, b) => b.forecast_pct - a.forecast_pct)
    .slice(0, 4);
  const signal = Math.max(
    0,
    Math.min(
      100,
      Math.round(58 + stock.sharpe * 12 - Math.min(10, stock.peg) * 2),
    ),
  );
  const valueRead =
    stock.peg > 0 && stock.peg <= 1
      ? "ATTRACTIVE"
      : stock.peg <= 2
        ? "BALANCED"
        : "DEMANDING";
  const riskRead =
    stock.volatility_pct <= 16
      ? "CONTROLLED"
      : stock.volatility_pct <= 22
        ? "MODERATE"
        : "ELEVATED";
  return (
    <section className="terminal-page stock-research">
      <div className="terminal-commandbar">
        <button onClick={back}>← MARKET MONITOR</button>
        <div>
          <span className="live-dot" /> SECURITY RESEARCH{" "}
          <b>BTI-QR/{stock.ticker}</b>
        </div>
        <div className="terminal-clock">
          SIMULATED QUOTE · MOVE {campaign.current_move}
        </div>
      </div>
      <div className="research-hero">
        <div className="research-identity">
          <i>{stock.ticker.slice(0, 2)}</i>
          <div>
            <span>{stock.sector} · NIFTY 50 SIMULATION</span>
            <h1>{stock.ticker}</h1>
            <p>
              {money(stock.close_paise)}{" "}
              <strong className={change >= 0 ? "positive" : "negative"}>
                {signed(change)}
              </strong>
            </p>
          </div>
        </div>
        <div className="research-hero-stats">
          <div>
            <span>VRIDDHI UNIVERSE RANK</span>
            <b>#{stock.overall_rank}</b>
          </div>
          <div>
            <span>HELD</span>
            <b>{holding} SHARES</b>
          </div>
          <div>
            <span>PUBLIC SIGNAL</span>
            <b className="positive">{signal}/100</b>
          </div>
        </div>
        <button
          className="primary"
          onClick={() => {
            add(stock);
            openWorkbench();
          }}
        >
          STAGE BUY ORDER +
        </button>
      </div>
      <div className="research-tape">
        <div>
          <span>OPEN</span>
          <b>{money(stock.open_paise)}</b>
        </div>
        <div>
          <span>HIGH</span>
          <b>{money(stock.high_paise)}</b>
        </div>
        <div>
          <span>LOW</span>
          <b>{money(stock.low_paise)}</b>
        </div>
        <div>
          <span>LAST</span>
          <b>{money(stock.close_paise)}</b>
        </div>
        <div>
          <span>HISTORICAL CAGR</span>
          <b>{signed(stock.historical_cagr_pct)}</b>
        </div>
        <div>
          <span>SIM 12M FORECAST</span>
          <b className="positive">{signed(stock.forecast_pct)}</b>
        </div>
      </div>
      <div className="research-grid">
        <main className="research-main">
          <div className="terminal-panel research-chart-panel">
            <div className="panel-label">
              <span>SIMULATED OHLC · GENERATED LOOKBACK</span>
              <small>{stock.ohlc_history.length} PERIODS</small>
            </div>
            <Candles stock={stock} />
          </div>
          <div className="terminal-panel research-chart-panel">
            <div className="panel-label">
              <span>FORECAST TERM STRUCTURE</span>
              <small>ANNUALISED INPUT → CUMULATIVE PATH</small>
            </div>
            <ForecastChart stock={stock} />
          </div>
          <div className="terminal-panel ratio-lab">
            <div className="panel-label">
              <span>VALUATION · QUALITY · RISK MATRIX</span>
              <small>PUBLIC INPUTS</small>
            </div>
            <div className="ratio-grid">
              <div>
                <span>PE</span>
                <b>{stock.pe.toFixed(1)}</b>
                <small>profit multiple</small>
              </div>
              <div>
                <span>PB</span>
                <b>{stock.pb.toFixed(1)}</b>
                <small>book premium</small>
              </div>
              <div>
                <span>PEG</span>
                <b className={stock.peg <= 1 ? "positive" : ""}>
                  {stock.peg.toFixed(2)}
                </b>
                <small>{valueRead}</small>
              </div>
              <div>
                <span>SHARPE</span>
                <b className="positive">{stock.sharpe.toFixed(2)}</b>
                <small>risk-adjusted</small>
              </div>
              <div>
                <span>VOLATILITY</span>
                <b>{stock.volatility_pct.toFixed(1)}%</b>
                <small>{riskRead}</small>
              </div>
              <div>
                <span>MAX DD</span>
                <b className="negative">{stock.drawdown_pct.toFixed(1)}%</b>
                <small>peak-to-trough</small>
              </div>
              <div>
                <span>VaR 95</span>
                <b className="negative">{stock.var_95_pct.toFixed(1)}%</b>
                <small>tail threshold</small>
              </div>
              <div>
                <span>EXPECTED SHORTFALL</span>
                <b className="negative">
                  {stock.expected_shortfall_95_pct.toFixed(1)}%
                </b>
                <small>severe-case mean</small>
              </div>
            </div>
          </div>
          <div className="terminal-panel security-news">
            <div className="panel-label">
              <span>{stock.ticker} NEWS & CATALYST MONITOR</span>
              <small>GENERATED FOR SIMULATION</small>
            </div>
            <div className="security-news-grid">
              <article>
                <small>SIM · COMPANY</small>
                <h3>
                  {stock.ticker} enters the month with {valueRead.toLowerCase()}{" "}
                  growth pricing
                </h3>
                <p>
                  At PEG {stock.peg.toFixed(2)}, the market asks the company to
                  convert its {signed(stock.forecast_pct)} annualised outlook
                  into durable delivery.
                </p>
              </article>
              <article>
                <small>SIM · {stock.sector.toUpperCase()}</small>
                <h3>{stock.sector} conditions shape the next test</h3>
                <p>
                  {sectorContext[stock.sector] ||
                    "Sector conditions and company execution jointly determine the generated path."}
                </p>
              </article>
              <article>
                <small>SIM · RISK DESK</small>
                <h3>
                  {riskRead.toLowerCase()} volatility keeps downside discipline
                  relevant
                </h3>
                <p>
                  VaR is {stock.var_95_pct.toFixed(1)}% and expected shortfall
                  is {stock.expected_shortfall_95_pct.toFixed(1)}% in the
                  current information set.
                </p>
              </article>
            </div>
          </div>
        </main>
        <aside className="research-side">
          <div className="terminal-panel research-thesis">
            <div className="panel-label">
              <span>RESEARCH SNAPSHOT</span>
              <small>NOT A HIDDEN VERDICT</small>
            </div>
            <div className="signal-dial">
              <b>{signal}</b>
              <span>
                PUBLIC
                <br />
                EVIDENCE
              </span>
            </div>
            <p>
              {valueRead} valuation · {riskRead} risk ·{" "}
              {signed(stock.forecast_pct)} simulated annualised forecast.
            </p>
          </div>
          <div className="terminal-panel peer-table">
            <div className="panel-label">
              <span>SECTOR PEERS</span>
              <small>SIM 12M</small>
            </div>
            <div className="peer-row peer-head">
              <span>TICKER</span>
              <span>PEG</span>
              <span>SHARPE</span>
              <span>FCST</span>
            </div>
            {peers.map((peer) => (
              <div className="peer-row" key={peer.ticker}>
                <b>{peer.ticker}</b>
                <span>{peer.peg.toFixed(2)}</span>
                <span>{peer.sharpe.toFixed(2)}</span>
                <strong className="positive">
                  {signed(peer.forecast_pct)}
                </strong>
              </div>
            ))}
          </div>
          <div className="terminal-panel research-guide">
            <div className="panel-label">
              <span>PORTFOLIO QUESTIONS</span>
            </div>
            <ol>
              <li>Is the growth worth the PEG you pay?</li>
              <li>Does this improve sector balance?</li>
              <li>Is the downside acceptable beside current holdings?</li>
              <li>What assumption would make this thesis fail?</li>
            </ol>
          </div>
        </aside>
      </div>
      <div className="terminal-statusbar">
        <span>SIMULATION MODE · GENERATED SECURITY RESEARCH</span>
        <span>REAL TICKER · NO LIVE QUOTE · NO INVESTMENT ADVICE</span>
        <span>DATASET RANK #{stock.overall_rank}</span>
      </div>
    </section>
  );
}
