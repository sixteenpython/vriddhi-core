import { useMemo, useState } from "react";
import type { Campaign, Market, Stock, Trade } from "./api";
import type { DraftPortfolio } from "./portfolioDraft";

type TileId = "pulse" | "internals" | "sector" | "quant" | "movers" | "news";
type TileConfig = { id: TileId; hidden: boolean; wide: boolean };
type Candle = {
  open: number;
  high: number;
  low: number;
  close: number;
};

const TILE_LABELS: Record<TileId, string> = {
  pulse: "MARKET PULSE",
  internals: "MARKET INTERNALS",
  sector: "SECTOR MONITOR",
  quant: "QUANT RISK",
  movers: "MONTHLY MOVERS",
  news: "TOP 5 GAINERS / LOSERS",
};
const DEFAULT_TILES: TileConfig[] = (Object.keys(TILE_LABELS) as TileId[]).map(
  (id) => ({ id, hidden: false, wide: id === "pulse" || id === "movers" }),
);
const LAYOUT_KEY = "bti_cockpit_layout_v3";

const rupees = (paise: number) =>
  new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(paise / 100);
const signed = (value: number, digits = 1) =>
  `${value >= 0 ? "+" : ""}${value.toFixed(digits)}%`;
const move = (stock: Stock) => (stock.close_paise / stock.open_paise - 1) * 100;

function loadLayout() {
  try {
    const saved = localStorage.getItem(LAYOUT_KEY);
    if (!saved) return DEFAULT_TILES;
    const parsed = JSON.parse(saved) as TileConfig[];
    return parsed.length === DEFAULT_TILES.length ? parsed : DEFAULT_TILES;
  } catch {
    return DEFAULT_TILES;
  }
}

function aggregateMarketCandles(market: Market): Candle[] {
  const histories = market.stocks.map((stock) =>
    stock.ohlc_history.slice(-252),
  );
  const length = Math.min(...histories.map((history) => history.length));
  return Array.from({ length }, (_, index) => {
    const normalized = histories.map((history) => {
      const base = history[history.length - length].close_paise || 1;
      const candle = history[history.length - length + index];
      return {
        open: (candle.open_paise / base) * 100,
        high: (candle.high_paise / base) * 100,
        low: (candle.low_paise / base) * 100,
        close: (candle.close_paise / base) * 100,
      };
    });
    const average = (field: keyof Candle) =>
      normalized.reduce((sum, candle) => sum + candle[field], 0) /
      normalized.length;
    return {
      open: average("open"),
      high: average("high"),
      low: average("low"),
      close: average("close"),
    };
  });
}

function compressCandles(candles: Candle[], step: number) {
  const result: Candle[] = [];
  for (let index = 0; index < candles.length; index += step) {
    const group = candles.slice(index, index + step);
    if (!group.length) continue;
    result.push({
      open: group[0].open,
      high: Math.max(...group.map((item) => item.high)),
      low: Math.min(...group.map((item) => item.low)),
      close: group[group.length - 1].close,
    });
  }
  return result;
}

function SignalChart({ candles }: { candles: Candle[] }) {
  const [hovered, setHovered] = useState<number | null>(null);
  const low = Math.min(...candles.map((item) => item.low));
  const high = Math.max(...candles.map((item) => item.high));
  const spread = high - low || 1;
  const y = (value: number) => 250 - ((value - low) / spread) * 215;
  const width = 920 / Math.max(candles.length, 1);
  const volume = candles.map(
    (item) => Math.abs(item.close - item.open) + (item.high - item.low) * 0.45,
  );
  const maxVolume = Math.max(...volume, 1);
  const line = candles
    .map(
      (item, index) =>
        `${index ? "L" : "M"}${index * width + width / 2},${y(item.close)}`,
    )
    .join(" ");
  return (
    <svg
      className="signal-chart"
      viewBox="0 0 920 330"
      role="img"
      aria-label="Simulated market OHLC chart"
      onMouseLeave={() => setHovered(null)}
      onMouseMove={(event) => {
        const bounds = event.currentTarget.getBoundingClientRect();
        const ratio = (event.clientX - bounds.left) / bounds.width;
        setHovered(
          Math.max(
            0,
            Math.min(candles.length - 1, Math.floor(ratio * candles.length)),
          ),
        );
      }}
    >
      {[0, 1, 2, 3, 4].map((row) => (
        <line key={row} x1="0" y1={40 + row * 65} x2="920" y2={40 + row * 65} />
      ))}
      {candles.map((item, index) => {
        const x = index * width + width / 2;
        const positive = item.close >= item.open;
        return (
          <g key={index} className={positive ? "up" : "down"}>
            <line x1={x} y1={y(item.high)} x2={x} y2={y(item.low)} />
            <rect
              x={x - Math.max(1.2, width * 0.28)}
              y={Math.min(y(item.open), y(item.close))}
              width={Math.max(2.4, width * 0.56)}
              height={Math.max(2, Math.abs(y(item.open) - y(item.close)))}
            />
          </g>
        );
      })}
      {volume.map((value, index) => (
        <rect
          key={`volume-${index}`}
          className={
            candles[index].close >= candles[index].open
              ? "volume-up"
              : "volume-down"
          }
          x={index * width + width * 0.12}
          y={320 - (value / maxVolume) * 48}
          width={Math.max(1, width * 0.76)}
          height={(value / maxVolume) * 48}
        />
      ))}
      <path className="close-line" d={line} />
      {hovered !== null && candles[hovered] && (
        <g className="chart-crosshair">
          <line
            x1={hovered * width + width / 2}
            y1="25"
            x2={hovered * width + width / 2}
            y2="320"
          />
          <rect x="12" y="8" width="300" height="23" />
          <text x="22" y="24">
            BAR {hovered + 1} · O {candles[hovered].open.toFixed(2)} · H{" "}
            {candles[hovered].high.toFixed(2)} · L{" "}
            {candles[hovered].low.toFixed(2)} · C{" "}
            {candles[hovered].close.toFixed(2)}
          </text>
        </g>
      )}
      <text x="8" y="20">
        {high.toFixed(1)}
      </text>
      <text x="8" y="320">
        {low.toFixed(1)}
      </text>
      <text x="835" y="320">
        SIM VOLUME
      </text>
    </svg>
  );
}

function SignalModal({ market, close }: { market: Market; close: () => void }) {
  const [range, setRange] = useState(252);
  const [resolution, setResolution] = useState(1);
  const raw = useMemo(() => aggregateMarketCandles(market), [market]);
  const candles = compressCandles(raw.slice(-range), resolution);
  const first = candles[0]?.open || 100;
  const last = candles.at(-1)?.close || first;
  return (
    <div
      className="signal-modal"
      role="dialog"
      aria-modal="true"
      aria-label="Expanded market analysis"
    >
      <div className="signal-modal-card">
        <header>
          <div>
            <small>SIMULATED MARKET AGGREGATE</small>
            <h2>NIFTY 50 DECISION CHART</h2>
            <b className={last >= first ? "positive" : "negative"}>
              {signed((last / first - 1) * 100, 2)}
            </b>
          </div>
          <button onClick={close}>CLOSE ×</button>
        </header>
        <div className="chart-controls">
          <span>RANGE</span>
          {[
            [21, "30D"],
            [63, "90D"],
            [126, "6M"],
            [252, "1Y"],
          ].map(([value, label]) => (
            <button
              key={label}
              className={range === value ? "active" : ""}
              onClick={() => setRange(Number(value))}
            >
              {label}
            </button>
          ))}
          <span>RESOLUTION</span>
          {[
            [1, "DAY"],
            [5, "WEEK"],
            [21, "MONTH"],
          ].map(([value, label]) => (
            <button
              key={label}
              className={resolution === value ? "active" : ""}
              onClick={() => setResolution(Number(value))}
            >
              {label}
            </button>
          ))}
        </div>
        <SignalChart candles={candles} />
        <footer>
          <span>OHLC INDEXED TO 100 · {candles.length} BARS</span>
          <span>SIMULATION MODE · NOT A LIVE INDEX FEED</span>
        </footer>
      </div>
    </div>
  );
}

export function PortfolioRibbon({
  campaign,
  draft,
  trades,
  openWorkbench,
}: {
  campaign: Campaign;
  draft: DraftPortfolio;
  trades: Trade[];
  openWorkbench: () => void;
}) {
  return (
    <div className="cockpit-ribbon">
      <div className="ribbon-title">
        <span className="live-dot" /> MOVE CONSTRUCTION{" "}
        <b>BTI-OMS/{campaign.campaign_id}</b>
      </div>
      <div>
        <span>BUYING POWER</span>
        <b>{rupees(draft.buyingPowerPaise)}</b>
        <small>SELL-FIRST CASH</small>
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
        <b className={draft.deploymentPct >= 90 ? "positive" : "gold"}>
          {draft.deploymentPct.toFixed(1)}%
        </b>
        <small>MINIMUM 90%</small>
      </div>
      <div>
        <span>CASH AFTER</span>
        <b className={draft.cashAfterPaise < 0 ? "negative" : ""}>
          {rupees(draft.cashAfterPaise)}
        </b>
        <small>ESTIMATED</small>
      </div>
      <div>
        <span>POSITIONS</span>
        <b>{draft.positions.length}</b>
        <small>{draft.sectors.length} SECTORS</small>
      </div>
      <div>
        <span>TOP WEIGHT</span>
        <b className={draft.concentrationPct > 25 ? "negative" : "positive"}>
          {draft.concentrationPct.toFixed(1)}%
        </b>
        <small>CONCENTRATION</small>
      </div>
      <button onClick={openWorkbench}>REVIEW MOVE →</button>
    </div>
  );
}

export function IntelligenceDeck({ market }: { market: Market }) {
  const [tiles, setTiles] = useState<TileConfig[]>(loadLayout);
  const [settings, setSettings] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [dragged, setDragged] = useState<TileId | null>(null);
  const persist = (next: TileConfig[]) => {
    setTiles(next);
    localStorage.setItem(LAYOUT_KEY, JSON.stringify(next));
  };
  const update = (id: TileId, patch: Partial<TileConfig>) =>
    persist(
      tiles.map((tile) => (tile.id === id ? { ...tile, ...patch } : tile)),
    );
  const preset = (
    name: "FUNDAMENTAL" | "TECHNICAL" | "QUANT" | "SENTIMENT" | "ALL",
  ) => {
    const visible: Record<typeof name, TileId[]> = {
      FUNDAMENTAL: ["internals", "sector", "news"],
      TECHNICAL: ["pulse", "movers", "sector"],
      QUANT: ["pulse", "internals", "quant", "sector"],
      SENTIMENT: ["pulse", "movers", "news"],
      ALL: DEFAULT_TILES.map((tile) => tile.id),
    };
    persist(
      tiles.map((tile) => ({
        ...tile,
        hidden: !visible[name].includes(tile.id),
      })),
    );
  };
  const movers = [...market.stocks]
    .sort((a, b) => Math.abs(move(b)) - Math.abs(move(a)))
    .slice(0, 6);
  const gainers = [...market.stocks]
    .filter((stock) => move(stock) > 0)
    .sort((a, b) => move(b) - move(a))
    .slice(0, 5);
  const losers = [...market.stocks]
    .filter((stock) => move(stock) < 0)
    .sort((a, b) => move(a) - move(b))
    .slice(0, 5);
  const sectors = Object.entries(
    market.stocks.reduce<Record<string, Stock[]>>((result, stock) => {
      (result[stock.sector] ||= []).push(stock);
      return result;
    }, {}),
  )
    .map(([sector, stocks]) => ({
      sector,
      forecast:
        stocks.reduce((sum, stock) => sum + stock.forecast_pct, 0) /
        stocks.length,
    }))
    .sort((a, b) => b.forecast - a.forecast);
  const average = (
    field: keyof Pick<
      Stock,
      "peg" | "sharpe" | "volatility_pct" | "var_95_pct"
    >,
  ) =>
    market.stocks.reduce((sum, stock) => sum + stock[field], 0) /
    market.stocks.length;
  const medianPeg =
    [...market.stocks]
      .map((stock) => stock.peg)
      .filter((value) => value > 0)
      .sort((a, b) => a - b)[
      Math.floor(market.stocks.filter((stock) => stock.peg > 0).length / 2)
    ] || 0;
  const renderTile = (id: TileId) => {
    if (id === "pulse") {
      const history = aggregateMarketCandles(market).map((item) => item.close);
      const first = history[0] || 100;
      const last = history.at(-1) || first;
      const path = history
        .map(
          (value, index) =>
            `${index ? "L" : "M"}${(index / Math.max(1, history.length - 1)) * 380},${80 - ((value - Math.min(...history)) / Math.max(1, Math.max(...history) - Math.min(...history))) * 70}`,
        )
        .join(" ");
      return (
        <>
          <div className="pulse-quote">
            <span>NIFTY 50 · SIM</span>
            <b>24,812.45</b>
            <strong className={last >= first ? "positive" : "negative"}>
              {signed((last / first - 1) * 100, 2)}
            </strong>
          </div>
          <svg
            className="tile-chart"
            viewBox="0 0 380 90"
            onClick={() => setExpanded(true)}
          >
            <path d={path} />
          </svg>
          <button className="expand-analysis" onClick={() => setExpanded(true)}>
            EXPAND OHLC · 30D–1Y ↗
          </button>
        </>
      );
    }
    if (id === "internals")
      return (
        <div className="tile-matrix">
          <div>
            <span>MEDIAN PEG</span>
            <b>{medianPeg.toFixed(2)}</b>
          </div>
          <div>
            <span>MEAN SHARPE</span>
            <b>{average("sharpe").toFixed(2)}</b>
          </div>
          <div>
            <span>ADV / DEC</span>
            <b>
              {market.stocks.filter((stock) => move(stock) > 0).length} /{" "}
              {market.stocks.filter((stock) => move(stock) < 0).length}
            </b>
          </div>
          <div>
            <span>SIM 12M</span>
            <b className="positive">
              {signed(
                market.stocks.reduce(
                  (sum, stock) => sum + stock.forecast_pct,
                  0,
                ) / market.stocks.length,
              )}
            </b>
          </div>
        </div>
      );
    if (id === "sector")
      return (
        <div className="tile-bars">
          {sectors.slice(0, 7).map((item) => (
            <div key={item.sector}>
              <span>{item.sector}</span>
              <i>
                <b
                  style={{
                    width: `${Math.max(4, Math.min(100, item.forecast * 4))}%`,
                  }}
                />
              </i>
              <strong>{signed(item.forecast)}</strong>
            </div>
          ))}
        </div>
      );
    if (id === "quant")
      return (
        <div className="tile-matrix">
          <div>
            <span>VOLATILITY</span>
            <b>{average("volatility_pct").toFixed(1)}%</b>
          </div>
          <div>
            <span>VaR 95</span>
            <b className="negative">{average("var_95_pct").toFixed(1)}%</b>
          </div>
          <div>
            <span>LOW-RISK NAMES</span>
            <b>
              {
                market.stocks.filter(
                  (stock) => stock.volatility_pct <= average("volatility_pct"),
                ).length
              }
            </b>
          </div>
          <div>
            <span>SHARPE &gt; 1</span>
            <b>{market.stocks.filter((stock) => stock.sharpe > 1).length}</b>
          </div>
        </div>
      );
    if (id === "movers")
      return (
        <div className="tile-movers momentum-columns">
          {movers.every((stock) => Math.abs(move(stock)) < 0.0001) && (
            <p>Awaiting the first simulated market advance.</p>
          )}
          <section>
            <header>TOP GAINERS · MONTH</header>
            {gainers.map((stock) => (
              <div key={stock.ticker}>
                <b>{stock.ticker}</b>
                <span>{stock.sector}</span>
                <strong className="positive">{signed(move(stock))}</strong>
              </div>
            ))}
          </section>
          <section>
            <header>TOP LOSERS · MONTH</header>
            {losers.map((stock) => (
              <div key={stock.ticker}>
                <b>{stock.ticker}</b>
                <span>{stock.sector}</span>
                <strong className="negative">{signed(move(stock))}</strong>
              </div>
            ))}
          </section>
        </div>
      );
    return (
      <div className="compact-movers-board">
        <section>
          <header>5 GAINERS</header>
          {gainers.map((stock) => (
            <div key={stock.ticker}>
              <b>{stock.ticker}</b>
              <strong className="positive">{signed(move(stock))}</strong>
            </div>
          ))}
        </section>
        <section>
          <header>5 LOSERS</header>
          {losers.map((stock) => (
            <div key={stock.ticker}>
              <b>{stock.ticker}</b>
              <strong className="negative">{signed(move(stock))}</strong>
            </div>
          ))}
        </section>
      </div>
    );
  };
  return (
    <>
      <div className="desk-layout-bar">
        <span>CONFIGURABLE INTELLIGENCE DESK</span>
        {(
          ["FUNDAMENTAL", "TECHNICAL", "QUANT", "SENTIMENT", "ALL"] as const
        ).map((name) => (
          <button key={name} onClick={() => preset(name)}>
            {name}
          </button>
        ))}
        <button
          className="layout-button"
          onClick={() => setSettings(!settings)}
        >
          DESK LAYOUT ⚙
        </button>
      </div>
      {settings && (
        <div className="layout-drawer">
          {tiles.map((tile) => (
            <div key={tile.id}>
              <b>{TILE_LABELS[tile.id]}</b>
              <button onClick={() => update(tile.id, { hidden: !tile.hidden })}>
                {tile.hidden ? "SHOW" : "HIDE"}
              </button>
              <button onClick={() => update(tile.id, { wide: !tile.wide })}>
                {tile.wide ? "COMPACT" : "WIDE"}
              </button>
            </div>
          ))}
          <button onClick={() => persist(DEFAULT_TILES)}>
            RESET DEFAULT DESK
          </button>
        </div>
      )}
      <div className="intelligence-deck">
        {tiles
          .filter((tile) => !tile.hidden)
          .map((tile) => (
            <section
              key={tile.id}
              className={`intelligence-tile ${tile.wide ? "wide" : ""}`}
              draggable
              onDragStart={() => setDragged(tile.id)}
              onDragOver={(event) => event.preventDefault()}
              onDrop={() => {
                if (!dragged || dragged === tile.id) return;
                const next = [...tiles];
                const from = next.findIndex((item) => item.id === dragged);
                const to = next.findIndex((item) => item.id === tile.id);
                const [moving] = next.splice(from, 1);
                next.splice(to, 0, moving);
                persist(next);
                setDragged(null);
              }}
            >
              <header>
                <span>⋮⋮ {TILE_LABELS[tile.id]}</span>
                <div>
                  <button onClick={() => update(tile.id, { wide: !tile.wide })}>
                    {tile.wide ? "↙" : "↗"}
                  </button>
                  <button onClick={() => update(tile.id, { hidden: true })}>
                    −
                  </button>
                </div>
              </header>
              {renderTile(tile.id)}
            </section>
          ))}
      </div>
      {expanded && (
        <SignalModal market={market} close={() => setExpanded(false)} />
      )}
    </>
  );
}
