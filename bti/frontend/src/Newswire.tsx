import { useMemo, useState } from "react";
import type { Campaign, Market, Stock } from "./api";

const money = (paise: number) =>
  new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 }).format(paise / 100);
const signed = (value: number) => `${value >= 0 ? "+" : ""}${value.toFixed(2)}%`;

type Story = {
  id: string;
  desk: string;
  headline: string;
  summary: string;
  ticker?: string;
  tone: "positive" | "negative" | "neutral";
  time: string;
};

function storiesFor(market: Market, campaign: Campaign): Story[] {
  const equities = market.stocks.filter((stock) => (stock.asset_class || "EQUITY") === "EQUITY");
  const gainers = [...equities].sort((a, b) => b.momentum_90d_pct - a.momentum_90d_pct);
  const losers = [...gainers].reverse();
  const held = equities.filter((stock) => campaign.holdings[stock.ticker]);
  const regime = market.regime;
  const lead = regime?.label || "Selective growth";
  const result: Story[] = [
    {
      id: "macro-lead",
      desk: "MARKET DESK",
      headline: `${lead}: the tape rewards evidence, not certainty`,
      summary: `${regime?.narrative || "Cross-asset signals are mixed."} Prices and events are simulated for this rated campaign.`,
      ticker: gainers[0]?.ticker,
      tone: "neutral",
      time: "SIM 09:12",
    },
    {
      id: "rates",
      desk: "MACRO",
      headline: "Rates hold; duration and quality move back onto the desk",
      summary: "Government bonds steady the defensive sleeve while expensive equities face a higher hurdle.",
      ticker: "GILT10Y",
      tone: "neutral",
      time: "SIM 08:56",
    },
    {
      id: "gold",
      desk: "COMMODITIES",
      headline: "Gold catches a hedge bid as cross-asset uncertainty rises",
      summary: "The move diversifies equity risk, but the path remains volatile and difficult to time.",
      ticker: "GOLD",
      tone: "positive",
      time: "SIM 08:41",
    },
  ];
  gainers.slice(0, 5).forEach((stock, index) => result.push({
    id: `gain-${stock.ticker}`,
    desk: index === 0 ? "BREAKING · MOMENTUM" : `${stock.sector.toUpperCase()} DESK`,
    headline: `${stock.ticker} extends a visible run; valuation discipline becomes the test`,
    summary: `Simulated 90-day momentum is ${signed(stock.momentum_90d_pct)}. Forecast ${signed(stock.forecast_pct)}, PEG ${stock.peg.toFixed(2)} and Sharpe ${stock.sharpe.toFixed(2)} tell a less one-dimensional story.`,
    ticker: stock.ticker,
    tone: "positive",
    time: `SIM ${String(8 - index).padStart(2, "0")}:${15 + index * 7}`,
  }));
  losers.slice(0, 4).forEach((stock, index) => result.push({
    id: `loss-${stock.ticker}`,
    desk: "RISK WATCH",
    headline: `${stock.ticker} sells off as the market reprices the near-term narrative`,
    summary: `The red tape may signal deterioration—or a cheaper entry. Drawdown ${Math.abs(stock.drawdown_pct).toFixed(1)}%, VaR ${stock.var_95_pct.toFixed(1)}%, forecast ${signed(stock.forecast_pct)}.`,
    ticker: stock.ticker,
    tone: "negative",
    time: `SIM 07:${42 - index * 6}`,
  }));
  held.slice(0, 3).forEach((stock, index) => result.push({
    id: `held-${stock.ticker}`,
    desk: "YOUR BOOK",
    headline: `${stock.ticker} enters the portfolio news radar`,
    summary: `You hold ${campaign.holdings[stock.ticker]} shares worth about ${money(stock.close_paise * campaign.holdings[stock.ticker])}. Separate portfolio relevance from headline intensity.`,
    ticker: stock.ticker,
    tone: stock.sentiment_score >= 0 ? "positive" : "negative",
    time: `SIM 06:${35 - index * 5}`,
  }));
  (market.events || []).forEach((event, index) => result.push({
    id: `engine-${index}`,
    desk: event.kind.includes("RISK") ? "RISK" : "MARKET INTELLIGENCE",
    headline: event.headline,
    summary: event.detail,
    tone: event.kind.includes("RISK") ? "negative" : "neutral",
    time: `SIM 06:${String(20 - index * 4).padStart(2, "0")}`,
  }));
  const sectors = [...new Set(equities.map((stock) => stock.sector))];
  sectors.slice(0, 6).forEach((sector, index) => {
    const members = equities.filter((stock) => stock.sector === sector);
    const leader = [...members].sort((a, b) => b.forecast_pct - a.forecast_pct)[0];
    const averageMove = members.reduce((sum, stock) => sum + stock.momentum_90d_pct, 0) / Math.max(1, members.length);
    result.push({
      id: `sector-${sector}`,
      desk: "SECTORS",
      headline: `${sector} desk: ${leader?.ticker || "leaders"} sets the forward signal while breadth stays ${averageMove >= 0 ? "constructive" : "fragile"}`,
      summary: `${members.length} simulated securities tracked. Sector momentum ${signed(averageMove)}; the strongest forward estimate belongs to ${leader?.ticker || "the current leader"} at ${signed(leader?.forecast_pct || 0)}.`,
      ticker: leader?.ticker,
      tone: averageMove >= 0 ? "positive" : "negative",
      time: `SIM 05:${String(58 - index * 6).padStart(2, "0")}`,
    });
  });
  [...equities].sort((a, b) => b.earnings_growth_pct - a.earnings_growth_pct).slice(0, 4).forEach((stock, index) => result.push({
    id: `earnings-${stock.ticker}`,
    desk: "EARNINGS",
    headline: `${stock.ticker} screens near the top of the simulated earnings-growth board`,
    summary: `Earnings growth ${signed(stock.earnings_growth_pct)}, ROE ${stock.roe_pct.toFixed(1)}%, margin ${stock.profit_margin_pct.toFixed(1)}% and PE ${stock.pe.toFixed(1)}. Growth is useful only after testing its price.`,
    ticker: stock.ticker,
    tone: "positive",
    time: `SIM 05:${String(24 - index * 5).padStart(2, "0")}`,
  }));
  [...equities].sort((a, b) => b.sharpe - a.sharpe).slice(0, 4).forEach((stock, index) => result.push({
    id: `quant-${stock.ticker}`,
    desk: "QUANT",
    headline: `${stock.ticker} rises on the risk-adjusted return radar`,
    summary: `Sharpe ${stock.sharpe.toFixed(2)}, volatility ${stock.volatility_pct.toFixed(1)}%, drawdown ${stock.drawdown_pct.toFixed(1)}% and VaR ${stock.var_95_pct.toFixed(1)}%. No single ratio owns the decision.`,
    ticker: stock.ticker,
    tone: stock.sharpe >= 1 ? "positive" : "neutral",
    time: `SIM 04:${String(52 - index * 6).padStart(2, "0")}`,
  }));
  return result;
}

export function Newswire({ market, campaign, select }: { market: Market; campaign: Campaign; select: (stock: Stock) => void }) {
  const [desk, setDesk] = useState("ALL");
  const stories = useMemo(() => storiesFor(market, campaign), [market, campaign]);
  const movers = [...market.stocks]
    .filter((stock) => (stock.asset_class || "EQUITY") === "EQUITY")
    .sort((a, b) => b.momentum_90d_pct - a.momentum_90d_pct);
  const visible = stories.filter((story) => desk === "ALL" || story.desk.includes(desk));
  const equities = market.stocks.filter((stock) => (stock.asset_class || "EQUITY") === "EQUITY");
  const advancing = equities.filter((stock) => stock.close_paise >= stock.open_paise).length;
  const averageSentiment = equities.reduce((sum, stock) => sum + stock.sentiment_score, 0) / Math.max(1, equities.length);
  const averageVar = equities.reduce((sum, stock) => sum + stock.var_95_pct, 0) / Math.max(1, equities.length);
  const open = (story: Story) => {
    const stock = market.stocks.find((candidate) => candidate.ticker === story.ticker);
    if (stock) select(stock);
  };
  return (
    <section className="newswire-page">
      <div className="newswire-tape" aria-label="Simulated ticker tape">
        {[...movers.slice(0, 5), ...movers.slice(-3)].map((stock) => (
          <button key={stock.ticker} onClick={() => select(stock)}>
            <b>{stock.ticker}</b><span>{money(stock.close_paise)}</span>
            <em className={stock.momentum_90d_pct >= 0 ? "positive" : "negative"}>{signed(stock.momentum_90d_pct)}</em>
          </button>
        ))}
      </div>
      <header className="newswire-header">
        <div><span>BTI NEWSWIRE · SIMULATED EDITION · MONTH {market.month}</span><h1>The tape is loud. Your decision must be clearer.</h1></div>
        <div className="newswire-sentiment"><small>MARKET REGIME</small><b>{market.regime?.label || "OPEN"}</b><em>{(market.regime?.difficulty || 1).toFixed(2)}× market intensity</em></div>
      </header>
      <div className="newswire-dashboard">
        <section><span>MARKET BREADTH</span><b>{advancing} / {equities.length}</b><em>ADVANCING / UNIVERSE</em></section>
        <section><span>NEWS SENTIMENT</span><b>{averageSentiment.toFixed(0)} / 100</b><em>{averageSentiment >= 55 ? "POSITIVE TAPE" : averageSentiment <= 45 ? "DEFENSIVE TAPE" : "MIXED TAPE"}</em></section>
        <section><span>AVERAGE VaR</span><b>{averageVar.toFixed(1)}%</b><em>ONE-MONTH 95%</em></section>
        <section><span>YOUR BOOK</span><b>{Object.keys(campaign.holdings).length} POSITIONS</b><em>{Object.keys(campaign.holdings).length ? "PORTFOLIO FEED ACTIVE" : "NO HOLDINGS YET"}</em></section>
        <section><span>STORIES ON DESK</span><b>{stories.length}</b><em>ONLY CURRENT-MONTH EVIDENCE</em></section>
      </div>
      <nav className="newswire-filters" aria-label="Newswire desks">
        {["ALL", "MARKET", "MACRO", "EARNINGS", "SECTORS", "QUANT", "RISK", "YOUR BOOK"].map((item) => <button key={item} className={desk === item ? "active" : ""} onClick={() => setDesk(item)}>{item}</button>)}
      </nav>
      <div className="newswire-layout">
        {visible[0] && <button className="newswire-lead" onClick={() => open(visible[0])}>
          <span>{visible[0].desk} · {visible[0].time}</span><h2>{visible[0].headline}</h2><p>{visible[0].summary}</p><em>OPEN INTELLIGENCE →</em>
        </button>}
        <div className="newswire-stream">
          {visible.slice(1).map((story) => <button key={story.id} className={`newswire-story ${story.tone}`} onClick={() => open(story)}>
            <span>{story.desk} · {story.time}</span><b>{story.headline}</b><p>{story.summary}</p>{story.ticker && <em>{story.ticker} →</em>}
          </button>)}
        </div>
        <aside className="newswire-side">
          <section><span>TOP GAINERS · 90D</span>{movers.slice(0, 5).map((stock) => <button onClick={() => select(stock)} key={stock.ticker}><b>{stock.ticker}</b><em className="positive">{signed(stock.momentum_90d_pct)}</em></button>)}</section>
          <section><span>TOP LOSERS · 90D</span>{movers.slice(-5).reverse().map((stock) => <button onClick={() => select(stock)} key={stock.ticker}><b>{stock.ticker}</b><em className="negative">{signed(stock.momentum_90d_pct)}</em></button>)}</section>
          <section className="newswire-discipline"><span>TRADER'S RULE</span><b>A headline is not a thesis.</b><p>Check valuation, quality, risk and portfolio fit before turning attention into capital.</p></section>
          <section className="newswire-signal-matrix"><span>SIGNAL MATRIX</span>{[
            ["VALUATION", equities.filter((stock) => stock.peg > 0 && stock.peg <= 1).length],
            ["QUALITY", equities.filter((stock) => stock.roe_pct >= 15).length],
            ["LOW RISK", equities.filter((stock) => stock.var_95_pct <= averageVar).length],
            ["MOMENTUM", equities.filter((stock) => stock.momentum_90d_pct > 0).length],
          ].map(([label, count]) => <div key={String(label)}><b>{label}</b><progress max={equities.length} value={Number(count)} /><em>{count}/{equities.length}</em></div>)}</section>
        </aside>
      </div>
    </section>
  );
}
