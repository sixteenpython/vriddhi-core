import type { Campaign, Market, Stock, Trade } from "./api";

export type DraftPosition = {
  stock: Stock;
  heldShares: number;
  draftDelta: number;
  projectedShares: number;
  valuePaise: number;
  weightPct: number;
};

export type DraftPortfolio = {
  openingCashPaise: number;
  sellProceedsPaise: number;
  buyingPowerPaise: number;
  buyTotalPaise: number;
  cashAfterPaise: number;
  deploymentPct: number;
  positions: DraftPosition[];
  investedPaise: number;
  sectors: Array<[string, number]>;
  concentrationPct: number;
  weighted: {
    pe: number;
    peg: number;
    sharpe: number;
    forecast: number;
    volatility: number;
  };
};

export const draftDeltaFor = (trades: Trade[], ticker: string) =>
  trades
    .filter((trade) => trade.ticker === ticker)
    .reduce(
      (delta, trade) => delta + trade.shares * (trade.side === "BUY" ? 1 : -1),
      0,
    );

export const setDraftDelta = (
  trades: Trade[],
  ticker: string,
  requestedDelta: number,
  heldShares: number,
) => {
  const delta = Math.max(-heldShares, Math.trunc(requestedDelta));
  const remaining = trades.filter((trade) => trade.ticker !== ticker);
  if (!delta) return remaining;
  return [
    ...remaining,
    {
      side: delta > 0 ? ("BUY" as const) : ("SELL" as const),
      ticker,
      shares: Math.abs(delta),
    },
  ];
};

export const normalizeTrades = (
  trades: Trade[],
  holdings: Record<string, number>,
) => {
  const tickers = [...new Set(trades.map((trade) => trade.ticker))];
  return tickers.flatMap((ticker) =>
    setDraftDelta(
      [],
      ticker,
      draftDeltaFor(trades, ticker),
      holdings[ticker] || 0,
    ),
  );
};

export function buildDraftPortfolio(
  campaign: Campaign,
  market: Market,
  trades: Trade[],
): DraftPortfolio {
  const prices = new Map(market.stocks.map((stock) => [stock.ticker, stock]));
  const sellProceedsPaise = trades
    .filter((trade) => trade.side === "SELL")
    .reduce(
      (sum, trade) =>
        sum + (prices.get(trade.ticker)?.close_paise || 0) * trade.shares,
      0,
    );
  const buyTotalPaise = trades
    .filter((trade) => trade.side === "BUY")
    .reduce(
      (sum, trade) =>
        sum + (prices.get(trade.ticker)?.close_paise || 0) * trade.shares,
      0,
    );
  const buyingPowerPaise = campaign.cash_paise + sellProceedsPaise;
  const cashAfterPaise = buyingPowerPaise - buyTotalPaise;
  const rawPositions = market.stocks
    .map((stock) => {
      const heldShares = campaign.holdings[stock.ticker] || 0;
      const draftDelta = draftDeltaFor(trades, stock.ticker);
      const projectedShares = Math.max(0, heldShares + draftDelta);
      return {
        stock,
        heldShares,
        draftDelta,
        projectedShares,
        valuePaise: projectedShares * stock.close_paise,
      };
    })
    .filter((position) => position.projectedShares > 0);
  const investedPaise = rawPositions.reduce(
    (sum, position) => sum + position.valuePaise,
    0,
  );
  const positions = rawPositions.map((position) => ({
    ...position,
    weightPct: investedPaise ? (position.valuePaise / investedPaise) * 100 : 0,
  }));
  const sectors = Object.entries(
    positions.reduce<Record<string, number>>((result, position) => {
      result[position.stock.sector] =
        (result[position.stock.sector] || 0) + position.valuePaise;
      return result;
    }, {}),
  ).sort((a, b) => b[1] - a[1]);
  const average = (field: keyof Stock) =>
    investedPaise
      ? positions.reduce(
          (sum, position) =>
            sum + Number(position.stock[field]) * position.valuePaise,
          0,
        ) / investedPaise
      : 0;
  return {
    openingCashPaise: campaign.cash_paise,
    sellProceedsPaise,
    buyingPowerPaise,
    buyTotalPaise,
    cashAfterPaise,
    deploymentPct: buyingPowerPaise
      ? (buyTotalPaise / buyingPowerPaise) * 100
      : 0,
    positions,
    investedPaise,
    sectors,
    concentrationPct: positions.length
      ? Math.max(...positions.map((position) => position.weightPct))
      : 0,
    weighted: {
      pe: average("pe"),
      peg: average("peg"),
      sharpe: average("sharpe"),
      forecast: average("forecast_pct"),
      volatility: average("volatility_pct"),
    },
  };
}
