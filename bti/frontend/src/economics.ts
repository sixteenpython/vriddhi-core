import type { Stock, Trade } from "./api";

export const playableCash = (
  cashPaise: number,
  monthlyRupees: number,
  active: boolean,
) => cashPaise + (active ? monthlyRupees * 100 : 0);

export const initialLumpSumMinimum = (totalCapitalRupees: number) =>
  Math.min(10_000_000, Math.round(totalCapitalRupees * 100 * 0.9));

export const executionCash = (
  openingPaise: number,
  trades: Trade[],
  stocks: Stock[],
) =>
  trades.reduce((cash, trade) => {
    const stock = stocks.find((candidate) => candidate.ticker === trade.ticker);
    if (!stock) throw new Error(`Unknown ticker ${trade.ticker}`);
    const movement =
      stock.close_paise * trade.shares * (trade.side === "SELL" ? 1 : -1);
    return cash + movement;
  }, openingPaise);
