import { describe, expect, it } from "vitest";
import type { Campaign, Market, Stock, Trade } from "./api";
import {
  buildDraftPortfolio,
  draftDeltaFor,
  normalizeTrades,
  setDraftDelta,
} from "./portfolioDraft";

const stock = (ticker: string, price: number, sector: string) =>
  ({
    ticker,
    close_paise: price,
    sector,
    pe: 10,
    peg: 1,
    sharpe: 1.2,
    forecast_pct: 12,
    volatility_pct: 15,
  }) as Stock;
const campaign = {
  cash_paise: 100_000,
  holdings: { HELD: 4 },
} as unknown as Campaign;
const market = {
  stocks: [stock("HELD", 10_000, "Energy"), stock("NEW", 5_000, "IT")],
} as unknown as Market;

describe("portfolio-aware draft", () => {
  it("nets one ticker into a single executable instruction", () => {
    const trades: Trade[] = [
      { side: "BUY", ticker: "HELD", shares: 3 },
      { side: "SELL", ticker: "HELD", shares: 1 },
    ];
    expect(draftDeltaFor(trades, "HELD")).toBe(2);
    expect(setDraftDelta(trades, "HELD", -4, 4)).toEqual([
      { side: "SELL", ticker: "HELD", shares: 4 },
    ]);
    expect(setDraftDelta(trades, "HELD", -9, 4)[0].shares).toBe(4);
    expect(normalizeTrades(trades, { HELD: 4 })).toEqual([
      { side: "BUY", ticker: "HELD", shares: 2 },
    ]);
  });

  it("matches the server sell-first deployment rule", () => {
    const draft = buildDraftPortfolio(campaign, market, [
      { side: "SELL", ticker: "HELD", shares: 2 },
      { side: "BUY", ticker: "NEW", shares: 20 },
    ]);
    expect(draft.sellProceedsPaise).toBe(20_000);
    expect(draft.buyingPowerPaise).toBe(120_000);
    expect(draft.buyTotalPaise).toBe(100_000);
    expect(draft.cashAfterPaise).toBe(20_000);
    expect(draft.deploymentPct).toBeCloseTo(83.3333);
    expect(
      draft.positions.find((item) => item.stock.ticker === "HELD")
        ?.projectedShares,
    ).toBe(2);
  });
});
