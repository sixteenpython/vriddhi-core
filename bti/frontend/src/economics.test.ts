import { describe, expect, it } from "vitest";
import { executionCash, initialLumpSumMinimum, playableCash } from "./economics";
import type { Stock } from "./api";

const stock = { ticker: "TEST", close_paise: 25_000 } as Stock;

describe("campaign economics", () => {
  it("makes the current monthly contribution playable without changing persisted cash", () => {
    expect(playableCash(1_250, 25_000, true)).toBe(2_501_250);
    expect(playableCash(1_250, 25_000, false)).toBe(1_250);
  });

  it("reconciles whole-share sell-first and buy-second execution", () => {
    expect(
      executionCash(
        100_000,
        [
          { side: "SELL", ticker: "TEST", shares: 2 },
          { side: "BUY", ticker: "TEST", shares: 5 },
        ],
        [stock],
      ),
    ).toBe(25_000);
  });

  it("allows whole-share residual cash at the one-lakh campaign floor", () => {
    expect(initialLumpSumMinimum(100_000)).toBe(9_000_000);
    expect(initialLumpSumMinimum(500_000)).toBe(10_000_000);
  });
});
