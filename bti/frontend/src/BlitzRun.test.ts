import { describe, expect, it } from "vitest";
import type { MoveResult } from "./api";
import { marketRunCompletionLabel } from "./BlitzRun";

describe("Rapid market-run completion CTA", () => {
  it("opens another rebalance window at an intermediate Rapid stop", () => {
    const result = { mode: "RAPID" } as MoveResult;

    expect(marketRunCompletionLabel(result)).toBe("OPEN REBALANCE STOP →");
  });

  it("opens the result after the final Rapid market leg", () => {
    const result = {
      mode: "RAPID",
      final_result: { status: "COMPLETED" },
    } as MoveResult;

    expect(marketRunCompletionLabel(result)).toBe(
      "GAME COMPLETED · VIEW RESULT →",
    );
  });
});
