import { beforeEach, describe, expect, it, vi } from "vitest";
import { api, ensureSession } from "./api";

const values = new Map<string, string>();
Object.defineProperty(globalThis, "localStorage", {
  value: {
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => values.set(key, value),
    removeItem: (key: string) => values.delete(key),
  },
});

const response = (status: number, payload: unknown) =>
  Promise.resolve({
    ok: status >= 200 && status < 300,
    status,
    json: async () => payload,
  } as Response);

describe("showcase session recovery", () => {
  beforeEach(() => {
    values.clear();
    vi.restoreAllMocks();
  });

  it("renews an expired token and retries a safe campaign read", async () => {
    values.set("bti_access_token", "expired-token-value");
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockImplementationOnce(() =>
        response(401, {
          error: { code: "INVALID_SESSION", message: "invalid" },
        }),
      )
      .mockImplementationOnce(() =>
        response(201, { data: { access_token: "fresh-token-value" } }),
      )
      .mockImplementationOnce(() => response(200, { data: [] }));
    await ensureSession();
    expect(values.get("bti_access_token")).toBe("fresh-token-value");
    expect(fetchMock).toHaveBeenCalledTimes(3);
  });

  it("does not replay a move mutation after an expired session", async () => {
    values.set("bti_access_token", "expired-token-value");
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockImplementationOnce(() =>
        response(401, {
          error: { code: "INVALID_SESSION", message: "invalid" },
        }),
      )
      .mockImplementationOnce(() =>
        response(201, { data: { access_token: "fresh-token-value" } }),
      );
    await expect(
      api.commit("CAMPAIGN01", 1, [{ side: "BUY", ticker: "BPCL", shares: 1 }]),
    ).rejects.toMatchObject({
      code: "SESSION_RESET",
    });
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});
