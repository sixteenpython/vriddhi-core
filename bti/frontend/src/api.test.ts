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
    headers: new Headers({ "content-type": "application/json" }),
    text: async () => JSON.stringify(payload),
  } as Response);

const htmlResponse = (status: number) =>
  Promise.resolve({
    ok: false,
    status,
    headers: new Headers({ "content-type": "text/html; charset=utf-8" }),
    text: async () => "<!DOCTYPE html><html><body>upstream unavailable</body></html>",
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
    await api.campaigns();
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

  it("turns an HTML platform response into a stable user-facing API error", async () => {
    values.set("bti_access_token", "valid-token-value");
    vi.spyOn(globalThis, "fetch").mockImplementationOnce(() => htmlResponse(503));
    await expect(api.campaigns()).rejects.toMatchObject({
      code: "NON_JSON_SERVER_RESPONSE",
      status: 503,
      message:
        "The game server is restarting or serving an outdated page. Refresh BTI and retry.",
    });
  });

  it("rejects malformed JSON without exposing a browser parser exception", async () => {
    values.set("bti_access_token", "valid-token-value");
    vi.spyOn(globalThis, "fetch").mockImplementationOnce(() =>
      Promise.resolve({
        ok: true,
        status: 200,
        headers: new Headers({ "content-type": "application/json" }),
        text: async () => '{"data":',
      } as Response),
    );
    await expect(api.campaigns()).rejects.toMatchObject({
      code: "INVALID_JSON_SERVER_RESPONSE",
    });
  });
});
