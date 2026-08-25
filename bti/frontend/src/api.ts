import { playableCash } from "./economics";

export type Campaign = {
  campaign_id: string;
  status: string;
  horizon_months: number;
  monthly_amount_rupees: number;
  current_move: number;
  moves_completed: number;
  holdings: Record<string, number>;
  cash_paise: number;
  portfolio_value_paise: number;
  benchmark_value_paise: number;
  alpha_pct: number;
  total_invested_paise: number;
  rating: number;
  last_result: MoveResult | null;
  market_label: string;
};
export type Stock = {
  ticker: string;
  sector: string;
  overall_rank: number;
  historical_cagr_pct: number;
  open_paise: number;
  high_paise: number;
  low_paise: number;
  close_paise: number;
  pe: number;
  pb: number;
  peg: number;
  forecast_pct: number;
  volatility_pct: number;
  sharpe: number;
  drawdown_pct: number;
  var_95_pct: number;
  expected_shortfall_95_pct: number;
  volume_index: number;
  sentiment_score: number;
  momentum_90d_pct: number;
  rsi_14: number;
  beta: number;
  roe_pct: number;
  earnings_growth_pct: number;
  profit_margin_pct: number;
  debt_to_equity: number;
  dividend_yield_pct: number;
  history_paise: number[];
  ohlc_history: Array<{
    month: number;
    open_paise: number;
    high_paise: number;
    low_paise: number;
    close_paise: number;
  }>;
  forecast_curve: Array<{
    months: number;
    annualized_pct: number;
    cumulative_pct: number;
  }>;
};
export type Market = {
  label: string;
  basis: string;
  month: number;
  stocks: Stock[];
};
export type Trade = { side: "BUY" | "SELL"; ticker: string; shares: number };
export type MoveResult = {
  score: number;
  classification: string;
  move: number;
  decision_quality: { headline: string; did_well: string[]; improve: string[] };
  execution: Array<
    Trade & { price_paise: number; cash_movement_paise: number }
  >;
  market_outcome: {
    headline: string;
    portfolio_return_pct: number;
    benchmark_return_pct: number;
    alpha_pct: number;
  };
  progress: {
    move: number;
    total: number;
    portfolio_value_paise: number;
    benchmark_value_paise: number;
  };
};

const TOKEN = "bti_access_token";
export class APIRequestError extends Error {
  constructor(
    message: string,
    public code: string,
    public status: number,
  ) {
    super(message);
  }
}
let renewal: Promise<void> | null = null;
async function createSession() {
  if (!renewal) {
    renewal = (async () => {
      const response = await fetch("/api/v1/showcase/session", {
        method: "POST",
        headers: { "content-type": "application/json" },
      });
      const payload = await response.json();
      if (!response.ok)
        throw new APIRequestError(
          payload?.error?.message || "Could not start a showcase session.",
          payload?.error?.code || "SESSION_START_FAILED",
          response.status,
        );
      localStorage.setItem(TOKEN, payload.data.access_token);
    })().finally(() => {
      renewal = null;
    });
  }
  await renewal;
}
async function request<T>(
  path: string,
  init: RequestInit = {},
  allowRecovery = true,
): Promise<T> {
  const token = localStorage.getItem(TOKEN);
  const response = await fetch(path, {
    ...init,
    headers: {
      "content-type": "application/json",
      ...(token ? { authorization: `Bearer ${token}` } : {}),
      ...init.headers,
    },
  });
  const payload = await response.json();
  if (!response.ok) {
    if (response.status === 401 && allowRecovery) {
      localStorage.removeItem(TOKEN);
      await createSession();
      const mutatingMove = path.includes("/moves") && init.method === "POST";
      if (mutatingMove)
        throw new APIRequestError(
          "The anonymous showcase restarted during a deployment. Start a fresh campaign before submitting another move.",
          "SESSION_RESET",
          401,
        );
      return request<T>(path, init, false);
    }
    throw new APIRequestError(
      payload?.error?.message ||
        "The game server could not complete that request.",
      payload?.error?.code || "REQUEST_FAILED",
      response.status,
    );
  }
  return payload.data as T;
}
export async function ensureSession() {
  if (localStorage.getItem(TOKEN)) {
    try {
      await request<Campaign[]>("/api/v1/campaigns");
      return;
    } catch {
      localStorage.removeItem(TOKEN);
    }
  }
  await createSession();
}
const withCurrentContribution = (campaign: Campaign): Campaign => ({
  ...campaign,
  cash_paise: playableCash(
    campaign.cash_paise,
    campaign.monthly_amount_rupees,
    campaign.status === "ACTIVE",
  ),
});
export const api = {
  campaigns: async () =>
    (await request<Campaign[]>("/api/v1/campaigns")).map(
      withCurrentContribution,
    ),
  create: async (monthly_amount_rupees: number, horizon_months: number) =>
    withCurrentContribution(
      await request<Campaign>("/api/v1/campaigns", {
        method: "POST",
        body: JSON.stringify({ monthly_amount_rupees, horizon_months }),
      }),
    ),
  market: (id: string) => request<Market>(`/api/v1/campaigns/${id}/market`),
  commit: async (id: string, expected_month: number, instructions: Trade[]) => {
    const response = await request<{ campaign: Campaign; result: MoveResult }>(
      `/api/v1/campaigns/${id}/moves`,
      {
        method: "POST",
        headers: {
          "idempotency-key": `${id}-${expected_month}-${crypto.randomUUID()}`,
        },
        body: JSON.stringify({
          expected_month: Math.max(0, expected_month - 1),
          instructions,
        }),
      },
    );
    return {
      ...response,
      campaign: withCurrentContribution(response.campaign),
    };
  },
};
