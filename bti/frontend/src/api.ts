import { playableCash } from "./economics";

export type Campaign = {
  campaign_id: string; status: string; horizon_months: number; monthly_amount_rupees: number;
  current_move: number; moves_completed: number; holdings: Record<string, number>; cash_paise: number;
  portfolio_value_paise: number; benchmark_value_paise: number; alpha_pct: number;
  total_invested_paise: number; rating: number; last_result: MoveResult | null; market_label: string;
};
export type Stock = {
  ticker: string; sector: string; open_paise: number; high_paise: number; low_paise: number;
  close_paise: number; pe: number; pb: number; peg: number; forecast_pct: number;
  volatility_pct: number; sharpe: number; drawdown_pct: number; var_95_pct: number;
  expected_shortfall_95_pct: number; history_paise: number[];
};
export type Market = { label: string; basis: string; month: number; stocks: Stock[] };
export type Trade = { side: "BUY" | "SELL"; ticker: string; shares: number };
export type MoveResult = {
  score: number; classification: string; move: number;
  decision_quality: { headline: string; did_well: string[]; improve: string[] };
  execution: Array<Trade & { price_paise: number; cash_movement_paise: number }>;
  market_outcome: { headline: string; portfolio_return_pct: number; benchmark_return_pct: number; alpha_pct: number };
  progress: { move: number; total: number; portfolio_value_paise: number; benchmark_value_paise: number };
};

const TOKEN = "bti_access_token";
async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const token = localStorage.getItem(TOKEN);
  const response = await fetch(path, {
    ...init,
    headers: { "content-type": "application/json", ...(token ? { authorization: `Bearer ${token}` } : {}), ...init.headers },
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload?.error?.message || "The game server could not complete that request.");
  return payload.data as T;
}
export async function ensureSession() {
  if (localStorage.getItem(TOKEN)) return;
  const data = await request<{ access_token: string }>("/api/v1/showcase/session", { method: "POST" });
  localStorage.setItem(TOKEN, data.access_token);
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
  campaigns: async () => (await request<Campaign[]>("/api/v1/campaigns")).map(withCurrentContribution),
  create: async (monthly_amount_rupees: number, horizon_months: number) => withCurrentContribution(await request<Campaign>("/api/v1/campaigns", { method: "POST", body: JSON.stringify({ monthly_amount_rupees, horizon_months }) })),
  market: (id: string) => request<Market>(`/api/v1/campaigns/${id}/market`),
  commit: async (id: string, expected_month: number, instructions: Trade[]) => {
    const response = await request<{ campaign: Campaign; result: MoveResult }>(`/api/v1/campaigns/${id}/moves`, { method: "POST", headers: { "idempotency-key": `${id}-${expected_month}-${crypto.randomUUID()}` }, body: JSON.stringify({ expected_month: Math.max(0, expected_month - 1), instructions }) });
    return { ...response, campaign: withCurrentContribution(response.campaign) };
  },
};
