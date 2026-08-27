import { playableCash } from "./economics";

export type Campaign = {
  campaign_id: string;
  status: string;
  mode: "CLASSIC" | "RAPID" | "BLITZ";
  capital_model: "MONTHLY_SIP" | "LUMP_SUM";
  horizon_months: number;
  monthly_amount_rupees: number;
  total_capital_rupees: number;
  decision_interval_months: number | null;
  total_decisions: number;
  months_completed: number;
  return_label: "SIP XIRR" | "CAGR";
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
  gameplay_mode: "RATED";
  current_regime: MarketRegime;
  move_history: MoveHistory[];
  performance_series: PerformancePoint[];
  can_repeat_last_move: boolean;
  last_move_instructions: Trade[];
  market_label: string;
  match_summary: MatchSummary;
  final_result: FinalResult | null;
  initial_market?: Market;
};
export type MatchSummary = {
  move: number;
  total_moves: number;
  overs_remaining: number;
  total_invested_paise: number;
  portfolio_value_paise: number;
  benchmark_value_paise: number;
  wealth_gap_paise: number;
  wealth_alpha_pct: number;
  portfolio_xirr_pct: number;
  benchmark_xirr_pct: number;
  xirr_advantage_pct: number;
  max_drawdown_pct: number;
  average_move_score: number;
  rating: number;
  classification_distribution: Record<string, number>;
  position: "LEADING" | "TRAILING" | "LEVEL";
};
export type FinalResult = {
  mode: "CLASSIC" | "RAPID" | "BLITZ";
  return_label: "SIP XIRR" | "CAGR";
  status: string;
  verdict: "BEAT_INDEX" | "INDEX_WON" | "PHOTO_FINISH";
  headline: string;
  months_completed: number;
  total_invested_paise: number;
  portfolio_value_paise: number;
  benchmark_value_paise: number;
  wealth_alpha_paise: number;
  portfolio_gain_paise: number;
  benchmark_gain_paise: number;
  wealth_alpha_pct: number;
  portfolio_money_weighted_annual_return_pct: number;
  benchmark_money_weighted_annual_return_pct: number;
  xirr_advantage_pct: number;
  max_drawdown_pct: number;
  average_move_score: number;
  rating: number;
  process_verdict: string;
  strategic_lesson: string;
  best_move: MoveHighlight | null;
  weakest_move: MoveHighlight | null;
  classification_distribution: Record<string, number>;
  decision_support_only: true;
};
export type MoveHighlight = {
  move: number;
  score: number;
  classification: string;
  notation: string;
};
export type MarketRegime = {
  month: number;
  label: string;
  narrative: string;
  difficulty: number;
};
export type PositionEvaluation = {
  value: number;
  display: string;
  label: string;
  player_advantage: boolean;
};
export type PortfolioHealth = {
  forecast: number;
  risk: number;
  concentration: number;
  sector_concentration: number;
  valuation: number;
  sharpe: number;
  drawdown_pct: number;
  var_95_pct: number;
  positions: number;
  sectors: number;
  top_weight_pct: number;
  health_score: number;
  health_label: string;
};
export type MoveHistory = {
  move: number;
  notation: string;
  score: number;
  classification: string;
  position_evaluation: PositionEvaluation;
  alpha_pct: number;
  rating_after: number;
  regime: MarketRegime;
};
export type PerformancePoint = {
  move: number;
  month?: number;
  total_invested_paise: number;
  portfolio_value_paise: number;
  benchmark_value_paise: number;
  wealth_gap_paise: number;
  alpha_pct: number;
  portfolio_xirr_pct: number;
  benchmark_xirr_pct: number;
  projected_annual_return_pct: number;
  benchmark_projected_annual_return_pct: number;
  position_evaluation: PositionEvaluation;
};
export type Stock = {
  ticker: string;
  name?: string;
  sector: string;
  asset_class?: "EQUITY" | "GOVERNMENT BOND" | "CORPORATE BOND" | "GOLD";
  yield_pct?: number;
  duration_years?: number;
  credit_quality?: string;
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
  regime?: MarketRegime;
  events?: Array<{ kind: string; headline: string; detail: string }>;
};
export type Trade = { side: "BUY" | "SELL"; ticker: string; shares: number };
export type MoveResult = {
  mode: "CLASSIC" | "RAPID" | "BLITZ";
  months_advanced: number;
  segment_series: Array<{
    month: number;
    portfolio_value_paise: number;
    benchmark_value_paise: number;
    portfolio_ohlc?: {
      open_paise: number;
      high_paise: number;
      low_paise: number;
      close_paise: number;
    };
    benchmark_ohlc?: {
      open_paise: number;
      high_paise: number;
      low_paise: number;
      close_paise: number;
    };
    alpha_pct?: number;
    portfolio_drawdown_pct?: number;
    event?: {
      desk: string;
      tone: "positive" | "negative" | "neutral";
      headline: string;
      detail: string;
      time: string;
    };
    regime: MarketRegime;
  }>;
  score: number;
  classification: string;
  move: number;
  notation: string;
  rating_after: number;
  regime: MarketRegime;
  position_evaluation: PositionEvaluation;
  portfolio_health: PortfolioHealth;
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
    month?: number;
    horizon_months?: number;
    return_label?: "SIP XIRR" | "CAGR";
    portfolio_value_paise: number;
    benchmark_value_paise: number;
    total_invested_paise: number;
    portfolio_xirr_pct: number;
    benchmark_xirr_pct: number;
    projected_annual_return_pct: number;
    benchmark_projected_annual_return_pct: number;
    position_evaluation: PositionEvaluation;
  };
  portfolio_before: {
    holdings: Record<string, number>;
    cash_paise: number;
    value_paise: number;
  };
  portfolio_after_execution: {
    holdings: Record<string, number>;
    cash_paise: number;
    value_paise: number;
    health: PortfolioHealth;
  };
  match_summary: MatchSummary;
  final_result?: FinalResult;
};
export type MoveReview = {
  review_mode: true;
  selected_move: number;
  live_move: number;
  result: MoveResult;
  market: Market;
  performance_series: PerformancePoint[];
  move_history: MoveHistory[];
  match_summary: MatchSummary;
};
export type PlayerProfile = {
  rating: number;
  completed: number;
  wins: number;
  losses: number;
  draws: number;
  beat_index_pct: number;
  best_alpha_pct: number;
  current_streak: number;
  by_mode: Record<
    "CLASSIC" | "RAPID" | "BLITZ",
    { completed: number; wins: number; losses: number; draws: number }
  >;
};
export type CampaignConfig = {
  mode: "CLASSIC" | "RAPID" | "BLITZ";
  horizon_months: number;
  monthly_amount_rupees?: number;
  total_capital_rupees?: number;
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

type APIEnvelope<T> = {
  data?: T;
  error?: { code?: string; message?: string };
};

async function readEnvelope<T>(response: Response): Promise<APIEnvelope<T>> {
  const contentType = response.headers?.get?.("content-type") || "";
  const raw = await response.text();
  if (!raw.trim()) {
    throw new APIRequestError(
      "The game server returned an empty response. Please retry in a moment.",
      "EMPTY_SERVER_RESPONSE",
      response.status,
    );
  }
  if (!contentType.toLowerCase().includes("json")) {
    throw new APIRequestError(
      "The game server is restarting or serving an outdated page. Refresh BTI and retry.",
      "NON_JSON_SERVER_RESPONSE",
      response.status,
    );
  }
  try {
    return JSON.parse(raw) as APIEnvelope<T>;
  } catch {
    throw new APIRequestError(
      "The game server returned incomplete data. Please retry in a moment.",
      "INVALID_JSON_SERVER_RESPONSE",
      response.status,
    );
  }
}

async function createSession() {
  if (!renewal) {
    renewal = (async () => {
      const response = await fetch("/api/v1/showcase/session", {
        method: "POST",
        headers: { "content-type": "application/json" },
      });
      const payload = await readEnvelope<{ access_token: string }>(response);
      if (!response.ok)
        throw new APIRequestError(
          payload?.error?.message || "Could not start a showcase session.",
          payload?.error?.code || "SESSION_START_FAILED",
          response.status,
        );
      if (!payload.data?.access_token)
        throw new APIRequestError(
          "The game server did not issue a valid session. Please refresh BTI.",
          "SESSION_START_FAILED",
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
  const payload = await readEnvelope<T>(response);
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
  if (payload.data === undefined)
    throw new APIRequestError(
      "The game server returned an incomplete response. Please retry.",
      "MISSING_RESPONSE_DATA",
      response.status,
    );
  return payload.data;
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
    campaign.status === "ACTIVE" && (campaign.mode || "CLASSIC") === "CLASSIC",
  ),
});
export const api = {
  campaigns: async () =>
    (await request<Campaign[]>("/api/v1/campaigns")).map(
      withCurrentContribution,
    ),
  create: async (config: CampaignConfig) =>
    withCurrentContribution(
      await request<Campaign>("/api/v1/campaigns", {
        method: "POST",
        body: JSON.stringify(config),
      }),
    ),
  profile: () => request<PlayerProfile>("/api/v1/profile"),
  market: (id: string) => request<Market>(`/api/v1/campaigns/${id}/market`),
  reviewMove: (id: string, move: number) =>
    request<MoveReview>(`/api/v1/campaigns/${id}/history/${move}`),
  abort: async (id: string) => {
    const response = await request<{ campaign: Campaign }>(
      `/api/v1/campaigns/${id}/abort`,
      { method: "POST", body: "{}" },
    );
    return withCurrentContribution(response.campaign);
  },
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
