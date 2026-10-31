# Dynamic Market Monitor & Research Newswire Synchronization

## 1. Executive Summary

On **October 31, 2026**, Beat the Index (BTI) was upgraded to resolve an architectural discrepancy between the **Market Monitor** terminal desk and the live simulated game engine.

Prior to this update, while the dedicated **Newswire** page (`Newswire.tsx`) generated dynamic stories from the active monthly universe, the desktop **Market Monitor** (`Terminal.tsx`) rendered five static, hardcoded headline cards referencing legacy securities (`BPCL`, `BEL`, `TCS`, `ITC`). This created cognitive friction whenever a promoted Vriddhi research refresh introduced new universe leaders or when market regimes diverged from the hardcoded text.

This update unifies the news generation architecture:
- Both **Market Monitor** and **Newswire** now share a single, deterministic intelligence engine.
- All headlines, price changes, P/E ratios, PEG ratios, 90-day momentums, and risk statistics reflect the active Vriddhi dataset (including the latest promoted `refresh-2026-10-30` release).
- Interactive navigation is preserved: clicking any news card routes directly to that security's live research sheet.

---

## 2. Architectural Philosophy: *"Deterministic First, AI Earns Its Place"*

Rather than deploying a non-deterministic Large Language Model (LLM) to generate market commentary at runtime, BTI adheres strictly to an enterprise-grade **"Deterministic First"** principle:

```
+-------------------------------------------------------------+
|               Vriddhi Research Engine (Python)              |
|   (Statistical factor models, PEG/Sharpe/VaR calculations)  |
+------------------------------+------------------------------+
                               | Promoted Release (SHA-256)
                               v
+-------------------------------------------------------------+
|                 BTI Simulation Engine (Python)              |
|        (Deterministic Monte Carlo & Regime Transitions)      |
+------------------------------+------------------------------+
                               | REST API JSON Contract
                               v
+-------------------------------------------------------------+
|              Deterministic Narrative Synthesizer            |
|            (Quant Filtering + Institutional Templates)      |
+------------------------------+------------------------------+
                               |
            +------------------+------------------+
            v                                     v
+-----------------------+             +-----------------------+
|     Market Monitor    |             |        Newswire       |
|   (Curated 5-Desk)    |             |   (Exhaustive Feed)   |
+-----------------------+             +-----------------------+
```

### Why Deterministic Synthesis Outperforms an LLM for This Surface:

| Attribute | Deterministic Engine (BTI Architecture) | Runtime LLM Call |
| :--- | :--- | :--- |
| **Execution Latency** | **< 1 millisecond** (executed purely client-side) | 2,000–5,000 ms (network hop + token generation) |
| **Operational Cost** | **$0.00** marginal cost per request | Per-token inference fees & provider subscription costs |
| **Factual Truthfulness** | **100% Guaranteed:** Headlines mathematically derived from engine state | Risk of hallucinated tickers, wrong percentages, or imaginary news |
| **Fair Play & Auditability** | **Deterministic:** Identical campaign seeds produce identical headlines for all players | Non-deterministic: Inconsistent player experiences across runs |
| **Availability / Offline** | Immune to external API rate limits, outages, or connectivity loss | Hard failure if LLM provider service degrades |

---

## 3. Implementation Architecture

### 3.1. Reusable Story Extraction (`bti/frontend/src/Newswire.tsx`)
The pure function `storiesFor(market, campaign)` was exported alongside its contract type `Story`:

```typescript
export type Story = {
  id: string;
  desk: string;
  headline: string;
  summary: string;
  ticker?: string;
  tone: "positive" | "negative" | "neutral";
  time: string;
};

export function storiesFor(market: Market, campaign: Campaign): Story[] {
  // Evaluates:
  // 1. Top gainers (sorted by momentum_90d_pct descending)
  // 2. Severe pullbacks / risk watches (drawdown_pct and var_95_pct)
  // 3. User's held positions (campaign.holdings)
  // 4. Macroeconomic regime narratives & engine breadth events
  // 5. Sector-wide momentum averages and forecast leaders
  // 6. Quantitative screens (highest Sharpe, top earnings growth)
}
```

### 3.2. Curated Desk Selection (`bti/frontend/src/Terminal.tsx`)
The Market Monitor requires exactly five curated cards across a horizontal grid. The `curatedTerminalStories` adapter selects a balanced, high-signal 5-story distribution:

```typescript
export function curatedTerminalStories(
  market: Market,
  campaign: Campaign,
): TerminalStory[] {
  const all = storiesFor(market, campaign);

  // Balanced 5-pillar editorial curation:
  const macro     = all.find((s) => s.id === "macro-lead") || all[0];
  const gainer    = all.find((s) => s.id.startsWith("gain-"));
  const loser     = all.find((s) => s.id.startsWith("loss-"));
  const held      = all.find((s) => s.id.startsWith("held-"));
  const valuation = all.find((s) => s.id.startsWith("earnings-") || s.id.startsWith("quant-"));
  const sector    = all.find((s) => s.id.startsWith("sector-"));

  const candidates = [macro, gainer, held || valuation, loser, sector].filter(Boolean) as Story[];
  
  // Graceful fallback to fill remaining slots if needed
  for (const s of all) {
    if (candidates.length >= 5) break;
    if (!candidates.some((c) => c.id === s.id)) candidates.push(s);
  }

  return candidates.slice(0, 5).map((story, index) => {
    const stock = story.ticker ? market.stocks.find((s) => s.ticker === story.ticker) : null;
    return {
      tag: story.desk,
      title: story.headline,
      body: story.summary,
      impact: stock ? `${stock.ticker} · ${stock.sector}` : (story.ticker || "CROSS-ASSET"),
      tone: mapTone(story, index),
      age: story.time,
      ticker: story.ticker || null,
    };
  });
}
```

### 3.3. React State Integration
Inside `MarketTerminal` and `NewsTerminal`, stories are memoized against campaign and market transitions:

```typescript
const stories = useMemo(
  () => curatedTerminalStories(market, campaign),
  [market, campaign],
);
```

Whenever the player commits a monthly move and advances time, `stories` re-evaluates in memory instantly, reflecting new leadership, sector rotations, and portfolio adjustments.

---

## 4. Verification and Release Gates

Before promotion, the update passed all automated verification gates:

1. **Backend & Contract Test Suite:**
   ```bash
   pytest vriddhi-core/bti/tests
   ============================= 40 passed in 45.35s =============================
   ```
2. **Frontend Vitest Suite:**
   ```bash
   npm run test --prefix bti/frontend
   Test Files  4 passed (4)
        Tests  12 passed (12)
   ```
3. **Production Client Build:**
   ```bash
   npm run build --prefix bti/frontend
   ✓ built in 502ms (Bundle: dist/assets/index-xrPhF3w9.js)
   ```
4. **Live Container Health Check on Render:**
   - Endpoint: `https://beat-the-index.onrender.com/api/v1/health`
   - Active Release: `refresh-2026-10-30`
   - Storage Durability: `{"backend": "postgres", "durable": true, "healthy": true}`

---

## 5. Summary of Benefits

- **Zero Manual Baby-Sitting:** Future monthly research refreshes (November 2026, December 2026, etc.) will automatically update both Market Monitor and Newswire without code edits.
- **Strict Domain Alignment:** No stock appears in the news feed unless it exists in the active 50-stock promoted universe.
- **Instant Responsiveness:** Zero latency penalty on Render free tier.
