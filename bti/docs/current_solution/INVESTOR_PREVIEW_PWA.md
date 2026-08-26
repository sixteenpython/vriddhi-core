# BTI investor preview PWA

## Release identity

- **Installed name:** BTI — Beat the Index
- **Release:** BTI Mobile Preview · v0.10.1
- **Production URL:** <https://beat-the-index.onrender.com/>
- **Primary device:** Samsung M30s, current Chrome, portrait orientation
- **Distribution:** installable web application; this release is not distributed through Google Play

The release preserves the complete deterministic BTI engine and adapts its two-screen workstation
to a phone. The browser and installed application call the same API and therefore cannot diverge in
campaign accounting, market evolution, scoring or final verdicts.

## Installation journey

1. Open the production URL in Chrome on the Samsung M30s.
2. Use the visible **Install BTI** action when Chrome makes it available. The equivalent Chrome menu
   route is **Add to Home screen** or **Install app**.
3. Accept the install prompt and launch BTI from its green-on-black home-screen icon.
4. Create a guest campaign, execute a move, close the installed application and reopen it.
5. Confirm that the same campaign and move are restored before the investor preview is accepted.

The manifest requests standalone display and portrait-primary orientation. The service worker caches
only the application shell and brand assets; gameplay API requests remain network-authoritative.

## Mobile information architecture

The phone experience is a dedicated scroll-first composition of the desktop product, not a compressed
analyst table and not a separate lite game. It preserves capability parity through progressive
disclosure.

- A compact header retains campaign identity, move number and the permanent simulation indicator.
- Navigation becomes a drawer on small screens. On desktop, the `<<` control collapses the rail and
  the corresponding `>>` control restores it.
- The campaign cockpit and table controls remain sticky where practical.
- Market becomes a vertical decision feed with a campaign scorecard, market briefing, top
  gainers/losers, generated headline and expandable 50-stock market-risk pulse.
- `ALL`, `VALUE`, `QUALITY`, `QUANT`, `TECHNICAL`, `NEWS` and `HELD` lenses reorder the universe and
  reveal the metrics relevant to the selected analytical style.
- Each security is a compact research-and-order card with an intramonth chart, contextual metrics,
  current/draft position and one-handed share controls. Tapping it opens the complete OHLC,
  fundamental, quantitative, forecast and news dossier.
- The live move dock remains fixed above bottom navigation. It opens a full-screen order sheet with
  Portfolio X-Ray, sector allocation, capital deployment and cash reconciliation.
- Game becomes an unfolding vertical match story: campaign score, cricket chase, execution sheet,
  permanent-commit gate, Stockfish-like score reveal, feedback and immutable move history.
- Market, Game and Campaigns form the permanent thumb-zone navigation.

The persistent **SIMULATION MODE** boundary remains visible in browser and installed modes.

## Persistence model

Production uses the dedicated `bti-production` Supabase PostgreSQL database through Render's private
`DATABASE_URL` environment variable. The backend creates and owns two tables:

- `bti_sessions` for anonymous signed session identities;
- `bti_campaigns` for versioned campaign state and timestamps.

Mutations lock the campaign row before applying deterministic state transitions. Secrets never enter
the repository or frontend bundle. Local development falls back to the existing JSON repository when
`DATABASE_URL` is absent.

This delivers restart-safe and same-device persistence. It does **not** provide cross-device recovery:
the guest identity token lives in that browser profile. Clearing site data loses the ownership token
even though the database record remains. Google OAuth is a later account release.

## Health and release evidence

`GET /api/v1/health` reports the release and storage posture. Production acceptance requires:

```json
{
  "status": "ok",
  "release": "0.10.1",
  "storage": {
    "backend": "postgres",
    "durable": true,
    "healthy": true
  }
}
```

The release gate also verifies the manifest, service worker, icon routes, desktop collapse control,
mobile navigation and responsive market/game surfaces. A clean production database was selected for
the preview; prior ephemeral showcase campaigns are intentionally not migrated.

## Explicit boundaries

- Guest mode only; no Google sign-in or cross-device account recovery.
- Network required for gameplay; offline mode is limited to the cached shell.
- No Google Play listing or native Android package in this release.
- Generated market information remains simulated and is not investment advice.
- The Render free instance can cold-start after inactivity.

After investor validation, the same PWA can be evaluated for Trusted Web Activity packaging and Play
distribution without forking the game engine.
