# BTI investor preview PWA

## Release identity

- **Installed name:** BTI — Beat the Index
- **Release:** BTI Investor Preview · v0.9.0
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

The phone experience is a prioritised composition of the desktop product, not a separate lite game.

- A compact header retains campaign identity, move number and the permanent simulation indicator.
- Navigation becomes a drawer on small screens. On desktop, the `<<` control collapses the rail and
  the corresponding `>>` control restores it.
- The campaign cockpit and table controls remain sticky where practical.
- Each market-table row becomes a compact research-and-order card containing identity, price,
  simulated move, draft controls, intramonth chart and the decisive valuation/risk signals.
- The live move tray remains accessible at the bottom of the screen so a player can reconcile the
  portfolio without losing research context.
- The Game Board stacks execution, position evaluation, move feedback, chase graph and history into
  one vertical decision narrative.

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
  "release": "0.9.0",
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
