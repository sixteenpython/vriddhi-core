# BTI Playtest 02 — investor preview PWA and mobile UX rebuild

- **Build window:** 26–27 August 2026
- **Production release:** BTI Mobile v0.11.0
- **Live application:** <https://beat-the-index.onrender.com/>
- **Release commit:** `f1cf4c7`
- **Production storage:** dedicated Supabase PostgreSQL project
- **Distribution:** responsive HTTPS application plus installable PWA
- **Primary release device:** Samsung M30s, current Chrome, portrait orientation
- **Status:** deployed, production health verified, live mobile Market and Game surfaces inspected

This record continues
[`PLAYTEST_01_ENDGAME_SCOREBOARD_AND_SIMULATION_FAIRNESS.md`](PLAYTEST_01_ENDGAME_SCOREBOARD_AND_SIMULATION_FAIRNESS.md).
Playtest 01 established the canonical match scoreboard, immutable endgame and simulation-fairness
programme. Playtest 02 records the work required to turn that engine into a credible investor web
release and a deliberately designed phone experience.

## Release decision and constraint

The intended Android release originally assumed Google Play internal testing. The release boundary
changed when no Google Play developer account was available and paying the registration fee before
company incorporation was not appropriate. The approved investor path therefore became:

1. publish the complete desktop game at one stable HTTPS address;
2. make that address an installable Progressive Web App;
3. adapt the same React application for portrait mobile rather than ship a second game engine;
4. use anonymous guest mode for this preview;
5. preserve campaigns durably on the same browser/device;
6. defer Google identity, cross-device recovery, native packaging and Play Store distribution.

This is a distribution change, not a reduction in game integrity. Desktop browser, mobile browser
and installed PWA use the same server-owned campaign, market, accounting, scoring and endgame
contracts.

## Product boundary

### Desktop

Desktop remains the full professional workstation: a dense simulated trading environment for
research, filtering, portfolio construction, execution, match analysis and historical review.

### Mobile

Mobile is not a scaled-down screenshot of that workstation. It is a scroll-first decision journey:

- understand the current match and market regime;
- inspect the most decision-relevant signals;
- change analytical lenses;
- open progressively deeper security research;
- construct the same complete monthly portfolio move;
- review Portfolio X-Ray and cash reconciliation;
- commit permanently;
- see the Nifty chase, score and immutable history.

### Installed PWA

The installed application is the same responsive mobile client running in standalone display mode.
It adds a home-screen icon, branded launch experience and cached application shell. Gameplay remains
network-authoritative; simulated market and campaign APIs are not converted into offline state.

## Workstation hardening completed before mobile release

The web reference was first brought to an investor-ready baseline.

### Analyst-grade Market Monitor

- A 50-security governed universe is available in one spreadsheet-like table.
- Headers, security identity, price and position/draft context remain usable while navigating a
  wide table.
- Synchronized top and bottom horizontal navigation reduce the cost of comparing distant columns.
- Securities can be sorted and filtered by valuation, quality, risk, position and draft state.
- `HELD`, `DRAFTED`, `HELD + DRAFTED` and `NOT HELD` views support repeat-month workflows.
- BUY/SELL drafting is available directly from the market surface.
- Portfolio X-Ray updates projected valuation, risk, return, sector mix, cash and concentration.
- Market Pulse and Quant Risk remain interactive rather than becoming decorative empty panels.
- The Quant Risk drill-down covers the entire 50-security universe.
- The desktop navigation rail is collapsible and the intelligence rail is responsive to browser zoom.

### Information environment

- The Newswire occupies the primary space immediately above the market table.
- Every news card is actionable and opens either security research or an appropriate generated-news
  context.
- Top five gainers and losers are visible from Move 1.
- Nifty and security research charts support multiple periods and interactive OHLC inspection.
- Public price, valuation, quality, forecast, volatility, drawdown, VaR, Sharpe, momentum and
  sentiment signals create multiple plausible analytical strategies.
- Flashy signals are intentionally present as realistic attention traps. The game teaches whether a
  player can separate signal from noise; it does not hide distraction from the player.

All of this information is simulated and permanently governed by the visible simulation boundary.

### Two-screen rated gameplay

The playable product is intentionally reduced to two active-campaign destinations:

1. **Market Monitor** — think, research and build the complete move.
2. **Game Board** — reconcile, commit, analyse, follow the Nifty chase and revisit past moves.

Puzzles, lessons, rankings and leaderboards were removed from this release. The game itself must
first be complete enough to teach through play.

The Game Board preserves:

- permanent rated moves with no takeback;
- exact repeat-last-move as a deliberate strategic risk;
- player-versus-Nifty progress after every monthly move;
- position score, move quality and plain-English feedback;
- an immutable move ledger;
- historical review that rewinds the display without altering live state;
- explicit endgame verdict and downloadable final report.

## Multi-campaign and durable-state architecture

The left campaign rail now behaves like a decision-work history:

- create multiple campaigns;
- keep more than one campaign active;
- switch between campaigns and resume the selected live move;
- abort a campaign without deleting its record;
- revisit completed or aborted campaigns and inspect historical moves.

The browser receives an anonymous signed guest identity. The server owns sessions and campaign
state. In production, Render receives a private `DATABASE_URL` for a dedicated Supabase project.
The persistence layer stores:

- `bti_sessions` — signed anonymous identity and lifecycle data;
- `bti_campaigns` — canonical versioned campaign state and timestamps.

The backend locks a campaign mutation before applying deterministic state transitions. Local
development retains a JSON repository fallback when the database environment variable is absent.

This release guarantees restart-safe and same-device resume. It does not claim account ownership or
cross-device recovery. Clearing the browser profile removes the guest ownership token even though
the database record remains.

## PWA release package

The investor preview added:

- application manifest with the installed name **BTI — Beat the Index**;
- portrait-primary standalone display configuration;
- green-on-black icon and splash direction;
- service-worker shell caching with versioned cache invalidation;
- responsive application metadata and theme colours;
- same URL for desktop browser, mobile browser and installed PWA;
- persistent **SIMULATION MODE** disclosure in every form factor;
- production health reporting of release and durable-storage posture.

The public production health contract for this release is:

```json
{
  "status": "ok",
  "service": "bti-immersive-api",
  "release": "0.11.0",
  "storage": {
    "backend": "postgres",
    "durable": true,
    "healthy": true
  }
}
```

## Mobile UX failure found during release validation

The first responsive mobile build was functionally complete but visually unacceptable. Real-device
screenshots showed:

- campaign headlines and KPI tiles colliding;
- six-to-ten-pixel typography that was technically rendered but not comfortably readable;
- a Game Board chase area that appeared blank before Move 1;
- compressed cards that inherited workstation density;
- poor visual hierarchy between match state, market briefing and action controls;
- a bottom navigation regression that could consume the viewport.

The most important root cause was structural: `.mobile-match-card` and `.mobile-game-hero` are
semantic `header` elements. A generic mobile `header` height rule constrained these complete cards
to the application-header height. Child content then overflowed and overlapped. Fixing individual
margins would have hidden the symptom without repairing the component boundary.

## v0.11.0 mobile rebuild

### Separate phone presentation layer

A final `mobile-polish.css` layer is loaded after the workstation stylesheet. It owns phone-specific
composition without duplicating business logic. The new boundary:

- explicitly scopes the application header;
- lets campaign hero cards size to content;
- prevents horizontal page overflow;
- establishes readable phone typography and line heights;
- supplies consistent card rhythm and 12-pixel page gutters;
- provides larger touch targets for filters, share steppers and actions;
- respects bottom safe-area space;
- keeps the live move dock above the thumb navigation;
- permits execution-sheet headings and long labels to wrap.

### Mobile Market journey

The live Market screen now reads in this order:

1. rated campaign and move identity;
2. plain-language Nifty target headline;
3. player, Nifty and gap KPIs;
4. market-regime briefing;
5. market pulse and primary quantitative risk context;
6. horizontally scrollable monthly movers;
7. generated attention headline;
8. search and analytical lenses;
9. ranked security decision cards;
10. persistent live-move dock and bottom navigation.

The monthly mover calculation was corrected to use the latest 22 trading observations. Move 1 no
longer compares an identical opening and closing seed point and therefore no longer renders every
security as `+0.0%`. Production validation produced plausible cross-sectional dispersion, including
EICHERMOT `+11.4%`, SHRIRAMFIN `+10.3%`, BAJAJ-AUTO `-9.1%` and TATASTEEL `-8.3%` for the governed
test campaign. These are deterministic game values, not market quotations.

### Mobile Game journey

The Game Board now presents:

1. rated campaign and move identity;
2. a clear **You are chasing Nifty** match headline;
3. player, Nifty and wealth-gap KPIs;
4. a visible benchmark target trajectory from Move 1;
5. player catch-up points after each completed move;
6. execution record or first-move call to action;
7. move feedback and immutable history as the campaign develops.

The benchmark path is projected over the fixed campaign horizon using the governed benchmark annual
target and actual benchmark values where moves are already complete. The player path begins at the
same visible origin and extends only through completed moves. This preserves the cricket-chase
metaphor before the player has scored the first over.

### Navigation and cache integrity

- Moving between Market and Game resets document scroll position.
- The phone menu remains a drawer; the desktop `<<` / `>>` rail control is not forced into the
  portrait layout.
- The service-worker cache key advanced to `bti-mobile-preview-v0.11.0` so installed clients do not
  remain trapped on the defective shell.
- A stale installed application can be recovered by fully closing and reopening it; uninstall and
  reinstall is the final cache-reset path.

## Verification evidence

### Automated

- Frontend component/contract suites: **three files, six tests passed**.
- Backend and immersive-contract suites: **13 tests passed**.
- TypeScript and Vite production build: **passed**.
- Diff whitespace and secret scans: **passed**.

The contract suite checks the PWA cache identity, mobile stylesheet load order, component-height
boundary, readable typography, target-path construction and health release identity.

### Local visual

The release was exercised at a `360 × 800` phone viewport before deployment:

- Market campaign hero and KPI grid remained fully contained;
- page width did not exceed the viewport;
- monthly movers were non-zero and directionally varied;
- move dock and bottom navigation did not collide;
- Game hero was readable;
- Nifty target curve appeared before Move 1;
- move sheet fitted the phone and headings wrapped correctly.

### Production visual

After Render reported v0.11.0 with healthy PostgreSQL storage, a fresh 24-move production campaign
was created and inspected at the same phone viewport. The live Market screen showed the complete
mobile hierarchy and realistic mover dispersion. The live Game screen showed a populated purple
Nifty target curve, green player origin, first-move instruction and accessible navigation without
the collisions present in the submitted screenshots.

GitHub `master` and the local release commit were both `f1cf4c7` at deployment handoff.

## Release chronology

| Commit | Purpose |
|---|---|
| `9008d92` | Complete investor web match experience |
| `97b208d` | Deepen Market Monitor decision desk |
| `fd3d30d` | Extend market Newswire alongside the universe |
| `c2783b8` | Prevent stale frontend-shell caching |
| `d7918fc` | Make the market table an analyst grid |
| `77fcaa3` | Promote Newswire above the market grid |
| `bfc6d2d` | Add multi-campaign trading desk |
| `4ff452e` | Make analytics dock responsive to browser zoom |
| `a5c7f53` | Refine responsive Quant Risk dock |
| `e6ef64c` | Ship installable investor-preview PWA and durable storage |
| `f8e0d64` | Introduce dedicated mobile gameplay composition |
| `a7ec896` | Restore mobile viewport usability |
| `7eb1eae` | Enforce correct mobile bottom-navigation height |
| `f1cf4c7` | Rebuild and production-validate mobile campaign experience |

## Remaining release boundaries

- Physical Samsung M30s installation and full 24-move playtest remain a human acceptance step.
- Guest identity is not a replacement for Google authentication.
- Same-device resume is not cross-device resume.
- Render free-tier cold starts remain possible.
- The cached shell is not an offline game.
- Native Android packaging and Google Play distribution remain deferred until the developer-account
  and corporate release boundary is ready.
- Investor feedback may still change information priority, interaction choreography and visual
  polish; it must not fork the canonical engine.

## Next acceptance loop

1. Install the production PWA on the Samsung M30s.
2. Create and complete at least one 24-move campaign in portrait orientation.
3. Close and reopen after several intermediate moves to verify same-device resume.
4. Test Market lenses, security research, order construction, move review, commit, feedback,
   historical review and endgame.
5. Record every investor observation as a bounded playtest item with explicit acceptance criteria.
6. Preserve one engine and one API while improving desktop and mobile presentation independently.

