# Playtest 04 — Response Contract and Multi-Mode Release Regression

**Release:** v0.12.1
**Date:** 27 August 2026
**Scope:** Classic, Rapid, Blitz, Newswire, desktop and portrait-mobile release paths

## Trigger

Investor playtesting exposed a raw browser parser error across Newswire and Game Board:

> Unexpected token '<', "<!DOCTYPE "... is not valid JSON

The visible error was not a financial-engine failure. The web client assumed every platform response was JSON. During a transient Render response or a mixed-version deployment window, an HTML platform page could therefore leak through as a JavaScript parser exception.

## Corrections

1. Every API response is now validated for a non-empty JSON content type before parsing.
2. HTML, empty and malformed responses become stable BTI errors; raw parser exceptions are never shown.
3. Player-profile refresh is explicitly non-critical. A move that has committed cannot be falsely reported as failed because the optional profile refresh was unavailable.
4. Unknown API routes remain JSON 404 responses and cannot fall through to the SPA shell.
5. The service-worker cache advanced to v0.12.1 so clients replace the older application shell.
6. Newswire header and desk filters are isolated from global shell header/navigation rules.
7. Newswire lead content is visible at the top of a bounded desktop card, and the portrait layout remains scroll-first.
8. Rapid and Blitz reports use CAGR labels throughout; Classic retains SIP XIRR.
9. A ₹1 lakh Rapid/Blitz campaign permits a whole-share residual by accepting 90% deployment. Larger campaigns continue to require at least ₹1 lakh initially.
10. The home statistic now says **INDEX BEATS**, meaning campaigns in which the player beat the index.

## Regression matrix

### Automated

- Frontend: 9/9 tests, including HTML-response, malformed-JSON and one-lakh deployment rules.
- API/content/immersive contracts: 17/17.
- Engine non-horizon suite: 12/12.
- Completed Classic campaigns: 24, 36, 48 and 60 months.
- TypeScript production compilation and Vite release build.

### Browser gameplay

- Completed a 24-month Blitz campaign through the desktop UI.
- Completed a 24-month Rapid campaign through the desktop UI, including an initial allocation and a second-stop HOLD.
- Confirmed successful endgame reports, immutable move histories and no error toast.
- Confirmed Newswire headline, desk filters, lead story, full stream, gainers and losers render at desktop width.
- Confirmed Newswire and Game Board at 412 × 915 portrait dimensions with no horizontal overflow.
- Confirmed mobile endgame shows verdict, player CAGR, Nifty CAGR and BTI rating.

## Important gameplay observation

A concentrated BPCL portfolio beat Nifty in the tested simulation but scored as a BLUNDER. This is intentional and desirable: BTI separates a lucky or path-dependent outcome from repeatable decision quality. The report states that the player won the outcome while retaining a fragile process.
