# Playtest 09 — Automatic Vriddhi Baseline and Free-Tier Handoff

**Date:** 29 August 2026

**Release:** BTI v0.16.0

**Commits:** `814a53e` and cross-platform deployment correction `c6e728e`

## 1. Outcome

BTI no longer requires a separate manual intelligence refresh after Vriddhi is refreshed. Every new
campaign resolves the latest successfully promoted Vriddhi release at service startup and records
that release identity in the campaign.

The player-facing web and mobile/PWA experiences intentionally remain unchanged. This release closes
the backend governance and operational-continuity gap.

## 2. Final synchronization contract

```text
Vriddhi monthly refresh
  -> isolated candidate
  -> validation and regression gates
  -> promoted manifest and hashed artifacts
  -> reviewed merge to GitHub master
  -> automatic Render deployment
  -> BTI startup artifact verification
  -> production health exposes active release
  -> every new campaign uses that release
```

The effective selector is:

```text
BTI_BASELINE = latest successfully promoted Vriddhi release
```

If a refresh fails, the previously healthy release remains active. A partially generated or
hash-inconsistent candidate cannot silently enter BTI.

## 3. Backend changes

`VriddhiArtifacts` now:

- requires the expanded 50-stock table and 2Y/3Y/4Y/5Y portfolio bundles;
- requires a manifest release ID, data-through date and passed validation status;
- verifies all BTI-consumed artifacts against promoted SHA-256 hashes;
- permits only Git's semantically neutral LF/CRLF text-checkout normalization;
- rejects actual content changes, missing artifacts and invalid releases;
- exposes deployment-safe release metadata.

`BTIService` loads and verifies the governed release during startup. An invalid research release
therefore prevents a new deployment from becoming healthy instead of failing later during gameplay.

The health endpoint exposes:

- BTI application release;
- storage backend, durability and health;
- Vriddhi release ID and data-through date;
- validation state and verified artifact count;
- provider, simulation version and `LATEST_PROMOTED_VRIDDHI_RELEASE` synchronization mode.

## 4. Rated-campaign fairness across refreshes

Every v4 campaign freezes:

- its release ID and data-through date;
- the 50-stock baseline needed to reconstruct its deterministic market;
- its horizon-specific private reference weights.

Consequently, a new monthly Vriddhi release affects only newly created campaigns. It cannot move the
goalposts in an already-started rated campaign. Historical market reconstruction and private
reference scoring remain bound to the original campaign world.

## 5. Deployment incident and correction

The first v0.16.0 Render build correctly failed closed because the existing August manifest had been
hashed from Windows CRLF files while Render's Linux Git checkout contained LF text. The semantic
content was identical, but raw byte hashes differed.

Commit `c6e728e` introduced a narrow cross-platform rule: accept the promoted expected hash only when
the deployed artifact matches the raw, canonical-LF or canonical-CRLF representation. Any other byte
change still fails. A regression test reproduces the Windows-to-Linux checkout path.

This is evidence that the promotion gate is operational: the last healthy v0.15.2 service remained
available until the corrected v0.16.0 build passed.

## 6. Verification evidence

Final local release evidence:

- 40 BTI backend tests passed;
- 12 frontend tests passed;
- Vite production build passed;
- Git diff checks passed;
- local `master` and GitHub `master` synchronized.

Final production health evidence:

```text
BTI release                 0.16.0
Storage                     postgres / durable / healthy
Sync mode                   LATEST_PROMOTED_VRIDDHI_RELEASE
Vriddhi release             refresh-2026-08-14
Data through                2026-08-14
Validation                  passed
Verified artifacts          5
Simulation                  bti-capital-market-2026-08-v5
Home page                   HTTP 200
Service worker              HTTP 200 / v0.16.0
```

## 7. Assurance boundary

This release makes BTI automatically current with the latest **monthly promoted Vriddhi release**.
It does not turn BTI into a real-time quote service and does not claim exact future-price prediction.

The full investor-facing production-fidelity proposal remains in
`bti/docs/investor_due_diligence/CAPITAL_MARKET_INTELLIGENCE_ASSURANCE_AND_REFRESH.md`. It defines the
future licensed-feed, daily opening-anchor, point-in-time, statistical, fairness, legal and
independent-review controls that would be required for stronger commercial freshness claims.

## 8. Engineering handoff

The Enterprise build phase is now transitioning to repository-guided free-tier execution. Future BTI
changes should arrive as small task sheets with exact scope, protected financial/gameplay boundaries,
acceptance criteria, responsive UX cases, regression commands, deployment authority and rollback.

The cross-product cold-start checkpoint is maintained in the ConSaaS Core repository as
`Endpoint_Codex_ConSaaS.md`.
