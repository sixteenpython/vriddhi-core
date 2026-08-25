# Testing, deployment and operations

## Automated test layers

### Python domain tests

The engine suite verifies:

- exact same-seed reproducibility and JSON restore;
- whole-share rules and exact cash movement;
- insufficient holdings, under-deployment and stale-move rejection;
- result immutability and resignation;
- no hidden-intelligence leakage;
- complete 24, 36, 48 and 60-move campaign simulations;
- approximate capital neutrality for equivalent decisions.

### ASGI/API tests

The server suite verifies health/content, authentication, campaign ownership, campaign creation,
validation without mutation, idempotent commit, persistence across application restart, stale-write
conflict, resignation and explicit invalid-input errors.

### Immersive-contract tests

Contract tests assert React/TypeScript presence, Streamlit absence, ASGI static serving, required
architecture language, public/private boundaries and persistent simulation disclosures.

### Frontend tests and build

The frontend runs TypeScript type-checking, Vitest economics/session-recovery/draft-portfolio tests
and a production Vite build. The release baseline passed 6 frontend tests and 23 Python tests.

## Manual live smoke test used for this baseline

1. Confirm Render marks the exact Git commit live.
2. Open the public URL with a returning/expired showcase token.
3. Verify automatic anonymous-session recovery.
4. Start a 36-move campaign.
5. Confirm professional contextual navigation.
6. Confirm 50 securities load with public metrics, multi-column sort/filter controls and direct
   portfolio construction.
7. Open a security and verify OHLC, forecast, ratios, generated catalysts and peer comparison.
8. Stage a whole-share order and confirm quantity, price and `+` control never collide.
9. Review and commit the move; verify the score reveal and advance to the next month.
10. Confirm Market Monitor and Newswire show directionally correct gainers and losers.
11. Confirm opening breadth shows `0 : 0` when every security is flat.
12. Confirm `+0.73%` renders once, not `++0.73%`.
13. Confirm the configurable cockpit layout persists and the expanded OHLC explorer renders data.
14. Confirm Month 2 shows Month 1 holdings, EXIT/HOLD semantics and net draft reconciliation.
15. Confirm simulation markers remain visible on every research/execution surface.

## Local runbook

From the repository root:

```powershell
cd bti\frontend
pnpm install
pnpm build
cd ..\..
.\.venv\Scripts\python.exe -m bti.server
```

Open <http://127.0.0.1:8000/>.

Recommended verification:

```powershell
cd bti\frontend
pnpm test
pnpm build
cd ..\..
.\.venv\Scripts\python.exe -m pytest bti\tests -q
.\.venv\Scripts\python.exe -m ruff check bti
```

## Production deployment

GitHub `sixteenpython/vriddhi-core`, branch `master`, is connected to the Render Blueprint service
`beat-the-index`. A pushed commit triggers a Docker auto-deploy. The health path is
`/api/v1/health`.

Release procedure:

1. run tests and production build locally;
2. review diff and protect unrelated workspace files;
3. commit the bounded change;
4. push the approved commit to GitHub;
5. wait for Render to report that exact commit as **live**;
6. run the public smoke test against the deployed asset;
7. report the URL and commit hash.

## Operational limitations

The free Render instance may cold-start after inactivity. `/tmp/bti-saves` is ephemeral, so current
campaign continuity is not guaranteed across deploys. These are accepted MVP constraints, not the
target architecture for public launch.
