# Beat the Index (BTI)

BTI is a web-first investment strategy game backed by Vriddhi's published intelligence.
Vriddhi remains the authoritative research engine; BTI owns game state, simulation, move analysis
and progression. The web showcase deliberately validates the game before a native Android client is
built around the same domain contracts.

The first release uses a deterministic, explicitly simulated 50-stock market derived from the
promoted Vriddhi release dated 2026-08-14. It does not pretend that current fundamentals were known
at historical dates.

See [`docs/BTI_REUSE_ASSESSMENT.md`](docs/BTI_REUSE_ASSESSMENT.md) and
[`docs/BTI_IMPLEMENTATION_PLAN.md`](docs/BTI_IMPLEMENTATION_PLAN.md).

## Run the playable showcase

From the repository root:

```powershell
.\.venv\Scripts\streamlit.exe run bti_web_app.py
```

Campaign state is saved under an unguessable campaign code for resume on the same deployment and can
also be downloaded as a deterministic JSON record. Shared identity, durable external persistence and
the Android client remain later slices.

See [`docs/BTI_IMPLEMENTATION_STATUS.md`](docs/BTI_IMPLEMENTATION_STATUS.md) for the exact boundary
between the playable showcase and planned platform infrastructure.
