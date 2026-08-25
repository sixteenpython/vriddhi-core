# BTI backlog

| ID | Slice | Acceptance criterion |
|---|---|---|
| BTI-001 | Reuse | Reusable, private and new capabilities are classified. |
| BTI-002 | Contracts | Versioned domain and HTTP contracts are documented and tested. |
| BTI-003 | Data | Vriddhi artifacts are read through a bounded, read-only adapter. |
| BTI-004 | Security | Threat model and secret/signing boundaries are documented. |
| BTI-005 | Simulation | Seeded monthly OHLC paths reproduce exactly without future leakage. |
| BTI-006 | Portfolio | Valid allocations settle deterministically with cash reconciliation. |
| BTI-007 | Move score | Moves receive stable score, class and bounded plain-English feedback. |
| BTI-008 | Rating | Rating changes by decision quality and uncertainty, not capital. |
| BTI-009 | Persistence | Games and immutable moves survive process restart. |
| BTI-010 | Audit | Every move contains engine versions and a tamper-evident chain hash. |
| BTI-011 | Auth | Valid Google tokens create short-lived server sessions. |
| BTI-012 | Game API | Create/read/list game endpoints enforce ownership and state. |
| BTI-013 | Move API | Idempotent commit prevents duplicate or stale moves. |
| BTI-014 | History | Completed games and ratings are retrievable. |
| BTI-015 | Learning | Versioned puzzle and lesson endpoints provide governed content. |
| BTI-016 | Privacy | Public projections never include hidden reference decisions. |
| BTI-017 | Android shell | Compose app targets API 36 and renders its navigation shell. |
| BTI-018 | Sign-in | Credential Manager performs Google sign-in and session exchange. |
| BTI-019 | Setup | Capital and 3/4/5-year horizon validation is clear. |
| BTI-020 | Market | Board and detail show only facts available at the move date. |
| BTI-021 | Allocation | User can construct and confirm a >=90% invested portfolio. |
| BTI-022 | Result | Move class, feedback and benchmark comparison render clearly. |
| BTI-023 | Progress | Month, wealth, benchmark and rating progress are visible. |
| BTI-024 | Final | Completed-game outcome explains what happened without advice claims. |
| BTI-025 | Observability | Structured request/game errors contain correlation IDs, not secrets. |
| BTI-026 | CI | Backend tests and Android lint/test/bundle are required gates. |
| BTI-027 | Smoke | Automated 36/48/60-move simulations complete successfully. |
| BTI-028 | Deploy | TLS backend and database backup/restore procedure are documented. |
| BTI-029 | Release | Signed AAB procedure and key custody are documented. |
| BTI-030 | Play | Store listing, data safety and financial declaration checklist exists. |

## Post-MVP

- BTI-031 point-in-time monthly knowledge-asset archive and corporate-action service.
- BTI-032 honest historical replay mode with constituent history.
- BTI-033 leagues, private challenges and anti-cheat controls.
- BTI-034 daily puzzles generated from frozen historical decision states.
- BTI-035 accessibility, localisation and low-bandwidth/offline cache.
- BTI-036 PostgreSQL repository and multi-instance concurrency validation.
- BTI-037 admin release console and artifact promotion workflow.
- BTI-038 calibrated difficulty and rating validation from live cohorts.
- BTI-039 privacy-preserving product analytics and deletion workflow.
- BTI-040 independent investment-policy and security review.
