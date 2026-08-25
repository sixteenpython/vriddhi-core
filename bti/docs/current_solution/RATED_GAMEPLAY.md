# Rated gameplay release

## Product boundary

The active campaign deliberately has only two destinations:

1. **Market Monitor** — think, research and construct.
2. **Game Board** — review, commit, analyse and study history.

Puzzles, lessons, rankings, leaderboards, profiles and separate newsroom pages are not shipped in
this web release. The aim is to perfect the full investment game before adding an ecosystem around
it.

## One move

A move is a complete whole-share monthly portfolio decision. The player researches the current
simulated information set, creates net BUY/SELL share deltas, watches the live portfolio X-Ray and
reviews one sell-first execution sheet. The server allows commitment only when holdings, cash and
the minimum 90% deployment rule are valid.

Commitment is permanent. There is no takeback and no unrated mode.

## Market as opponent

At campaign creation the engine precomputes a deterministic regime schedule independent of future
player holdings. Each committed move advances to the next regime and generated information set.
The opponent therefore becomes progressively more demanding without inspecting the player’s move
and inventing a punishment for it.

## Repeat last move

From Move 2 onward, **Repeat Last Move** copies the previous signed whole-share instructions
exactly. It does not copy target weights and it never silently rescales quantities. Current prices,
cash and holdings can make the copied move invalid or strategically weak. The Game Board exposes
that risk and disables permanent execution until the move again satisfies the rules.

## Decision and outcome

After commitment the deterministic engine returns:

- move quality from 0–100 and a chess-like classification;
- a signed position evaluation centred on `+0.00`;
- public portfolio health and concrete improvement feedback;
- the independently simulated next-month player, Nifty and alpha outcome;
- rating change and an immutable notation for the move.

Decision quality is based on the pre-move information set. Outcome is shown separately and cannot
rewrite the decision grade.

## The cricket chase

Nifty is the team batting first. Its visible projected return supplies a dashed target trajectory
over the fixed campaign horizon; this is a projection, not disclosure of the hidden future market
path. The player line contains only committed months and therefore advances one over at a time.

The scoreboard reports Nifty’s current par, player score, lead or gap, overs remaining and the
required catch-up. Return labels remain **FORMING** for the first two moves because annualising such
a short cash-flow history would be visually misleading.

## Historical review without takeback

Every completed move is selectable. Review reconstructs the pre-move market, execution, score,
outcome and progress series at that moment. The chase chart rewinds to the selected move. Review is
read-only, the campaign revision does not change, and any current live draft remains preserved.

This distinction is fundamental:

- **review** changes the player’s viewpoint;
- **takeback** would change canonical history and is not supported.
