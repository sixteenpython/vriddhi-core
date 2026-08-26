# BTI Playtest 01 — endgame, match scoreboard and simulation fairness

- **Date:** 25 August 2026
- **Campaign completed:** 24 monthly moves
- **Status:** investor web release implementation completed on 26 August 2026
- **Implementation decision:** canonical scoreboard/endgame shipped; fairness harness established

## Non-negotiable investor release definition

The final investor release is due by midnight on 26 August 2026. It is not sufficient to produce a
local Android build or a responsive website.

Two independent investor journeys must work:

### Mobile investor journey

1. Investor receives a Google Play link.
2. Investor opens the listing using the authorised Google account.
3. Investor installs BTI through Google Play.
4. Investor signs in to BTI with that Google account.
5. Investor starts a campaign, researches the market, commits moves and receives analysis.
6. Campaign state survives application restart and re-authentication.

The delivery target is an installable Google Play release. Given the deadline, the controlled path
is Google Play **internal testing** with the investor’s email allow-listed and the opt-in link tested
on a real device. Internal testing is still secure Play Store distribution, but the application is
available through its tester URL rather than public Play search. A production-searchable listing is
a separate goal and depends on account eligibility and Google review timing.

### Web investor journey

1. Investor opens the final published HTTPS URL.
2. Investor signs in with the authorised Google account.
3. Investor starts or resumes a campaign.
4. Investor completes the same canonical Market Monitor → Game Board loop.
5. Refresh and a new browser session preserve the authenticated campaign.

### Cross-client requirements

- Android and web use the same authenticated user identity, campaign engine, API contracts,
  scoring, market schedule and persistence layer.
- A campaign created on one client is visible on the other after sign-in.
- Anonymous showcase storage is not acceptable for the investor release.
- Google authentication failure, cancellation and expiry have polished recovery states.
- The simulation disclosure remains persistent on both clients.
- No hidden Vriddhi/reference intelligence is shipped to either client.
- Both clients must be exercised using investor-like accounts, not only a developer account.

### Release evidence required before handoff

- final web production URL and successful Google sign-in recording;
- signed Android App Bundle and immutable version code;
- Google Play tester opt-in URL and confirmed investor allow-list;
- clean installation from Google Play on a physical Android device;
- complete mobile and web golden-path recordings;
- shared campaign/resume verification across clients;
- backend health, durable persistence and error-recovery evidence;
- privacy policy, account/data handling, store listing and data-safety material;
- release commit, build provenance and rollback instructions.

### External dependency warning

Google states that internal test releases support up to 100 selected testers and normally make a new
Android App Bundle available within minutes, although a first test link or later changes can take
several hours. Therefore Play Console access, developer-account readiness, package identity,
signing, tester email addresses and Google Auth configuration must be confirmed at the beginning of
release day. Public production approval by midnight cannot be treated as an engineering-controlled
deadline.

Official references:

- [Google Play testing tracks](https://support.google.com/googleplay/android-developer/answer/9845334?hl=en)
- [Prepare and roll out a release](https://support.google.com/googleplay/android-developer/answer/9859348?hl=en)
- [Sign in with Google using Credential Manager](https://developer.android.com/identity/sign-in/credential-manager-siwg)

## Playtest iteration 02 — seasoned-trader workstation and attention traps

Implemented on 26 August 2026 as a targeted Market Monitor release:

- Monthly Movers now separates the six strongest gainers and losers, preserving the immediate
  visual temptation to chase green and avoid red.
- The Portfolio X-Ray rail is followed by an Attention Wire and crowd-sentiment heat board. These
  are realistic simulated information signals, explicitly labelled as noisy rather than alpha.
- The security table can sort and filter by `POSITION / DRAFT`, including held, drafted,
  held-or-drafted and not-held views. This makes repeat-month portfolio review practical.
- The market table has synchronized horizontal navigation at both its top and bottom and uses
  larger, high-contrast headers.
- Desktop navigation and intelligence rails are horizontally resizable. Compact and mobile
  breakpoints deliberately return to governed fixed layouts.
- The synthetic market now includes deterministic short-horizon mean reversion after unusually
  crowded moves. The market still advances independently of the player; it does not inspect or
  punish the player portfolio.

The eight-seed, 24-month strategy tournament after calibration produced mean wealth alpha of
`-0.12%` for the concentrated momentum strategy versus `+6.72%` for the high-Sharpe strategy.
Momentum won 50% of scenarios rather than becoming a universal shortcut. Lowest-PEG also remained
non-universal (`-1.21%` mean wealth alpha; 37.5% win rate). These are release diagnostics for game
fairness, not investment research or forecasts.

## Player observation

The first complete 24-move campaign was broadly enjoyable and the overall experience was decent.
The playtest exposed three product areas that should be improved together:

1. the campaign ends without an unmistakable declaration that the game is over or whether the
   player won;
2. the post-move experience does not assemble the player’s current match position into one clear
   scoreboard;
3. the relationship between the synthetic game and real-world Indian-market behaviour needs to be
   defined, measured and communicated more rigorously.

The player used a simple strategy: select low-PEG securities while maintaining diversification.
That strategy beat the simulated Nifty, but the result raised a useful question about whether BTI’s
market is a fair representation of real markets and why the apparent edge differs from Vriddhi’s
walk-forward evidence.

## Improvement 1 — explicit endgame verdict

At the final committed move, BTI must transition into a distinct campaign-complete state. It should
not leave the player on an ordinary move-analysis screen.

The primary verdict should be one of:

- **YOU BEAT THE INDEX**;
- **THE INDEX WON**;
- **PHOTO FINISH / NEAR DRAW**, using an explicitly documented tolerance.

The endgame report should show:

- total invested;
- final portfolio value and illustrative gain or loss;
- Nifty-equivalent ending value;
- portfolio and Nifty SIP XIRR;
- XIRR advantage and ending wealth lead or deficit;
- maximum drawdown and risk-adjusted measures;
- average move quality and classification distribution;
- starting and ending BTI rating;
- best and weakest decisions;
- the campaign’s main strategic lesson;
- access to the immutable move record and a downloadable report.

### Acceptance criteria

- The endgame appears automatically and exactly once after the final successful commit.
- Campaign status is canonically `COMPLETED`; refresh cannot reopen a new move.
- The winner is derived from final player and benchmark wealth using a documented near-draw band.
- XIRR is used for monthly cash-flow performance; ambiguous `CAGR` labels are absent.
- Decision quality and realised outcome remain visibly separate.
- The report can be revisited after refresh without mutating campaign history.

## Improvement 2 — after-every-move match scoreboard

Every committed move should conclude with a prominent answer to: **Where do I stand now?** This is
the equivalent of the scoreboard after every cricket over.

The scoreboard should include:

- move completed and total campaign moves;
- total invested to date;
- current portfolio and Nifty-equivalent values;
- portfolio and Nifty SIP XIRR;
- XIRR advantage and current wealth lead or deficit;
- portfolio maximum drawdown;
- average move quality to date;
- current BTI rating;
- moves/overs remaining.

For the first two moves, annualised cash-flow returns should continue to display **FORMING** instead
of unstable annualised percentages.

### Acceptance criteria

- The scoreboard is visible without leaving the Game Board after every move.
- All values come from canonical server state, not client recomputation.
- Historical review rewinds the scoreboard to the selected move.
- Returning live restores the latest scoreboard and preserves the current draft.
- At the last move, the scoreboard feeds directly into the endgame report.

## Improvement 3 — simulation-fairness programme

### Current truth

BTI provides a coherent deterministic synthetic market, not yet a statistically validated replica
of the Indian equity market. Realistic securities, metrics, regimes, accounting, a benchmark and a
private reference engine do not prove that return distributions, correlations, valuation effects,
drawdowns or factor premiums resemble historical Indian markets.

Vriddhi’s evidence is based on historical walk-forward out-of-sample evaluation. A BTI synthetic
campaign should not be expected to reproduce Vriddhi’s historical return automatically.

### Product objective

BTI need not predict the real market. It should be a fair decision-training environment in which:

- information available to the player is internally coherent;
- no single simplistic heuristic wins across every regime;
- good decisions improve the probability of good outcomes but do not guarantee them;
- valuation, quality, growth, diversification, risk and regime adaptation all matter;
- the market never inspects the submitted portfolio and fabricates a punishment;
- identical seeds and decisions reproduce identical outcomes;
- the simulation remains explicitly generated and non-investment advice.

### Strategy-archetype stress tests

Create deterministic tournament simulations for at least:

1. lowest PEG only;
2. highest forecast only;
3. highest Sharpe only;
4. lowest volatility only;
5. concentrated momentum;
6. diversified buy-and-repeat;
7. equal-weight diversification;
8. adaptive Vriddhi-like monthly rebalancing;
9. random selection with valid deployment;
10. cash-heavy or minimally deployed behaviour where rules permit comparison.

Run each archetype across many governed seeds and all supported horizons. Report win rate versus
Nifty, wealth-alpha distribution, XIRR advantage, volatility, maximum drawdown, move-quality
distribution, regime performance, turnover, concentration and heuristic dominance frequency.

### Calibration requirements

Check the synthetic market against governed historical reference ranges for:

- monthly index and security returns;
- cross-sectional and sector correlations;
- volatility clustering and drawdown duration;
- bull, bear, sideways, recovery and rotation regimes;
- valuation dispersion and mean reversion;
- growth revisions and fundamental deterioration;
- factor premiums and reversals;
- benchmark breadth and concentration;
- transaction-cost and turnover sensitivity, if costs enter gameplay.

This is a calibration exercise, not a claim that simulated prices are forecasts.

### Initial fairness hypothesis

A low-PEG diversified strategy should be credible but not universally optimal. Some apparently cheap
stocks should represent genuine growth at a reasonable price; others should reflect cyclical peaks,
weak quality, deteriorating growth or unreliable forecasts. Adaptive portfolio construction should
have a durable probabilistic edge across varied regimes without being guaranteed to win every
campaign.

### Acceptance criteria

- No static single-factor archetype dominates across all regimes and horizons.
- Random valid portfolios underperform disciplined adaptive strategies over a broad governed seed set.
- The hidden reference has an aggregate advantage, not perfect foresight in every move.
- Player-visible information is sufficient to explain decisions after commitment.
- Simulation parameters, calibration dataset version and results are versioned.
- A plain-English Method section distinguishes realism, calibration, uncertainty and non-prediction.

## Consolidated implementation sequence

When this backlog is approved alongside later playtest feedback:

1. freeze the endgame and scoreboard data contracts;
2. build deterministic campaign-summary aggregation and completion invariants;
3. build the strategy-archetype tournament harness before tuning outcomes;
4. measure the current engine baseline;
5. tune only against explicit fairness and realism targets;
6. add endgame and match-scoreboard UX;
7. add engine, API, browser and long-campaign regression tests;
8. run complete 24-, 36-, 48- and 60-move simulations;
9. deploy only after narrative, numerical and gameplay review.

## 26 August implementation record

The web release now derives both historical and live match status from one server-owned
`match_summary`. It shows invested capital, player and Nifty wealth, SIP XIRRs, the wealth gap,
contribution-adjusted maximum drawdown, average move quality, rating and moves remaining. The first
two SIP XIRRs display `FORMING`. Historical review uses the same aggregation at the selected move.

The final committed move now creates an immutable `final_result`. A 0.5% final-wealth band produces
`PHOTO FINISH`; otherwise the larger ending portfolio wins. The endgame separately reports outcome
and decision discipline, identifies the strongest and weakest move, explains the strategic lesson
and offers a downloadable plain-text match report.

`bti/tools/simulation_fairness.py` is the first repeatable strategy-archetype release gate. It found
that the former fixed-drift benchmark was too easy for the growth characteristics of the playable
universe, so simulation v3 now derives 85% of the monthly benchmark move from a trimmed broad-market
constituent return and 15% from the macro index factor. A three-seed, 24-month diagnostic then found
that diversified low PEG was **not** a universal exploit (66.7% win rate; +0.11% mean wealth alpha),
low risk won 33.3%, and cheapest/concentrated lost all paths. High-forecast and composite rules were
still strong in this deliberately small baseline, which is a candidate for broader calibration. These
figures are game-engine diagnostics—not investment evidence. Broader multi-horizon calibration is
still required before making any statistical-realism claim.

Automated evidence for this increment: 12 game-engine tests, nine content/architecture tests and
four API tests pass; the TypeScript production bundle compiles successfully.

## Open questions for later playtesting

- What wealth/XIRR difference should count as a near draw?
- Should endgame award separate badges for beating Nifty, decision quality and risk discipline?
- Should beating Nifty through consistently weak decisions produce a win with a warning?
- Should transaction costs and taxes be part of the game or an advanced mode?
- How much regime information should be visible before a move versus diagnosed afterward?
- Which report fields belong on screen versus only in the downloadable report?
