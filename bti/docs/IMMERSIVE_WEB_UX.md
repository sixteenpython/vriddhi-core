# BTI immersive web UX

## Experience promise

BTI should feel like an investment strategy game with the information density of a market terminal
and the teaching clarity of Lichess analysis. It must not look like a dashboard, spreadsheet or
reskinned recommendation product.

The repeated emotional rhythm is:

```text
ORIENT -> INVESTIGATE -> CONSTRUCT -> COMMIT -> REVEAL -> LEARN -> CONTINUE
```

The primary test is whether a novice understands the next action while an experienced player can
inspect deeper evidence without leaving the board.

## Simulation is a first-class product invariant

BTI is a simulated investment strategy game for education and entertainment. Except for the real
names and tickers of listed securities, prices, financial metrics, forecasts, news, events,
portfolio outcomes and all other gameplay information are generated for the simulation. They are
not real-time market information, investment advice, or recommendations to buy or sell securities.

This boundary is part of the experience, not a footer-only disclaimer:

- every game and market surface carries a persistent **SIMULATION MODE** indicator;
- onboarding says: **The stocks are real. The market you are about to play in is not.**;
- campaign setup says that decisions affect the game, not real money;
- simulated news, charts, prices, metrics and outcomes remain labelled in their local context;
- no simulated data may be visually indistinguishable from a real-time feed without an explicit
  simulation indicator.

Absence of these signals from any playable financial surface is a release blocker.

## Visual direction

- Deep ink/navy game surface with warm off-white reading surfaces; lime is reserved for action and
  positive progress, amber for uncertainty, red for destructive/risk states and cyan for benchmark.
- Display typography creates campaign drama; tabular numerals keep prices and scores stable.
- Cards use depth, edge lighting and restrained gradients. Charts, numbers and hierarchy create the
  premium feeling—not decorative glass everywhere.
- Motion communicates state change: market advance, transaction execution, score reveal and alpha
  movement. Respect `prefers-reduced-motion` and never delay access to information solely for drama.
- Always show **SIMULATED MARKET** near market data and decision-support language near results.

## Application shell and information hierarchy

Desktop uses a persistent left rail, top campaign HUD and main board. Tablet collapses the rail.
Mobile uses a compact HUD and bottom navigation while preserving the same task order.

Persistent campaign HUD:

1. `BTI Campaign #…` and scenario label;
2. `Move 17 / 36` with campaign progress;
3. portfolio value, Nifty value and alpha;
4. cash available and monthly contribution;
5. provisional skill rating;
6. saved/sync status.

Primary navigation:

- **Market** — investigate the current information set;
- **Move** — construct and reconcile the monthly execution sheet;
- **Analysis** — review committed decisions, not future answers;
- **Learn** — lessons and puzzles grounded in demonstrated gaps;
- **Profile** — history and honest skill evidence.

## Screens

### 1. Landing and campaign setup

The first viewport says **Can you beat the index?**, explains one month/one complete move, and offers
a single **Start a campaign** action. Campaign setup freezes ₹10K–₹1L per month, 24/36/48/60 moves
and the available simulated scenario. It previews total contributions and approximate play length,
not projected winnings. A three-move practice introduction may be offered as explicitly unrated.

### 2. Market board — first wow moment

Top strip: Nifty level, monthly move, breadth, median PEG, portfolio risk and three event cards.

Below it, a searchable/sortable 50-stock board shows:

- ticker and sector;
- price and current movement;
- sparkline/OHLC access;
- PE, PB and PEG;
- forecast with uncertainty language;
- Sharpe, drawdown and risk;
- current holding and weight, if any.

Default columns prioritise price, movement, PEG, forecast, risk and sector. Advanced metrics are
progressively disclosed. Sorting is visibly player-controlled and never defaults to a hidden
"best stock" ranking. Colour is supplemented by sign, arrow and text.

Selecting a row opens a stock intelligence panel rather than navigating the player away from their
decision context. It contains a candlestick/line toggle, metric explanations in plain English,
current public evidence, portfolio fit and simulated event timeline. It does not say buy/sell or
expose hidden reference preference. **Add to move** stages a player-authored action.

### 3. Move workshop

This is a visual execution workbench, not a form table. It contains:

- current portfolio map by stock and sector;
- opening cash, contribution and estimated sale proceeds;
- BUY/SELL instruction composer using whole shares;
- ordered execution ledger with running cash;
- allocation-before/allocation-after comparison;
- concentration and utilisation warnings;
- sticky reconciliation footer: buying power, planned buys, remaining cash, utilisation;
- **Review complete move** as the only primary action.

HOLD is represented by retained holdings and requires no redundant transaction. A review modal
shows the final SELL-first/BUY-second ledger, consequences and immutability statement. The commit
button uses an idempotency key and becomes inert while in flight.

### 4. Move analysis — Stockfish moment

The reveal order is deliberate:

1. classification appears: **EXCELLENT**;
2. numerical decision score appears: **91**;
3. one plain-English judgment appears;
4. the player sees strengths and the highest-value improvement;
5. only then does the market advance and show outcome.

Decision quality and market luck occupy separate panels. A falling portfolio can still be a process
win; a lucky weak decision remains weak. Show portfolio return, Nifty return and alpha as outcome,
never as retroactive justification. The page ends with one dominant action: **Continue to move 18**.

During an active campaign, analysis may name public signals and decision dimensions but not show the
master portfolio. A completed campaign may show bounded counterfactuals without publishing the full
reference trajectory.

### 5. Live progression and final result

The HUD continuously shows portfolio versus Nifty and `Move N / horizon`. The campaign timeline
marks classifications without making red/green return the score. Completion presents:

- contributed capital, portfolio, benchmark and wealth alpha;
- correctly named annualised money-weighted returns;
- average decision quality and classification distribution;
- strongest skill and recurring weakness, only when evidence is sufficient;
- campaign record download;
- **Play another campaign**.

Resignation is deliberately secondary, requires confirmation, records a loss and is never triggered
by inactivity.

## Plain-English content rules

- Lead with the conclusion: “You improved expected growth without materially increasing downside.”
- Explain one causal relationship per sentence.
- Pair every unfamiliar metric with its decision meaning.
- Say “simulated”, “illustrative”, “estimate” and “uncertainty” where applicable.
- Never say guaranteed, safe, sure winner, should buy, future winner or optimal stock.
- Never celebrate profit if decision quality was weak; celebrate process separately.
- Do not show a score without an explanation or an explanation without inspectable evidence.

## Accessibility and responsive standards

Release target is WCAG 2.2 AA:

- semantic landmarks, heading order and real buttons/links;
- complete keyboard path with visible focus and a skip link;
- 44 x 44 CSS-pixel minimum pointer targets;
- text contrast at least 4.5:1 and large-text/UI contrast at least 3:1;
- no information conveyed only by colour, position, hover or animation;
- charts include a concise text summary and accessible data table;
- form errors identify the field, rule and corrective action;
- dialogs trap focus and restore it on close;
- live regions announce saved state, validation and move-score completion without reading every
  animated number;
- reduced motion removes count-up and market-transition animation;
- zoom to 200% without horizontal page scrolling, except deliberately scrollable data tables.

Responsive acceptance widths are 360, 390, 768, 1024 and 1440 CSS pixels. At 360 px the player can
complete the full move with no clipped primary action. At desktop width the market board and stock
panel can coexist. Financial tables use sticky identity columns or switch to cards; the page itself
must not acquire an accidental horizontal scrollbar.

## Complete ten-minute acceptance test

A fresh user on desktop must be able to complete this without instruction from the team:

1. Understand within 30 seconds that BTI is a simulated educational investment game.
2. Start a ₹25K, 24-move campaign.
3. Identify the move number, cash, portfolio, Nifty and simulation label.
4. Sort/filter the market and inspect two stocks.
5. Explain PEG and drawdown using the in-product help.
6. Stage a diversified whole-share BUY move.
7. See why an under-90% or overspent draft cannot be submitted.
8. Review the ordered execution sheet and commit exactly once.
9. See a score and understand why the decision received that classification.
10. Distinguish decision quality from the next month's market outcome.
11. See portfolio versus Nifty progress and continue to move 2.
12. Refresh the browser and recover the same committed state.

Failure of any step is a release blocker for the investor showcase. The observer records time to
first move, validation errors, abandoned steps, comprehension and whether the user voluntarily wants
to play move 2.

## Honest feature boundary

Working in the immersive showcase: anonymous local session, deterministic simulated campaign,
current market board, stock evidence, whole-share execution, atomic move, score reveal, progression,
lessons/puzzles already backed by content, resume on the same deployment and campaign export.

Future until implemented and tested: Google login, cloud-synchronised identity, real/live market
data, social leaderboard, multiplayer, tournaments, brokerage connection, Android, push alerts and
true historical point-in-time replay. Marketing and UI labels must mirror this boundary exactly.
