# BTI move scoring

## Principle

BTI scores the quality of a complete monthly portfolio decision using information available when it was
made. It does not reward simple imitation and it does not allow a lucky next-month return to redefine a
weak process as a strong one.

## Feasible reference and equivalent good moves

The hidden Vriddhi provider solves the player's actual feasible problem: same market, cash, holdings,
monthly contribution and whole-share constraints. The scorer measures normalised opportunity loss
(`regret`) rather than distance from one weight vector. A portfolio inside a tolerance band for expected
return, downside, valuation quality and diversification can be excellent even when its holdings differ
from the principal reference.

## Decision-quality dimensions

- horizon-aware expected return and forecast confidence;
- valuation-growth balance, including PEG;
- volatility, drawdown, VaR and expected shortfall;
- diversification, concentration and correlated sector exposure;
- improvement from the prior portfolio;
- deployable-cash efficiency and whole-share feasibility;
- robustness across plausible perturbations.

The versioned policy converts total regret and fatal errors to a 0–100 score and one label: `BEST MOVE`,
`EXCELLENT`, `GOOD`, `INACCURACY`, `MISS` or `BLUNDER`. Thresholds are calibrated with generated feasible
portfolios and frozen with the campaign.

## Process versus outcome

The result contains two separate panels:

1. **Decision quality** — ex-ante score, label, strength and improvement.
2. **Market outcome** — realised portfolio return, benchmark return and alpha after the next month.

Outcome is diagnostic context and a small bounded longitudinal signal, never the dominant move score.
A good decision with a poor month is explicitly recognised as a process win.

Explanations may cite public facts but cannot reveal the hidden optimal execution sheet, frontier or
coefficients during an active campaign.
