import { useEffect, useState } from "react";
import type { Campaign, Market, MoveResult, MoveReview, Trade } from "./api";
import { buildDraftPortfolio } from "./portfolioDraft";
import { initialLumpSumMinimum } from "./economics";
import { RapidClock } from "./RapidClock";
import { BlitzRun } from "./BlitzRun";

type Props = {
  campaign: Campaign;
  market: Market;
  trades: Trade[];
  result: MoveResult | null;
  reviewData: MoveReview | null;
  busy: boolean;
  execute: () => void;
  editMove: () => void;
  continueGame: () => void;
  reviewMove: (move: number) => void;
  returnLive: () => void;
};

const money = (paise: number, compact = false) => {
  const value = paise / 100;
  if (compact && Math.abs(value) >= 100_000)
    return `₹${(value / 100_000).toFixed(1)}L`;
  return `₹${Math.round(value).toLocaleString("en-IN")}`;
};
const signed = (value: number) =>
  `${value >= 0 ? "+" : ""}${value.toFixed(2)}%`;

function MobileChase({
  campaign,
  series,
}: {
  campaign: Campaign;
  series: Campaign["performance_series"];
}) {
  const annualTargetPct =
    series.at(-1)?.benchmark_projected_annual_return_pct || 8;
  const monthlyTargetRate = Math.pow(1 + annualTargetPct / 100, 1 / 12) - 1;
  const contributionPaise = campaign.monthly_amount_rupees * 100;
  const startingValue = campaign.mode === "CLASSIC" ? 0 : campaign.total_capital_rupees * 100;
  const niftyTarget = [{ move: 0, value: startingValue }];
  let projected = startingValue;
  for (let move = 1; move <= campaign.horizon_months; move += 1) {
    const actualPoint = series.find((point) => (point.month || point.move) === move);
    const actual = actualPoint?.benchmark_value_paise;
    projected =
      actual ?? projected * (1 + monthlyTargetRate) + contributionPaise;
    niftyTarget.push({ move, value: projected });
  }
  const player = [
    { move: 0, value: startingValue },
    ...series.map((point) => ({
      move: point.month || point.move,
      value: point.portfolio_value_paise,
    })),
  ];
  const max = Math.max(
    ...niftyTarget.map((point) => point.value),
    ...player.map((point) => point.value),
    contributionPaise,
    1,
  );
  const makePath = (points: Array<{ move: number; value: number }>) =>
    points
      .map(
        (point) =>
          `${(point.move / campaign.horizon_months) * 100},${54 - (point.value / max) * 48}`,
      )
      .join(" ");
  return (
    <section className="mobile-chase-card">
      <div>
        <span>
          <i className="player" />
          PLAYER
        </span>
        <span>
          <i className="nifty" />
          NIFTY TARGET
        </span>
      </div>
      <svg viewBox="0 0 100 58" preserveAspectRatio="none">
        <polyline className="nifty" points={makePath(niftyTarget)} />
        <polyline className="player" points={makePath(player)} />
        <circle className="player-origin" cx="0" cy="54" r="1.6" />
      </svg>
      <footer>
        <span>M1</span>
        <b>CHASE AFTER EVERY MOVE</b>
        <span>M{campaign.horizon_months}</span>
      </footer>
    </section>
  );
}

export function MobileGameBoard({
  campaign,
  market,
  trades,
  result,
  reviewData,
  busy,
  execute,
  editMove,
  continueGame,
  reviewMove,
  returnLive,
}: Props) {
  const [blitzRevealed, setBlitzRevealed] = useState(false);
  const historical = Boolean(reviewData);
  const draft = buildDraftPortfolio(campaign, market, trades);
  const rapidHold = !historical && !result && campaign.mode === "RAPID" && campaign.current_move > 1 && trades.length === 0;
  const preCommit = !historical && !result && (trades.length > 0 || rapidHold);
  const displayed = preCommit
    ? null
    : reviewData?.result || result || campaign.last_result;
  const series = reviewData?.performance_series || campaign.performance_series;
  const summary =
    reviewData?.match_summary ||
    displayed?.match_summary ||
    campaign.match_summary;
  const selectedMove =
    reviewData?.selected_move || displayed?.move || campaign.current_move;
  const execution = preCommit
    ? [
        ...trades.filter((trade) => trade.side === "SELL"),
        ...trades.filter((trade) => trade.side === "BUY"),
      ].map((trade) => {
        const price =
          market.stocks.find((stock) => stock.ticker === trade.ticker)
            ?.close_paise || 0;
        return {
          ...trade,
          price_paise: price,
          cash_movement_paise:
            (trade.side === "SELL" ? 1 : -1) * price * trade.shares,
        };
      })
    : displayed?.execution || [];
  const final = !historical
    ? displayed?.final_result || campaign.final_result
    : null;
  const activeMarketRun = Boolean(
    !historical &&
      result &&
      result.mode !== "CLASSIC" &&
      result.segment_series?.length &&
      !blitzRevealed,
  );
  useEffect(() => setBlitzRevealed(false), [result?.move]);
  const ready = preCommit && draft.cashAfterPaise >= 0 && (
    campaign.mode === "CLASSIC"
      ? draft.deploymentPct >= 90
      : rapidHold ||
        draft.buyTotalPaise >= initialLumpSumMinimum(campaign.total_capital_rupees) ||
        campaign.current_move > 1
  );
  return (
    <section className="mobile-game-page">
      <header className="mobile-game-hero">
        <small>
          {historical
            ? `REVIEWING MOVE ${selectedMove}`
            : `RATED CAMPAIGN · ${campaign.mode || "CLASSIC"} · DECISION ${campaign.current_move}/${campaign.total_decisions || campaign.horizon_months}`}
        </small>
        <h1>
          {final
            ? final.headline
            : summary.position === "LEADING"
              ? "You are beating the index."
              : "You are chasing Nifty."}
        </h1>
        <div>
          <span>
            <small>PLAYER</small>
            <b>{money(summary.portfolio_value_paise, true)}</b>
          </span>
          <span>
            <small>NIFTY</small>
            <b>{money(summary.benchmark_value_paise, true)}</b>
          </span>
          <span>
            <small>GAP</small>
            <b
              className={
                summary.wealth_gap_paise >= 0 ? "positive" : "negative"
              }
            >
              {money(summary.wealth_gap_paise, true)}
            </b>
          </span>
        </div>
      </header>
      {historical && (
        <div className="mobile-review-mode">
          <button
            disabled={selectedMove <= 1}
            onClick={() => reviewMove(selectedMove - 1)}
          >
            ← M{selectedMove - 1}
          </button>
          <b>IMMUTABLE REVIEW</b>
          <button
            disabled={selectedMove >= campaign.moves_completed}
            onClick={() => reviewMove(selectedMove + 1)}
          >
            M{selectedMove + 1} →
          </button>
          <button onClick={returnLive}>RETURN LIVE</button>
        </div>
      )}
      <RapidClock campaign={campaign} onExpire={execute} disabled={busy || historical || Boolean(result)} />
      {activeMarketRun && result && <BlitzRun result={result} onComplete={() => setBlitzRevealed(true)} />}
      {!activeMarketRun && <>
      {final && (
        <section className="mobile-final-card">
          <span>CAMPAIGN COMPLETE</span>
          <h2>
            {final.verdict === "BEAT_INDEX"
              ? "YOU BEAT THE INDEX"
              : final.verdict === "PHOTO_FINISH"
                ? "PHOTO FINISH"
                : "NIFTY WON"}
          </h2>
          <p>{final.strategic_lesson}</p>
          <div>
            <b>
              {signed(final.portfolio_money_weighted_annual_return_pct)}
              <small>PLAYER {final.return_label || "SIP XIRR"}</small>
            </b>
            <b>
              {signed(final.benchmark_money_weighted_annual_return_pct)}
              <small>NIFTY {final.return_label || "SIP XIRR"}</small>
            </b>
            <b>
              {final.rating}
              <small>BTI RATING</small>
            </b>
          </div>
        </section>
      )}
      {!historical && result?.mode === "RAPID" && campaign.status === "ACTIVE" && (
        <section className="rapid-stop-card mobile">
          <div><span>RAPID STOP · MONTH {campaign.months_completed}</span><h2>Care to rebalance?</h2><p>Tap through the replay, news and current portfolio signals before deciding whether to HOLD or change course.</p><RapidClock campaign={campaign} onExpire={execute} disabled={busy || historical} forceActive /></div>
          <button className="primary full" onClick={continueGame}>OPEN THE NEXT DECISION WINDOW →</button>
        </section>
      )}
      <MobileChase campaign={campaign} series={series} />
      {summary.move > 0 && (
        <section className="mobile-score-strip">
          <span>
            <small>INVESTED</small>
            <b>{money(summary.total_invested_paise, true)}</b>
          </span>
          <span>
            <small>PLAYER {campaign.return_label || "SIP XIRR"}</small>
            <b>{signed(summary.portfolio_xirr_pct)}</b>
          </span>
          <span>
            <small>NIFTY {campaign.return_label || "SIP XIRR"}</small>
            <b>{signed(summary.benchmark_xirr_pct)}</b>
          </span>
          <span>
            <small>MOVES LEFT</small>
            <b>{summary.overs_remaining}</b>
          </span>
        </section>
      )}

      <section className="mobile-execution-card">
        <header>
          <div>
            <small>{preCommit ? "FINAL REVIEW" : "EXECUTION RECORD"}</small>
            <h2>
              {preCommit
                ? "This move is permanent."
                : displayed?.notation || "Build your first move"}
            </h2>
          </div>
          {preCommit && <button onClick={editMove}>EDIT</button>}
        </header>
        {execution.map((item, index) => (
          <div className="mobile-execution-row" key={`${item.ticker}-${index}`}>
            <b>{String(index + 1).padStart(2, "0")}</b>
            <span>
              <strong>
                {item.side} {item.shares} {item.ticker}
              </strong>
              <small>{money(item.price_paise)} per share</small>
            </span>
            <em
              className={
                item.cash_movement_paise >= 0 ? "positive" : "negative"
              }
            >
              {money(item.cash_movement_paise)}
            </em>
          </div>
        ))}
        {!execution.length && (
          <p className="mobile-board-empty">
            {rapidHold ? "Hold the current portfolio and advance the simulation one year." : "Return to Market, research the decision universe and construct a complete portfolio move."}
          </p>
        )}
        {preCommit && (
          <>
            <div className="mobile-commit-summary">
              <span>
                <small>DEPLOYMENT</small>
                <b>{draft.deploymentPct.toFixed(1)}%</b>
              </span>
              <span>
                <small>CASH AFTER</small>
                <b>{money(draft.cashAfterPaise)}</b>
              </span>
              <span>
                <small>POSITIONS</small>
                <b>{draft.positions.length}</b>
              </span>
            </div>
            <button
              className="primary full"
              disabled={busy || !ready}
              onClick={execute}
            >
              {busy
                ? "EVALUATING…"
                : !ready
                  ? campaign.mode === "CLASSIC" ? "ADJUST MOVE TO PASS THE 90% GATE" : "DEPLOY AT LEAST ₹1 LAKH"
                  : rapidHold ? "RECORD HOLD · ADVANCE ONE YEAR →" : campaign.mode === "BLITZ" ? "RUN FULL CAMPAIGN →" : "EXECUTE PERMANENT MOVE →"}
            </button>
          </>
        )}
      </section>

      {displayed && (
        <section className="mobile-verdict-card">
          <div className="mobile-score-orb">
            <span>{displayed.score}</span>
            <small>/100</small>
          </div>
          <div>
            <small>
              MOVE {displayed.move} · POSITION{" "}
              {displayed.position_evaluation.display}
            </small>
            <h2>{displayed.classification}</h2>
            <p>{displayed.decision_quality.headline}</p>
          </div>
          <div className="mobile-outcomes">
            <span>
              PLAYER{" "}
              <b>{signed(displayed.market_outcome.portfolio_return_pct)}</b>
            </span>
            <span>
              NIFTY{" "}
              <b>{signed(displayed.market_outcome.benchmark_return_pct)}</b>
            </span>
            <span>
              ALPHA <b>{signed(displayed.market_outcome.alpha_pct)}</b>
            </span>
          </div>
          <section>
            <small>WHAT WORKED</small>
            {displayed.decision_quality.did_well.map((item) => (
              <p key={item}>✓ {item}</p>
            ))}
          </section>
          <section className="edge">
            <small>NEXT EDGE</small>
            {displayed.decision_quality.improve.map((item) => (
              <p key={item}>↗ {item}</p>
            ))}
          </section>
          {!historical && result && campaign.status === "ACTIVE" && (
            <button className="primary full" onClick={continueGame}>
              {campaign.mode === "RAPID" ? "CARE TO REBALANCE? →" : `BUILD MOVE ${campaign.current_move} →`}
            </button>
          )}
        </section>
      )}

      <section className="mobile-history-card">
        <header>
          <span>MOVE HISTORY</span>
          <small>TAP TO REPLAY THE ANALYSIS</small>
        </header>
        {campaign.move_history
          .slice()
          .reverse()
          .map((item) => (
            <button
              key={item.move}
              className={
                selectedMove === item.move && displayed ? "active" : ""
              }
              onClick={() => reviewMove(item.move)}
            >
              <b>M{String(item.move).padStart(2, "0")}</b>
              <span>
                {item.classification}
                <small>{signed(item.alpha_pct)} alpha</small>
              </span>
              <strong>{item.score}</strong>
              <em
                className={
                  item.position_evaluation.value >= 0 ? "positive" : "negative"
                }
              >
                {item.position_evaluation.display}
              </em>
            </button>
          ))}
        {!campaign.move_history.length && (
          <p>Your permanent rated record begins after Move 1.</p>
        )}
      </section>
      </>}
    </section>
  );
}
