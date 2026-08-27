import { useEffect, useState } from "react";
import type {
  Campaign,
  FinalResult,
  Market,
  MatchSummary,
  MoveResult,
  MoveReview,
  PerformancePoint,
  Trade,
} from "./api";
import { buildDraftPortfolio } from "./portfolioDraft";
import { initialLumpSumMinimum } from "./economics";
import { RapidClock } from "./RapidClock";
import { BlitzRun } from "./BlitzRun";

const rupees = (paise: number, compact = false) => {
  const value = paise / 100;
  if (compact && Math.abs(value) >= 1e7)
    return `₹${(value / 1e7).toFixed(2)}Cr`;
  if (compact && Math.abs(value) >= 1e5) return `₹${(value / 1e5).toFixed(1)}L`;
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(value);
};
const signed = (value: number, suffix = "%") =>
  `${value >= 0 ? "+" : ""}${value.toFixed(2)}${suffix}`;

function chartPath(
  values: number[],
  moves: number[],
  horizon: number,
  width = 720,
  height = 250,
  sharedMinimum?: number,
  sharedMaximum?: number,
) {
  if (!values.length) return "";
  const minimum = sharedMinimum ?? Math.min(...values);
  const maximum = sharedMaximum ?? Math.max(...values);
  const spread = maximum - minimum || 1;
  return values
    .map(
      (value, index) =>
        `${index ? "L" : "M"}${(moves[index] / horizon) * width},${height - 18 - ((value - minimum) / spread) * (height - 36)}`,
    )
    .join(" ");
}

function ChaseChart({
  series,
  horizon,
  monthlyAmountRupees,
  returnLabel,
}: {
  series: PerformancePoint[];
  horizon: number;
  monthlyAmountRupees: number;
  returnLabel: "SIP XIRR" | "CAGR";
}) {
  const portfolio = series.map((point) => point.portfolio_value_paise);
  const nifty = series.map((point) => point.benchmark_value_paise);
  const moves = series.map((point) => point.month || point.move);
  const latest = series.at(-1);
  const targetMoves = [...moves];
  const target = [...nifty];
  if (latest) {
    const monthlyRate =
      Math.pow(1 + latest.benchmark_projected_annual_return_pct / 100, 1 / 12) -
      1;
    let projected = latest.benchmark_value_paise;
    for (let move = (latest.month || latest.move) + 1; move <= horizon; move += 1) {
      projected = projected * (1 + monthlyRate) + monthlyAmountRupees * 100;
      targetMoves.push(move);
      target.push(projected);
    }
  }
  const combined = [...portfolio, ...target];
  const minimum = Math.min(...combined, 0);
  const maximum = Math.max(...combined, 1);
  const y = (value: number) =>
    232 - ((value - minimum) / (maximum - minimum || 1)) * 214;
  const gap = latest?.wealth_gap_paise || 0;
  const remaining = Math.max(0, horizon - (latest ? latest.month || latest.move : 0));
  const catchUp =
    gap < 0 && remaining ? Math.ceil(Math.abs(gap) / remaining) : 0;
  return (
    <section className="terminal-panel chase-card">
      <div className="panel-label">
        <span>THE CHASE · PORTFOLIO VS NIFTY</span>
        <small>ONE OVER = ONE COMMITTED MONTH</small>
      </div>
      <div className="chase-scoreboard">
        <div>
          <span>TARGET SET BY NIFTY</span>
          <b>{rupees(latest?.benchmark_value_paise || 0, true)}</b>
        </div>
        <div>
          <span>PLAYER SCORE</span>
          <b className={gap >= 0 ? "positive" : "gold"}>
            {rupees(latest?.portfolio_value_paise || 0, true)}
          </b>
        </div>
        <div>
          <span>{gap >= 0 ? "LEAD" : "RUNS TO CHASE"}</span>
          <b className={gap >= 0 ? "positive" : "negative"}>
            {rupees(Math.abs(gap), true)}
          </b>
        </div>
        <div>
          <span>OVERS REMAINING</span>
          <b>{remaining}</b>
        </div>
        <div>
          <span>REQUIRED CATCH-UP / MOVE</span>
          <b>{catchUp ? rupees(catchUp, true) : "IN CONTROL"}</b>
        </div>
      </div>
      {series.length ? (
        <div className="chase-chart">
          <div className="chase-legend">
            <span className="purple">┄ NIFTY · PROJECTED TARGET</span>
            <span className="green">━ PLAYER · COMPLETED OVERS</span>
          </div>
          <svg
            viewBox="0 0 720 250"
            preserveAspectRatio="none"
            aria-label="Portfolio chasing Nifty progress chart"
          >
            <path
              d={chartPath(
                target,
                targetMoves,
                horizon,
                720,
                250,
                minimum,
                maximum,
              )}
              className="chase-nifty-target"
            />
            <path
              d={chartPath(nifty, moves, horizon, 720, 250, minimum, maximum)}
              className="chase-nifty"
            />
            <path
              d={chartPath(
                portfolio,
                moves,
                horizon,
                720,
                250,
                minimum,
                maximum,
              )}
              className="chase-player"
            />
            {series.map((point, index) => {
              const x = ((point.month || point.move) / horizon) * 720;
              const playerY = y(portfolio[index]);
              const niftyY = y(nifty[index]);
              const showLabel =
                index === series.length - 1 || (point.month || point.move) % 12 === 0;
              return (
                <g key={`${point.move}-${point.month || point.move}`}>
                  <circle
                    cx={x}
                    cy={niftyY}
                    r="4"
                    className="chase-dot-nifty"
                  />
                  <circle
                    cx={x}
                    cy={playerY}
                    r="5"
                    className="chase-dot-player"
                  />
                  {showLabel && (
                    <text x={x} y="246" textAnchor="middle">
                      M{point.month || point.move}
                    </text>
                  )}
                </g>
              );
            })}
            <text x="710" y="18" textAnchor="end" className="target-label">
              NIFTY TARGET · M{horizon}
            </text>
          </svg>
        </div>
      ) : (
        <div className="chase-empty">
          Nifty is waiting to set the first target. Execute Move 1 to begin the
          chase.
        </div>
      )}
      {latest && (
        <div className="chase-rates">
          <span>
            PLAYER {returnLabel}{" "}
            <b>
              {returnLabel === "SIP XIRR" && latest.move < 3
                ? "FORMING"
                : signed(latest.portfolio_xirr_pct)}
            </b>
          </span>
          <span>
            NIFTY {returnLabel}{" "}
            <b>
              {returnLabel === "SIP XIRR" && latest.move < 3
                ? "FORMING"
                : signed(latest.benchmark_xirr_pct)}
            </b>
          </span>
          <span>
            PLAYER PROJECTED <b>{signed(latest.projected_annual_return_pct)}</b>
          </span>
          <span>
            NIFTY PROJECTED{" "}
            <b>{signed(latest.benchmark_projected_annual_return_pct)}</b>
          </span>
        </div>
      )}
    </section>
  );
}

function EvaluationBar({ result }: { result: MoveResult | null }) {
  const evaluation = result?.position_evaluation;
  const value = evaluation?.value || 0;
  const marker = Math.max(2, Math.min(98, 50 + value * 15));
  return (
    <div className="evaluation-card terminal-panel">
      <div className="panel-label">
        <span>POSITION EVALUATION</span>
        <small>PLAYER ↔ MARKET</small>
      </div>
      <div className="evaluation-score">
        <b className={value >= 0 ? "positive" : "negative"}>
          {evaluation?.display || "+0.00"}
        </b>
        <span>{evaluation?.label || "AWAITING COMMIT"}</span>
      </div>
      <div className="evaluation-track">
        <i style={{ left: `${marker}%` }} />
        <span>MARKET</span>
        <span>LEVEL</span>
        <span>PLAYER</span>
      </div>
      <p>
        A position score grades the resulting portfolio. It is separate from
        what happened next.
      </p>
    </div>
  );
}

function MatchScoreboard({ summary, returnLabel = "SIP XIRR" }: { summary: MatchSummary; returnLabel?: string }) {
  const forming = returnLabel === "SIP XIRR" && summary.move < 3;
  const gapTone = summary.wealth_gap_paise >= 0 ? "positive" : "negative";
  return (
    <section className="terminal-panel match-scoreboard">
      <div className="panel-label">
        <span>MATCH STATUS · AFTER MOVE {summary.move}</span>
        <small>{summary.position} · {summary.overs_remaining} MOVES REMAINING</small>
      </div>
      <div className="match-metrics">
        <div><span>TOTAL INVESTED</span><b>{rupees(summary.total_invested_paise, true)}</b></div>
        <div><span>PLAYER PORTFOLIO</span><b>{rupees(summary.portfolio_value_paise, true)}</b></div>
        <div><span>NIFTY PORTFOLIO</span><b>{rupees(summary.benchmark_value_paise, true)}</b></div>
        <div><span>WEALTH {summary.wealth_gap_paise >= 0 ? "LEAD" : "GAP"}</span><b className={gapTone}>{rupees(Math.abs(summary.wealth_gap_paise), true)}</b></div>
        <div><span>PLAYER {returnLabel}</span><b>{forming ? "FORMING" : signed(summary.portfolio_xirr_pct)}</b></div>
        <div><span>NIFTY {returnLabel}</span><b>{forming ? "FORMING" : signed(summary.benchmark_xirr_pct)}</b></div>
        <div><span>MAX DRAWDOWN</span><b className="negative">-{summary.max_drawdown_pct.toFixed(2)}%</b></div>
        <div><span>AVG MOVE QUALITY</span><b>{summary.average_move_score.toFixed(1)} / 100</b></div>
        <div><span>BTI RATING</span><b className="gold">{summary.rating}</b></div>
      </div>
    </section>
  );
}

function Endgame({ campaign, final }: { campaign: Campaign; final: FinalResult }) {
  const won = final.verdict === "BEAT_INDEX";
  const draw = final.verdict === "PHOTO_FINISH";
  const download = () => {
    const report = [
      `BTI CAMPAIGN ${campaign.campaign_id}`,
      final.headline,
      `${campaign.mode || "CLASSIC"} · ${final.months_completed}-month rated simulated campaign`,
      "",
      `Total invested: ${rupees(final.total_invested_paise)}`,
      `Final player portfolio: ${rupees(final.portfolio_value_paise)}`,
      `Nifty equivalent: ${rupees(final.benchmark_value_paise)}`,
      `Wealth lead: ${rupees(final.wealth_alpha_paise)}`,
      `Player ${final.return_label || "SIP XIRR"}: ${signed(final.portfolio_money_weighted_annual_return_pct)}`,
      `Nifty ${final.return_label || "SIP XIRR"}: ${signed(final.benchmark_money_weighted_annual_return_pct)}`,
      `${final.return_label || "Return"} advantage: ${signed(final.xirr_advantage_pct)}`,
      `Max drawdown: -${final.max_drawdown_pct.toFixed(2)}%`,
      `Average move quality: ${final.average_move_score.toFixed(1)} / 100`,
      `Final BTI rating: ${final.rating}`,
      "",
      `Process verdict: ${final.process_verdict}`,
      final.strategic_lesson,
      "",
      "SIMULATION MODE. This is an educational strategy game, not investment advice.",
    ].join("\n");
    const url = URL.createObjectURL(new Blob([report], { type: "text/plain" }));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `BTI-${campaign.campaign_id}-match-report.txt`;
    anchor.click();
    URL.revokeObjectURL(url);
  };
  return (
    <section className={`terminal-panel endgame-card ${won ? "won" : draw ? "draw" : "lost"}`}>
      <div className="endgame-kicker">{campaign.mode || "CLASSIC"} CAMPAIGN COMPLETE · {final.months_completed} MONTHS SIMULATED</div>
      <h1>{final.headline}</h1>
      <p className="endgame-deck">
        {won
          ? `You finished ${rupees(Math.abs(final.wealth_alpha_paise), true)} ahead of the simulated Nifty portfolio.`
          : draw
            ? "Neither side established a decisive wealth advantage. The match was settled within 0.5%."
            : `The simulated Nifty portfolio finished ${rupees(Math.abs(final.wealth_alpha_paise), true)} ahead.`}
      </p>
      <div className="endgame-scoreline">
        <div><span>PLAYER</span><b>{rupees(final.portfolio_value_paise, true)}</b><small>{signed(final.portfolio_money_weighted_annual_return_pct)} {final.return_label || "SIP XIRR"}</small></div>
        <strong>VS</strong>
        <div><span>NIFTY</span><b>{rupees(final.benchmark_value_paise, true)}</b><small>{signed(final.benchmark_money_weighted_annual_return_pct)} {final.return_label || "SIP XIRR"}</small></div>
      </div>
      <div className="endgame-grid">
        <div><span>TOTAL INVESTED</span><b>{rupees(final.total_invested_paise, true)}</b></div>
        <div><span>{final.return_label || "RETURN"} ADVANTAGE</span><b className={final.xirr_advantage_pct >= 0 ? "positive" : "negative"}>{signed(final.xirr_advantage_pct)}</b></div>
        <div><span>MAX DRAWDOWN</span><b className="negative">-{final.max_drawdown_pct.toFixed(2)}%</b></div>
        <div><span>MOVE QUALITY</span><b>{final.average_move_score.toFixed(1)} / 100</b></div>
        <div><span>FINAL RATING</span><b className="gold">{final.rating}</b></div>
      </div>
      <div className="endgame-process">
        <span>PROCESS VERDICT</span>
        <h2>{final.process_verdict}</h2>
        <p>{final.strategic_lesson}</p>
      </div>
      <div className="endgame-moves">
        {final.best_move && <div><span>BEST MOVE</span><b>M{final.best_move.move} · {final.best_move.classification} · {final.best_move.score}</b></div>}
        {final.weakest_move && <div><span>MOVE TO STUDY</span><b>M{final.weakest_move.move} · {final.weakest_move.classification} · {final.weakest_move.score}</b></div>}
      </div>
      <button className="primary" onClick={download}>DOWNLOAD MATCH REPORT ↓</button>
    </section>
  );
}

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

export function GameBoard({
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
  const displayedSeries =
    reviewData?.performance_series || campaign.performance_series;
  const draft = buildDraftPortfolio(campaign, market, trades);
  const rapidHold = !historical && !result && campaign.mode === "RAPID" && campaign.current_move > 1 && trades.length === 0;
  const preCommit = !historical && !result && (trades.length > 0 || rapidHold);
  const displayedResult = preCommit
    ? null
    : reviewData?.result || result || campaign.last_result;
  const readyToExecute = preCommit && draft.cashAfterPaise >= 0 && (
    campaign.mode === "CLASSIC"
      ? draft.deploymentPct >= 90
      : rapidHold ||
        draft.buyTotalPaise >= initialLumpSumMinimum(campaign.total_capital_rupees) ||
        campaign.current_move > 1
  );
  const execution = preCommit
    ? [
        ...trades.filter((trade) => trade.side === "SELL"),
        ...trades.filter((trade) => trade.side === "BUY"),
      ].map((trade) => {
        const stock = market.stocks.find(
          (item) => item.ticker === trade.ticker,
        )!;
        const amount = stock.close_paise * trade.shares;
        return {
          ...trade,
          price_paise: stock.close_paise,
          cash_movement_paise: trade.side === "SELL" ? amount : -amount,
        };
      })
    : displayedResult?.execution || [];
  let runningCash = preCommit
    ? campaign.cash_paise
    : (displayedResult?.portfolio_before?.cash_paise || 0) +
      campaign.monthly_amount_rupees * 100;
  const selectedMove =
    reviewData?.selected_move ||
    (preCommit ? campaign.current_move : displayedResult?.move) ||
    campaign.current_move;
  const canNextHistory = historical && selectedMove < campaign.moves_completed;
  const summary = reviewData?.match_summary || displayedResult?.match_summary || campaign.match_summary;
  const final = !historical ? displayedResult?.final_result || campaign.final_result : null;
  const activeMarketRun = Boolean(
    !historical &&
      result &&
      result.mode !== "CLASSIC" &&
      result.segment_series?.length &&
      !blitzRevealed,
  );
  useEffect(() => setBlitzRevealed(false), [result?.move]);
  return (
    <section className="terminal-page game-board-page">
      <div className="terminal-commandbar game-board-command">
        <div>
          <span className="live-dot" /> RATED GAME BOARD{" "}
          <b>BTI-MATCH/{campaign.campaign_id}</b>
        </div>
        <div className="terminal-clock">
          {historical
            ? `REVIEWING MOVE ${selectedMove}`
            : preCommit
              ? `FINAL REVIEW · MOVE ${campaign.current_move}`
              : `MOVE ${selectedMove} ANALYSIS`}
        </div>
        <div className="history-controls">
          <button
            disabled={selectedMove <= 1}
            onClick={() => reviewMove(selectedMove - 1)}
          >
            ← BACK
          </button>
          <button
            disabled={!canNextHistory}
            onClick={() => reviewMove(selectedMove + 1)}
          >
            FORWARD →
          </button>
          {historical && (
            <button className="live-return" onClick={returnLive}>
              RETURN TO LIVE MOVE
            </button>
          )}
        </div>
      </div>
      {historical && (
        <div className="review-mode-strip">
          REVIEW MODE · HISTORY IS IMMUTABLE · THE LIVE DRAFT IS PRESERVED
        </div>
      )}
      <RapidClock campaign={campaign} onExpire={execute} disabled={busy || historical || Boolean(result)} />
      {activeMarketRun && result && <BlitzRun result={result} onComplete={() => setBlitzRevealed(true)} />}
      {final && !activeMarketRun && <Endgame campaign={campaign} final={final} />}
      {!activeMarketRun && !historical && result?.mode === "RAPID" && campaign.status === "ACTIVE" && (
        <section className="rapid-stop-card">
          <div><span>RAPID STOP · MONTH {campaign.months_completed}</span><h2>Care to rebalance?</h2><p>The travelled path is now evidence. Inspect the OHLC replay, current Newswire and portfolio health before choosing HOLD or changing course.</p><RapidClock campaign={campaign} onExpire={execute} disabled={busy || historical} forceActive /></div>
          <button className="primary" onClick={continueGame}>OPEN THE NEXT DECISION WINDOW →</button>
        </section>
      )}
      {!activeMarketRun && <>
      <div className="game-board-grid">
        <main>
          <section className="terminal-panel execution-board">
            <div className="panel-label">
              <span>
                {preCommit
                  ? "FINAL EXECUTION SHEET"
                  : displayedResult?.notation || "NO COMPLETED MOVE"}
              </span>
              <small>
                {historical
                  ? "READ ONLY"
                  : preCommit
                    ? "PERMANENT AFTER EXECUTION"
                    : displayedResult?.classification}
              </small>
            </div>
            <div className="game-sheet-head">
              <span>#</span>
              <span>INSTRUCTION</span>
              <span>SIM PRICE</span>
              <span>CASH MOVEMENT</span>
              <span>CASH AFTER</span>
            </div>
            {execution.map((item, index) => {
              runningCash += item.cash_movement_paise;
              return (
                <div className="game-sheet-row" key={`${item.ticker}-${index}`}>
                  <b>{String(index + 1).padStart(2, "0")}</b>
                  <strong>
                    {item.side} {item.shares} {item.ticker}
                  </strong>
                  <span>{rupees(item.price_paise)}</span>
                  <span
                    className={
                      item.cash_movement_paise >= 0 ? "positive" : "negative"
                    }
                  >
                    {rupees(item.cash_movement_paise)}
                  </span>
                  <b>{rupees(runningCash)}</b>
                </div>
              );
            })}
            {!execution.length && (
              <div className="board-empty">
                {rapidHold ? "HOLD every current position and advance the simulation one year." : "Build a complete move in Market Monitor before entering the Game Board."}
              </div>
            )}
            {preCommit && (
              <div className="commit-bar">
                <button onClick={editMove}>← EDIT ON MARKET MONITOR</button>
                <div>
                  <span>DEPLOYMENT</span>
                  <b>{draft.deploymentPct.toFixed(1)}%</b>
                </div>
                <div>
                  <span>CASH AFTER</span>
                  <b>{rupees(draft.cashAfterPaise)}</b>
                </div>
                <button
                  className="primary"
                  disabled={busy || !readyToExecute}
                  onClick={execute}
                >
                  {busy
                    ? "EVALUATING…"
                    : draft.cashAfterPaise < 0
                      ? "ADJUST MOVE · CASH OVERDRAWN"
                    : campaign.mode === "CLASSIC" && draft.deploymentPct < 90
                        ? `ADJUST MOVE · DEPLOYMENT ${draft.deploymentPct.toFixed(1)}%`
                        : rapidHold ? "RECORD HOLD · ADVANCE ONE YEAR →" : campaign.mode === "BLITZ" ? "RUN FULL CAMPAIGN →" : "EXECUTE PERMANENT MOVE →"}
                </button>
              </div>
            )}
          </section>
          <ChaseChart
            series={displayedSeries}
            horizon={campaign.horizon_months}
            monthlyAmountRupees={campaign.monthly_amount_rupees}
            returnLabel={campaign.return_label}
          />
          {summary.move > 0 && <MatchScoreboard summary={summary} returnLabel={campaign.return_label} />}
        </main>
        <aside className="game-board-side">
          <EvaluationBar result={displayedResult || null} />
          {displayedResult ? (
            <section className="terminal-panel move-verdict">
              <div className="move-score">
                <span>MOVE QUALITY</span>
                <b>{displayedResult.score}</b>
                <small>/100</small>
              </div>
              <h2>{displayedResult.classification}</h2>
              <p>{displayedResult.decision_quality.headline}</p>
              <div className="outcome-triplet">
                <span>
                  PLAYER{" "}
                  <b>
                    {signed(
                      displayedResult.market_outcome.portfolio_return_pct,
                    )}
                  </b>
                </span>
                <span>
                  NIFTY{" "}
                  <b>
                    {signed(
                      displayedResult.market_outcome.benchmark_return_pct,
                    )}
                  </b>
                </span>
                <span>
                  ALPHA{" "}
                  <b
                    className={
                      displayedResult.market_outcome.alpha_pct >= 0
                        ? "positive"
                        : "negative"
                    }
                  >
                    {signed(displayedResult.market_outcome.alpha_pct)}
                  </b>
                </span>
              </div>
              <div className="game-regime-recap">
                <span>
                  MARKET COUNTERMOVE ·{" "}
                  {displayedResult.regime?.label || "SIMULATED REGIME"}
                </span>
                <p>
                  {displayedResult.regime?.narrative ||
                    "The market advanced independently of the submitted portfolio."}
                </p>
              </div>
              <div className="feedback-block good">
                <span>WHAT WORKED</span>
                {displayedResult.decision_quality.did_well.map((item) => (
                  <p key={item}>✓ {item}</p>
                ))}
              </div>
              <div className="feedback-block improve">
                <span>NEXT EDGE</span>
                {displayedResult.decision_quality.improve.map((item) => (
                  <p key={item}>↗ {item}</p>
                ))}
              </div>
              {!historical && result && campaign.status === "ACTIVE" && (
                <button className="primary full" onClick={continueGame}>
                  {campaign.mode === "RAPID" ? "CARE TO REBALANCE? →" : `CONTINUE TO MOVE ${campaign.current_move} →`}
                </button>
              )}
            </section>
          ) : (
            <section className="terminal-panel move-verdict awaiting">
              <span className="eyebrow">DECISION ENGINE</span>
              <h2>Score sealed until commit.</h2>
              <p>
                The hidden Vriddhi reference is evaluated only after your
                permanent move is submitted.
              </p>
            </section>
          )}
        </aside>
      </div>
      <section className="terminal-panel move-history-board">
        <div className="panel-label">
          <span>IMMUTABLE MOVE HISTORY</span>
          <small>PRESS BACK OR SELECT ANY COMPLETED MOVE</small>
        </div>
        <div className="history-timeline">
          {campaign.move_history.map((item) => (
            <button
              className={
                selectedMove === item.move && displayedResult ? "active" : ""
              }
              key={item.move}
              onClick={() => reviewMove(item.move)}
            >
              <span>M{String(item.move).padStart(2, "0")}</span>
              <b>{item.classification}</b>
              <em>{item.score}</em>
              <strong
                className={
                  item.position_evaluation.value >= 0 ? "positive" : "negative"
                }
              >
                {item.position_evaluation.display}
              </strong>
              <small>{signed(item.alpha_pct)}</small>
            </button>
          ))}
          {!campaign.move_history.length && (
            <p>
              No completed moves yet. Your rated record begins after the first
              execution.
            </p>
          )}
        </div>
      </section>
      </>}
      <div className="terminal-statusbar">
        <span>RATED CAMPAIGN · NO TAKEBACKS</span>
        <span>REVIEW RECONSTRUCTS HISTORY; IT NEVER CHANGES IT</span>
        <span>SIMULATION MODE · NO REAL ORDERS</span>
      </div>
    </section>
  );
}
