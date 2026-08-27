"""Safe application service over the deterministic BTI game engine."""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path
from typing import Any

from bti.game_engine import BTIGame
from bti.game_engine.engine import GameRuleError

from .errors import APIError
from .repository import JSONRepository, PostgresRepository


class BTIService:
    API_VERSION = "1.0"

    def __init__(self, save_dir: str | Path, repository_root: str | Path | None = None,
                 content_dir: str | Path | None = None) -> None:
        database_url = os.getenv("DATABASE_URL", "").strip()
        self.repo = (
            PostgresRepository(database_url)
            if database_url
            else JSONRepository(save_dir)
        )
        self.repository_root = Path(repository_root) if repository_root else None
        self.content_dir = Path(content_dir or Path(__file__).resolve().parents[1] / "content")

    def new_session(self) -> dict[str, Any]:
        token, profile = self.repo.create_session()
        return {"access_token": token, "token_type": "Bearer", "profile": profile}

    def authenticate(self, token: str | None) -> str:
        return self.repo.authenticate(token)

    def storage_status(self) -> dict[str, Any]:
        healthy = self.repo.healthcheck() if hasattr(self.repo, "healthcheck") else True
        return {"backend": self.repo.backend, "durable": self.repo.backend == "postgres", "healthy": healthy}

    def create_campaign(self, owner: str, payload: Any) -> dict[str, Any]:
        body = self._object(payload)
        mode = str(body.get("mode", "CLASSIC")).upper()
        amount, horizon = body.get("monthly_amount_rupees", 0), body.get("horizon_months")
        total_capital = body.get("total_capital_rupees")
        if mode == "CLASSIC" and (isinstance(amount, bool) or not isinstance(amount, int)):
            raise APIError(422, "INVALID_MONTHLY_AMOUNT", "monthly_amount_rupees must be a whole number.")
        if mode != "CLASSIC" and (isinstance(total_capital, bool) or not isinstance(total_capital, int)):
            raise APIError(422, "INVALID_TOTAL_CAPITAL", "total_capital_rupees must be a whole number.")
        if isinstance(horizon, bool) or not isinstance(horizon, int):
            raise APIError(422, "INVALID_HORIZON", "horizon_months must be 24, 36, 48 or 60.")
        try:
            game = BTIGame.create(amount, horizon, seed=secrets.token_urlsafe(24),
                                  repository_root=self.repository_root,
                                  campaign_id=secrets.token_hex(6).upper(),
                                  mode=mode, total_capital_rupees=total_capital)
        except GameRuleError as exc:
            raise APIError(422, "GAME_RULE_VIOLATION", str(exc)) from exc
        current_rating = self.profile(owner)["rating"]
        game.state["starting_rating"] = current_rating
        game.state["rating"] = current_rating
        self.repo.create_campaign(owner, game.to_json())
        return {**self._campaign(game), "initial_market": game.market_view()}

    def campaigns(self, owner: str) -> list[dict[str, Any]]:
        games = [self._load(item) for item in self.repo.list_campaigns(owner)]
        return [self._campaign(game) for game in sorted(games, key=lambda g: g.state["campaign_id"])]

    def profile(self, owner: str) -> dict[str, Any]:
        games = [self._load(item) for item in self.repo.list_campaigns(owner)]
        records: dict[str, dict[str, int]] = {
            mode: {"wins": 0, "losses": 0, "draws": 0, "completed": 0}
            for mode in ("CLASSIC", "RAPID", "BLITZ")
        }
        rating_delta = 0.0
        best_alpha = -999.0
        streak = 0
        for game in games:
            mode = game.state.get("mode", "CLASSIC")
            if game.status == "COMPLETED":
                result = game.final_result()
                records[mode]["completed"] += 1
                if result["verdict"] == "BEAT_INDEX":
                    records[mode]["wins"] += 1
                    streak += 1
                    outcome = 18
                elif result["verdict"] == "PHOTO_FINISH":
                    records[mode]["draws"] += 1
                    streak = 0
                    outcome = 2
                else:
                    records[mode]["losses"] += 1
                    streak = 0
                    outcome = -16
                weight = {"CLASSIC": 1.0, "RAPID": 0.7, "BLITZ": 0.35}[mode]
                process = (result["average_move_score"] - 70) * 0.35
                rating_delta += (outcome + process) * weight
                best_alpha = max(best_alpha, float(result["wealth_alpha_pct"]))
            elif game.status == "ABORTED":
                records[mode]["losses"] += 1
                streak = 0
                rating_delta -= {"CLASSIC": 18, "RAPID": 13, "BLITZ": 7}[mode]
        wins = sum(item["wins"] for item in records.values())
        losses = sum(item["losses"] for item in records.values())
        draws = sum(item["draws"] for item in records.values())
        completed = sum(item["completed"] for item in records.values())
        return {
            "rating": max(600, round(1200 + rating_delta)),
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "completed": completed,
            "beat_index_pct": round(wins / completed * 100, 1) if completed else 0.0,
            "best_alpha_pct": best_alpha if best_alpha > -900 else 0.0,
            "current_streak": streak,
            "by_mode": records,
            "rating_model": "ONE BTI RATING · MODE-NORMALISED EVIDENCE",
        }

    def state(self, owner: str, campaign_id: str) -> dict[str, Any]:
        return self._campaign(self._load(self.repo.get_campaign(owner, campaign_id)))

    def market(self, owner: str, campaign_id: str) -> dict[str, Any]:
        return self._load(self.repo.get_campaign(owner, campaign_id)).market_view()

    def validate_move(self, owner: str, campaign_id: str, payload: Any) -> dict[str, Any]:
        body = self._move_body(payload)
        game = self._load(self.repo.get_campaign(owner, campaign_id))
        clone = BTIGame.from_json(game.to_json(), repository_root=self.repository_root)
        try:
            result = clone.submit_move(body["instructions"], body["expected_month"])
        except GameRuleError as exc:
            raise APIError(422, "INVALID_MOVE", str(exc)) from exc
        return {"valid": True, "expected_month": body["expected_month"],
                "execution": result["execution"], "cash_remaining_paise": clone.state["cash_paise"],
                "notice": "Validation does not reserve prices or commit this move."}

    def commit_move(self, owner: str, campaign_id: str, payload: Any,
                    idempotency_key: str | None) -> dict[str, Any]:
        body = self._move_body(payload)
        if not idempotency_key or not 8 <= len(idempotency_key) <= 128:
            raise APIError(400, "IDEMPOTENCY_KEY_REQUIRED", "Provide an Idempotency-Key of 8–128 characters.")

        def mutation(envelope: dict[str, Any]):
            cached = envelope.get("idempotency", {}).get(idempotency_key)
            if cached is not None:
                return envelope, cached
            game = self._load(envelope)
            try:
                result = game.submit_move(body["instructions"], body["expected_month"])
            except GameRuleError as exc:
                stale = "stale" in str(exc).lower()
                raise APIError(409 if stale else 422, "STALE_MOVE" if stale else "INVALID_MOVE", str(exc)) from exc
            response = {"campaign": self._campaign(game), "result": result}
            envelope["game_json"] = game.to_json()
            entries = dict(envelope.get("idempotency", {}))
            entries[idempotency_key] = response
            envelope["idempotency"] = dict(list(entries.items())[-100:])
            return envelope, response

        return self.repo.mutate_campaign(owner, campaign_id, mutation)

    def latest_result(self, owner: str, campaign_id: str) -> dict[str, Any]:
        game = self._load(self.repo.get_campaign(owner, campaign_id))
        if not game.state["moves"]:
            raise APIError(404, "RESULT_NOT_AVAILABLE", "Submit the first move to create a result.")
        return game.state["moves"][-1]

    def move_history(self, owner: str, campaign_id: str) -> dict[str, Any]:
        game = self._load(self.repo.get_campaign(owner, campaign_id))
        return {
            "gameplay_mode": "RATED",
            "moves": game.move_history(),
            "performance_series": game.performance_series(),
        }

    def review_move(self, owner: str, campaign_id: str, move_number: int) -> dict[str, Any]:
        game = self._load(self.repo.get_campaign(owner, campaign_id))
        try:
            return game.review_move(move_number)
        except GameRuleError as exc:
            raise APIError(404, "MOVE_NOT_AVAILABLE", str(exc)) from exc

    def abort(self, owner: str, campaign_id: str) -> dict[str, Any]:
        def mutation(envelope: dict[str, Any]):
            game = self._load(envelope)
            try:
                result = game.abort()
            except GameRuleError as exc:
                raise APIError(409, "CAMPAIGN_NOT_ACTIVE", str(exc)) from exc
            envelope["game_json"] = game.to_json()
            return envelope, {"campaign": self._campaign(game), "result": result}
        return self.repo.mutate_campaign(owner, campaign_id, mutation)

    def resign(self, owner: str, campaign_id: str) -> dict[str, Any]:
        return self.abort(owner, campaign_id)

    def content(self, kind: str) -> dict[str, Any]:
        if kind not in {"lessons", "puzzles"}:
            raise APIError(404, "CONTENT_NOT_FOUND", "Content not found.")
        return json.loads((self.content_dir / f"{kind}.json").read_text(encoding="utf-8"))

    def _load(self, envelope: dict[str, Any]) -> BTIGame:
        try:
            return BTIGame.from_json(envelope["game_json"], repository_root=self.repository_root)
        except (GameRuleError, KeyError, json.JSONDecodeError) as exc:
            raise APIError(409, "CAMPAIGN_UNAVAILABLE", "The saved campaign cannot be loaded by this engine release.") from exc

    @staticmethod
    def _object(payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise APIError(400, "INVALID_JSON", "The request body must be a JSON object.")
        return payload

    def _move_body(self, payload: Any) -> dict[str, Any]:
        body = self._object(payload)
        expected, instructions = body.get("expected_month"), body.get("instructions")
        if isinstance(expected, bool) or not isinstance(expected, int) or expected < 0:
            raise APIError(422, "INVALID_EXPECTED_MONTH", "expected_month must be a non-negative whole number.")
        if not isinstance(instructions, list) or len(instructions) > 100:
            raise APIError(422, "INVALID_INSTRUCTIONS", "instructions must be a list of at most 100 trades.")
        if any(not isinstance(item, dict) for item in instructions):
            raise APIError(422, "INVALID_INSTRUCTIONS", "Each trade instruction must be an object.")
        return {"expected_month": expected, "instructions": instructions}

    @staticmethod
    def _campaign(game: BTIGame) -> dict[str, Any]:
        return game.public_state()
