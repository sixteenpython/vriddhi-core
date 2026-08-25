"""Small, durable JSON repository for the showcase deployment.

Secrets and engine state stay in the server-owned save directory. Public handlers
never serialize an envelope directly.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .errors import APIError


def _now() -> str:
    return datetime.now(UTC).isoformat()


class JSONRepository:
    def __init__(self, save_dir: str | Path) -> None:
        self.root = Path(save_dir)
        self.sessions = self.root / "sessions"
        self.campaigns = self.root / "campaigns"
        self.sessions.mkdir(parents=True, exist_ok=True)
        self.campaigns.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    @staticmethod
    def token_hash(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def create_session(self) -> tuple[str, dict[str, Any]]:
        token = secrets.token_urlsafe(32)
        owner_hash = self.token_hash(token)
        session = {
            "session_id": secrets.token_hex(8),
            "owner_hash": owner_hash,
            "created_at": _now(),
            "display_name": "Guest investor",
        }
        self._write(self.sessions / f"{owner_hash}.json", session)
        return token, {k: v for k, v in session.items() if k != "owner_hash"}

    def authenticate(self, token: str | None) -> str:
        if not token or len(token) < 20:
            raise APIError(401, "AUTH_REQUIRED", "A valid showcase bearer token is required.")
        owner_hash = self.token_hash(token)
        if not (self.sessions / f"{owner_hash}.json").is_file():
            raise APIError(401, "INVALID_SESSION", "This showcase session is not valid.")
        return owner_hash

    def create_campaign(self, owner_hash: str, game_json: str) -> dict[str, Any]:
        state = json.loads(game_json)
        campaign_id = state["campaign_id"]
        path = self.campaigns / f"{campaign_id}.json"
        with self._lock:
            if path.exists():
                raise APIError(409, "CAMPAIGN_EXISTS", "That campaign already exists.")
            envelope = {
                "owner_hash": owner_hash,
                "revision": 0,
                "created_at": _now(),
                "updated_at": _now(),
                "game_json": game_json,
                "idempotency": {},
            }
            self._write(path, envelope)
        return envelope

    def list_campaigns(self, owner_hash: str) -> list[dict[str, Any]]:
        result = []
        with self._lock:
            for path in self.campaigns.glob("*.json"):
                envelope = self._read(path)
                if envelope.get("owner_hash") == owner_hash:
                    result.append(envelope)
        return result

    def get_campaign(self, owner_hash: str, campaign_id: str) -> dict[str, Any]:
        if not campaign_id or any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-" for c in campaign_id):
            raise APIError(404, "CAMPAIGN_NOT_FOUND", "Campaign not found.")
        path = self.campaigns / f"{campaign_id}.json"
        with self._lock:
            if not path.is_file():
                raise APIError(404, "CAMPAIGN_NOT_FOUND", "Campaign not found.")
            envelope = self._read(path)
        if not secrets.compare_digest(str(envelope.get("owner_hash", "")), owner_hash):
            # Do not disclose that another owner has this identifier.
            raise APIError(404, "CAMPAIGN_NOT_FOUND", "Campaign not found.")
        return envelope

    def mutate_campaign(self, owner_hash: str, campaign_id: str, mutation) -> Any:
        path = self.campaigns / f"{campaign_id}.json"
        with self._lock:
            envelope = self.get_campaign(owner_hash, campaign_id)
            new_envelope, response = mutation(dict(envelope))
            new_envelope["revision"] = int(envelope["revision"]) + 1
            new_envelope["updated_at"] = _now()
            self._write(path, new_envelope)
        return response

    @staticmethod
    def _read(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _write(path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
