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
    backend = "json"

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


class PostgresRepository:
    """Durable PostgreSQL repository used by the investor-preview deployment.

    The complete campaign envelope remains an opaque engine-owned JSON document.
    PostgreSQL supplies durability, owner isolation and row-level locking without
    leaking game internals into HTTP handlers or duplicating engine schemas.
    """

    backend = "postgres"

    def __init__(self, database_url: str) -> None:
        if not database_url:
            raise ValueError("database_url is required")
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - deployment dependency guard
            raise RuntimeError("psycopg is required for PostgreSQL persistence") from exc
        self._psycopg = psycopg
        self.database_url = database_url
        self._lock = threading.RLock()
        self._initialise()

    @staticmethod
    def token_hash(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def _connect(self):
        return self._psycopg.connect(self.database_url, connect_timeout=10)

    def _initialise(self) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS bti_sessions (
                    owner_hash TEXT PRIMARY KEY,
                    session JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS bti_campaigns (
                    campaign_id TEXT PRIMARY KEY,
                    owner_hash TEXT NOT NULL REFERENCES bti_sessions(owner_hash)
                        ON DELETE CASCADE,
                    envelope JSONB NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 0,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS bti_campaign_owner_idx "
                "ON bti_campaigns(owner_hash, updated_at DESC)"
            )

    def healthcheck(self) -> bool:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            return cursor.fetchone() == (1,)

    def create_session(self) -> tuple[str, dict[str, Any]]:
        token = secrets.token_urlsafe(32)
        owner_hash = self.token_hash(token)
        session = {
            "session_id": secrets.token_hex(8),
            "owner_hash": owner_hash,
            "created_at": _now(),
            "display_name": "Guest investor",
        }
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO bti_sessions(owner_hash, session) VALUES (%s, %s::jsonb)",
                (owner_hash, json.dumps(session)),
            )
        return token, {key: value for key, value in session.items() if key != "owner_hash"}

    def authenticate(self, token: str | None) -> str:
        if not token or len(token) < 20:
            raise APIError(401, "AUTH_REQUIRED", "A valid showcase bearer token is required.")
        owner_hash = self.token_hash(token)
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT 1 FROM bti_sessions WHERE owner_hash = %s", (owner_hash,)
            )
            if cursor.fetchone() is None:
                raise APIError(401, "INVALID_SESSION", "This showcase session is not valid.")
        return owner_hash

    def create_campaign(self, owner_hash: str, game_json: str) -> dict[str, Any]:
        state = json.loads(game_json)
        campaign_id = state["campaign_id"]
        envelope = {
            "owner_hash": owner_hash,
            "revision": 0,
            "created_at": _now(),
            "updated_at": _now(),
            "game_json": game_json,
            "idempotency": {},
        }
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO bti_campaigns(
                        campaign_id, owner_hash, envelope, revision, created_at, updated_at
                    ) VALUES (%s, %s, %s::jsonb, 0, %s, %s)
                    """,
                    (
                        campaign_id,
                        owner_hash,
                        json.dumps(envelope),
                        envelope["created_at"],
                        envelope["updated_at"],
                    ),
                )
        except self._psycopg.errors.UniqueViolation as exc:
            raise APIError(409, "CAMPAIGN_EXISTS", "That campaign already exists.") from exc
        return envelope

    def list_campaigns(self, owner_hash: str) -> list[dict[str, Any]]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT envelope FROM bti_campaigns "
                "WHERE owner_hash = %s ORDER BY updated_at DESC",
                (owner_hash,),
            )
            return [self._document(row[0]) for row in cursor.fetchall()]

    def get_campaign(self, owner_hash: str, campaign_id: str) -> dict[str, Any]:
        self._validate_campaign_id(campaign_id)
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT owner_hash, envelope FROM bti_campaigns WHERE campaign_id = %s",
                (campaign_id,),
            )
            row = cursor.fetchone()
        if row is None or not secrets.compare_digest(str(row[0]), owner_hash):
            raise APIError(404, "CAMPAIGN_NOT_FOUND", "Campaign not found.")
        return self._document(row[1])

    def mutate_campaign(self, owner_hash: str, campaign_id: str, mutation) -> Any:
        self._validate_campaign_id(campaign_id)
        with self._lock, self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT owner_hash, envelope FROM bti_campaigns "
                "WHERE campaign_id = %s FOR UPDATE",
                (campaign_id,),
            )
            row = cursor.fetchone()
            if row is None or not secrets.compare_digest(str(row[0]), owner_hash):
                raise APIError(404, "CAMPAIGN_NOT_FOUND", "Campaign not found.")
            envelope = self._document(row[1])
            new_envelope, response = mutation(dict(envelope))
            new_envelope["revision"] = int(envelope["revision"]) + 1
            new_envelope["updated_at"] = _now()
            cursor.execute(
                """
                UPDATE bti_campaigns
                SET envelope = %s::jsonb, revision = %s, updated_at = %s
                WHERE campaign_id = %s
                """,
                (
                    json.dumps(new_envelope),
                    new_envelope["revision"],
                    new_envelope["updated_at"],
                    campaign_id,
                ),
            )
        return response

    @staticmethod
    def _validate_campaign_id(campaign_id: str) -> None:
        if not campaign_id or any(
            character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
            for character in campaign_id
        ):
            raise APIError(404, "CAMPAIGN_NOT_FOUND", "Campaign not found.")

    @staticmethod
    def _document(value: Any) -> dict[str, Any]:
        return value if isinstance(value, dict) else json.loads(value)
