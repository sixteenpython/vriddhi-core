"""Contract tests for the immersive-web ASGI boundary (without an HTTP client dependency)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from bti.server.app import create_app


def request(app, method: str, path: str, body: Any = None,
            headers: dict[str, str] | None = None) -> tuple[int, dict[str, Any]]:
    raw = b"" if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
    sent = []
    delivered = False

    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": raw, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": method, "scheme": "http", "path": path, "raw_path": path.encode(),
        "query_string": b"", "root_path": "", "server": ("test", 80),
        "client": ("test", 123),
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
    }
    asyncio.run(app(scope, receive, send))
    start = next(item for item in sent if item["type"] == "http.response.start")
    payload = b"".join(item.get("body", b"") for item in sent if item["type"] == "http.response.body")
    return start["status"], json.loads(payload or b"{}")


def setup(tmp_path: Path):
    app = create_app(save_dir=tmp_path / "saves")
    status, session = request(app, "POST", "/api/v1/showcase/session")
    assert status == 201
    token = session["data"]["access_token"]
    auth = {"authorization": f"Bearer {token}", "content-type": "application/json"}
    return app, token, auth


def test_health_content_and_auth_boundary(tmp_path):
    app, _, auth = setup(tmp_path)
    status, health = request(app, "GET", "/api/v1/health")
    assert status == 200
    assert health["data"]["release"] == "0.11.0"
    assert health["data"]["storage"] == {
        "backend": "json",
        "durable": False,
        "healthy": True,
    }
    assert request(app, "GET", "/api/v1/lessons")[1]["data"]["lessons"]
    status, failure = request(app, "GET", "/api/v1/campaigns")
    assert status == 401
    assert failure["error"]["code"] == "AUTH_REQUIRED"
    assert request(app, "GET", "/api/v1/campaigns", headers=auth)[1]["data"] == []


def test_campaign_move_is_validated_committed_once_and_resumable(tmp_path):
    app, token, auth = setup(tmp_path)
    status, created = request(app, "POST", "/api/v1/campaigns",
                              {"monthly_amount_rupees": 10_000, "horizon_months": 24}, auth)
    assert status == 201
    campaign = created["data"]
    campaign_id = campaign["campaign_id"]
    assert "seed" not in json.dumps(campaign).lower()
    assert "reference" not in json.dumps(campaign).lower()

    market = request(app, "GET", f"/api/v1/campaigns/{campaign_id}/market", headers=auth)[1]["data"]
    assert any(stock["drawdown_pct"] > 0 for stock in market["stocks"])
    cheapest = min(market["stocks"], key=lambda stock: stock["close_paise"])
    shares = 1_000_000 // cheapest["close_paise"]
    move = {"expected_month": 0,
            "instructions": [{"side": "BUY", "ticker": cheapest["ticker"], "shares": shares}]}
    assert request(app, "POST", f"/api/v1/campaigns/{campaign_id}/moves/validate", move, auth)[0] == 200
    assert request(app, "GET", f"/api/v1/campaigns/{campaign_id}", headers=auth)[1]["data"]["moves_completed"] == 0

    commit_headers = {**auth, "idempotency-key": "move-month-0001"}
    first = request(app, "POST", f"/api/v1/campaigns/{campaign_id}/moves", move, commit_headers)
    second = request(app, "POST", f"/api/v1/campaigns/{campaign_id}/moves", move, commit_headers)
    assert first[0] == second[0] == 201
    assert first[1] == second[1]
    assert first[1]["data"]["campaign"]["moves_completed"] == 1
    assert request(app, "GET", f"/api/v1/campaigns/{campaign_id}/result", headers=auth)[0] == 200
    history = request(
        app, "GET", f"/api/v1/campaigns/{campaign_id}/history", headers=auth
    )
    assert history[0] == 200
    assert history[1]["data"]["gameplay_mode"] == "RATED"
    assert len(history[1]["data"]["moves"]) == 1
    review = request(
        app, "GET", f"/api/v1/campaigns/{campaign_id}/history/1", headers=auth
    )
    assert review[0] == 200
    assert review[1]["data"]["review_mode"] is True
    assert review[1]["data"]["performance_series"][0]["move"] == 1
    assert request(
        app, "GET", f"/api/v1/campaigns/{campaign_id}/history/2", headers=auth
    )[0] == 404

    # A fresh app instance resumes from durable storage using the same opaque token.
    resumed = create_app(save_dir=tmp_path / "saves")
    resumed_auth = {"authorization": f"Bearer {token}"}
    assert request(resumed, "GET", f"/api/v1/campaigns/{campaign_id}", headers=resumed_auth)[1]["data"]["moves_completed"] == 1
    stale_headers = {**auth, "idempotency-key": "different-key-0002"}
    stale = request(app, "POST", f"/api/v1/campaigns/{campaign_id}/moves", move, stale_headers)
    assert stale[0] == 409
    assert stale[1]["error"]["code"] == "STALE_MOVE"


def test_campaign_ownership_validation_and_resignation(tmp_path):
    app, _, owner_a = setup(tmp_path)
    created = request(app, "POST", "/api/v1/campaigns",
                      {"monthly_amount_rupees": 25_000, "horizon_months": 36}, owner_a)[1]["data"]
    _, _, owner_b = setup(tmp_path)
    status, failure = request(app, "GET", f"/api/v1/campaigns/{created['campaign_id']}", headers=owner_b)
    assert status == 404 and failure["error"]["code"] == "CAMPAIGN_NOT_FOUND"
    status, resigned = request(app, "POST", f"/api/v1/campaigns/{created['campaign_id']}/resign",
                               {}, owner_a)
    assert status == 200
    assert resigned["data"]["result"]["recorded_as"] == "LOSS"
    assert resigned["data"]["campaign"]["status"] == "ABORTED"


def test_multiple_campaigns_can_be_active_and_aborted_campaigns_remain_visible(tmp_path):
    app, _, auth = setup(tmp_path)
    first = request(app, "POST", "/api/v1/campaigns",
                    {"monthly_amount_rupees": 10_000, "horizon_months": 24}, auth)[1]["data"]
    second = request(app, "POST", "/api/v1/campaigns",
                     {"monthly_amount_rupees": 50_000, "horizon_months": 60}, auth)[1]["data"]
    assert first["initial_market"]["stocks"]
    status, aborted = request(
        app, "POST", f"/api/v1/campaigns/{first['campaign_id']}/abort", {}, auth
    )
    assert status == 200
    assert aborted["data"]["campaign"]["status"] == "ABORTED"
    campaigns = request(app, "GET", "/api/v1/campaigns", headers=auth)[1]["data"]
    assert {campaign["status"] for campaign in campaigns} == {"ACTIVE", "ABORTED"}
    assert {campaign["campaign_id"] for campaign in campaigns} == {
        first["campaign_id"], second["campaign_id"]
    }


def test_explicit_validation_errors(tmp_path):
    app, _, auth = setup(tmp_path)
    bad_json = request(app, "POST", "/api/v1/campaigns", b"{bad", auth)
    assert bad_json[0] == 400 and bad_json[1]["error"]["code"] == "INVALID_JSON"
    bad_horizon = request(app, "POST", "/api/v1/campaigns",
                          {"monthly_amount_rupees": 10_000, "horizon_months": 12}, auth)
    assert bad_horizon[0] == 422
    assert bad_horizon[1]["error"]["code"] == "GAME_RULE_VIOLATION"
