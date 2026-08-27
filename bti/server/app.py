"""Starlette API and production static-site hook for the immersive BTI client."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from .errors import APIError
from .service import BTIService


def _ok(data: Any, status: int = 200) -> JSONResponse:
    return JSONResponse({"api_version": BTIService.API_VERSION, "data": data}, status_code=status)


def _bearer(request: Request) -> str | None:
    value = request.headers.get("authorization", "")
    scheme, _, token = value.partition(" ")
    return token if scheme.lower() == "bearer" and token else None


async def _json(request: Request) -> Any:
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > 1_000_000:
        raise APIError(413, "REQUEST_TOO_LARGE", "Request bodies are limited to 1 MB.")
    try:
        return await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise APIError(400, "INVALID_JSON", "Request body is not valid JSON.") from exc


def create_app(save_dir: str | Path | None = None,
               repository_root: str | Path | None = None) -> Starlette:
    root = Path(repository_root or Path(__file__).resolve().parents[2])
    service = BTIService(save_dir or os.getenv("BTI_SAVE_DIR", root / ".bti_saves"), root)

    async def health(_: Request) -> Response:
        return _ok({"status": "ok", "service": "bti-immersive-api",
                    "release": "0.12.1", "storage": service.storage_status()})

    async def session(_: Request) -> Response:
        return _ok(service.new_session(), 201)

    def owner(request: Request) -> str:
        return service.authenticate(_bearer(request))

    async def list_campaigns(request: Request) -> Response:
        return _ok(service.campaigns(owner(request)))

    async def profile(request: Request) -> Response:
        return _ok(service.profile(owner(request)))

    async def create_campaign(request: Request) -> Response:
        return _ok(service.create_campaign(owner(request), await _json(request)), 201)

    async def campaign(request: Request) -> Response:
        return _ok(service.state(owner(request), request.path_params["campaign_id"]))

    async def market(request: Request) -> Response:
        return _ok(service.market(owner(request), request.path_params["campaign_id"]))

    async def validate_move(request: Request) -> Response:
        return _ok(service.validate_move(owner(request), request.path_params["campaign_id"],
                                         await _json(request)))

    async def commit_move(request: Request) -> Response:
        result = service.commit_move(owner(request), request.path_params["campaign_id"],
                                     await _json(request), request.headers.get("idempotency-key"))
        return _ok(result, 201)

    async def result(request: Request) -> Response:
        return _ok(service.latest_result(owner(request), request.path_params["campaign_id"]))

    async def move_history(request: Request) -> Response:
        return _ok(service.move_history(owner(request), request.path_params["campaign_id"]))

    async def review_move(request: Request) -> Response:
        return _ok(
            service.review_move(
                owner(request),
                request.path_params["campaign_id"],
                request.path_params["move_number"],
            )
        )

    async def resign(request: Request) -> Response:
        return _ok(service.resign(owner(request), request.path_params["campaign_id"]))

    async def abort(request: Request) -> Response:
        return _ok(service.abort(owner(request), request.path_params["campaign_id"]))

    async def lessons(_: Request) -> Response:
        return _ok(service.content("lessons"))

    async def puzzles(_: Request) -> Response:
        return _ok(service.content("puzzles"))

    routes = [
        Route("/api/v1/health", health, methods=["GET"]),
        Route("/api/v1/showcase/session", session, methods=["POST"]),
        Route("/api/v1/campaigns", list_campaigns, methods=["GET"]),
        Route("/api/v1/profile", profile, methods=["GET"]),
        Route("/api/v1/campaigns", create_campaign, methods=["POST"]),
        Route("/api/v1/campaigns/{campaign_id:str}", campaign, methods=["GET"]),
        Route("/api/v1/campaigns/{campaign_id:str}/market", market, methods=["GET"]),
        Route("/api/v1/campaigns/{campaign_id:str}/moves/validate", validate_move, methods=["POST"]),
        Route("/api/v1/campaigns/{campaign_id:str}/moves", commit_move, methods=["POST"]),
        Route("/api/v1/campaigns/{campaign_id:str}/result", result, methods=["GET"]),
        Route("/api/v1/campaigns/{campaign_id:str}/history", move_history, methods=["GET"]),
        Route(
            "/api/v1/campaigns/{campaign_id:str}/history/{move_number:int}",
            review_move,
            methods=["GET"],
        ),
        Route("/api/v1/campaigns/{campaign_id:str}/resign", resign, methods=["POST"]),
        Route("/api/v1/campaigns/{campaign_id:str}/abort", abort, methods=["POST"]),
        Route("/api/v1/lessons", lessons, methods=["GET"]),
        Route("/api/v1/puzzles", puzzles, methods=["GET"]),
    ]
    frontend = root / "bti" / "frontend" / "dist"
    if frontend.is_dir():
        assets = frontend / "assets"
        if assets.is_dir():
            routes.append(Mount("/assets", StaticFiles(directory=assets), name="assets"))
        icons = frontend / "icons"
        if icons.is_dir():
            routes.append(Mount("/icons", StaticFiles(directory=icons), name="icons"))

        async def manifest(_: Request) -> Response:
            return FileResponse(
                frontend / "manifest.webmanifest",
                media_type="application/manifest+json",
                headers={"Cache-Control": "public, max-age=3600"},
            )

        async def service_worker(_: Request) -> Response:
            return FileResponse(
                frontend / "sw.js",
                media_type="application/javascript",
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Service-Worker-Allowed": "/",
                },
            )

        routes.extend([
            Route("/manifest.webmanifest", manifest),
            Route("/sw.js", service_worker),
        ])

        async def spa(_: Request) -> Response:
            # The HTML shell points at content-hashed assets and must be revalidated on
            # every release. Caching it can strand users on an older JavaScript bundle
            # even after Render has promoted the new container.
            return FileResponse(
                frontend / "index.html",
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0",
                },
            )

        async def missing_api(_: Request) -> Response:
            raise APIError(404, "ENDPOINT_NOT_FOUND", "API endpoint not found.")

        # Unknown API paths must remain JSON failures instead of falling into the SPA shell.
        routes.extend([Route("/api/{path:path}", missing_api), Route("/", spa),
                       Route("/{path:path}", spa)])

    async def api_error(_: Request, exc: APIError) -> JSONResponse:
        return JSONResponse({"api_version": BTIService.API_VERSION,
                             "error": {"code": exc.code, "message": exc.message}},
                            status_code=exc.status)

    async def unhandled(_: Request, __: Exception) -> JSONResponse:
        return JSONResponse({"api_version": BTIService.API_VERSION,
                             "error": {"code": "INTERNAL_ERROR",
                                       "message": "The request could not be completed."}}, status_code=500)

    app = Starlette(debug=False, routes=routes,
                    middleware=[Middleware(GZipMiddleware, minimum_size=1000)],
                    exception_handlers={APIError: api_error, Exception: unhandled})
    app.state.bti_service = service
    return app


app = create_app()
