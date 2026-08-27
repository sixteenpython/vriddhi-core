"""Guardrails for the immersive client while it is built incrementally.

The tests deliberately run before the React or HTTP adapter exists. Once those files appear, the
conditional checks become active without weakening the current deterministic engine contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bti.game_engine import BTIGame

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "bti" / "docs"
WEB = ROOT / "bti" / "frontend"

PRIVATE_KEY_FRAGMENTS = {
    "coefficient",
    "efficient_frontier",
    "feature_weight",
    "future",
    "optimum",
    "optimizer",
    "reference",
    "scenario_seed",
    "target_weight",
}


def _keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            found.add(str(key).lower())
            found.update(_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            found.update(_keys(nested))
    return found


def test_immersive_blueprint_is_complete_and_unambiguous() -> None:
    architecture = (DOCS / "IMMERSIVE_WEB_ARCHITECTURE.md").read_text(encoding="utf-8")
    ux = (DOCS / "IMMERSIVE_WEB_UX.md").read_text(encoding="utf-8")
    release = (DOCS / "IMMERSIVE_WEB_RELEASE.md").read_text(encoding="utf-8")
    architecture = " ".join(architecture.split())
    ux = " ".join(ux.split())
    release = " ".join(release.split())

    for phrase in (
        "React + TypeScript",
        "single Python ASGI process",
        "/api/v1/campaigns/{id}/moves",
        "Idempotency-Key",
        "Public/private intelligence boundary",
    ):
        assert phrase in architecture
    for phrase in (
        "ORIENT -> INVESTIGATE -> CONSTRUCT -> COMMIT -> REVEAL -> LEARN -> CONTINUE",
        "Move analysis — Stockfish moment",
        "WCAG 2.2 AA",
        "Complete ten-minute acceptance test",
        "Decision quality and market luck",
        "SIMULATION MODE",
        "The stocks are real. The market you are about to play in is not.",
    ):
        assert phrase in ux
    for phrase in (
        "one Python service",
        "Streamlit is not part of the target runtime",
        "CI gates",
        "Investor-showcase release checklist",
        "Definition of done",
    ):
        assert phrase in release


def test_current_public_game_projections_do_not_leak_hidden_intelligence() -> None:
    game = BTIGame.create(
        monthly_amount_rupees=25_000,
        horizon_months=24,
        seed="immersive-contract",
        repository_root=ROOT,
        campaign_id="CONTRACTCHECK0001",
    )
    payload = {"campaign": game.public_state(), "market": game.market_view()}
    public_keys = _keys(payload)
    leaked = {
        key
        for key in public_keys
        if any(fragment in key for fragment in PRIVATE_KEY_FRAGMENTS)
    }
    assert not leaked, f"Public projection leaked server-private keys: {sorted(leaked)}"
    assert json.dumps(payload)
    assert payload["market"]["label"] == "SIMULATED MARKET"


def test_frontend_contract_activates_when_react_scaffold_is_added() -> None:
    """Do not require an unbuilt client, but reject a malformed scaffold the moment one appears."""
    package_file = WEB / "package.json"
    source_dir = WEB / "src"
    if not package_file.exists() and not source_dir.exists():
        return

    assert package_file.is_file(), "A React source tree requires bti/frontend/package.json"
    assert source_dir.is_dir(), "bti/frontend/package.json requires bti/frontend/src"
    package = json.loads(package_file.read_text(encoding="utf-8"))
    scripts = package.get("scripts", {})
    assert {"build", "test"} <= set(scripts)
    dependencies = {
        **package.get("dependencies", {}),
        **package.get("devDependencies", {}),
    }
    assert "react" in dependencies
    assert "typescript" in dependencies

    forbidden_runtime = "streamlit"
    for source in source_dir.rglob("*"):
        if source.suffix.lower() in {".ts", ".tsx", ".js", ".jsx"}:
            assert forbidden_runtime not in source.read_text(encoding="utf-8").lower()


def test_asgi_contract_activates_when_server_scaffold_is_added() -> None:
    server = ROOT / "bti" / "server"
    application = server / "app.py"
    if not application.exists():
        return

    assert application.is_file()
    source = application.read_text(encoding="utf-8").lower()
    assert "/api/v1" in source
    assert "dist" in source, "The Python service must serve the compiled React dist"
    assert "streamlit" not in source


def test_frontend_keeps_simulation_boundary_visible() -> None:
    app = (WEB / "src" / "App.tsx").read_text(encoding="utf-8")
    assert "SIMULATION MODE" in app
    assert "The stocks are real. The market you are about to play in is not." in app
    assert "not a live quote or investment recommendation" in app


def test_investor_preview_is_installable_and_mobile_first() -> None:
    manifest = json.loads(
        (WEB / "public" / "manifest.webmanifest").read_text(encoding="utf-8")
    )
    service_worker = (WEB / "public" / "sw.js").read_text(encoding="utf-8")
    app = (WEB / "src" / "App.tsx").read_text(encoding="utf-8")
    styles = (WEB / "src" / "styles.css").read_text(encoding="utf-8")

    assert manifest["name"] == "BTI — Beat the Index"
    assert manifest["display"] == "standalone"
    assert manifest["orientation"] == "portrait-primary"
    assert {192, 512} <= {
        int(icon["sizes"].split("x", 1)[0]) for icon in manifest["icons"]
    }
    assert "bti-release-candidate-v0.14.1" in service_worker
    assert "INSTALL BTI" in app
    assert 'railCollapsed ? ">>" : "<<"' in app
    assert "mobile-navigation-open" in styles
    assert "purpose-built installed mobile game" in styles


def test_research_and_market_analysis_surfaces_remain_first_class() -> None:
    terminal = (WEB / "src" / "Terminal.tsx").read_text(encoding="utf-8")
    cockpit = (WEB / "src" / "Cockpit.tsx").read_text(encoding="utf-8")
    research = (WEB / "src" / "StockResearch.tsx").read_text(encoding="utf-8")
    game_board = (WEB / "src" / "GameBoard.tsx").read_text(encoding="utf-8")

    for phrase in (
        "Minimum forecast",
        "Maximum PEG",
        "TOP GAINERS",
        "TOP LOSERS",
        "MONTHLY MOVERS",
    ):
        assert phrase in terminal or phrase in cockpit
    for phrase in (
        "SIMULATED OHLC",
        "FORECAST TERM STRUCTURE",
        "VALUATION · QUALITY · RISK MATRIX",
        "NEWS & CATALYST MONITOR",
        "SIMULATION MODE",
    ):
        assert phrase in research
    for phrase in (
        "RATED GAME BOARD",
        "THE CHASE · PORTFOLIO VS NIFTY",
        "IMMUTABLE MOVE HISTORY",
        "RETURN TO LIVE MOVE",
        "NO TAKEBACKS",
    ):
        assert phrase in game_board


def test_mobile_uses_progressive_disclosure_not_a_compressed_terminal() -> None:
    app = (WEB / "src" / "App.tsx").read_text(encoding="utf-8")
    market = (WEB / "src" / "MobileMarket.tsx").read_text(encoding="utf-8")
    game = (WEB / "src" / "MobileGameBoard.tsx").read_text(encoding="utf-8")
    styles = (WEB / "src" / "styles.css").read_text(encoding="utf-8")
    mobile_polish = (WEB / "src" / "mobile-polish.css").read_text(encoding="utf-8")
    main = (WEB / "src" / "main.tsx").read_text(encoding="utf-8")

    assert "useMobileViewport" in app
    assert "MobileMarket" in app and "MobileGameBoard" in app
    for phrase in (
        "VALUE",
        "QUALITY",
        "QUANT",
        "TECHNICAL",
        "NEWS",
        "MARKET PULSE",
        "LIVE MOVE",
        "PORTFOLIO",
        "REPEAT LAST MOVE",
    ):
        assert phrase in market
    for phrase in (
        "RATED CAMPAIGN",
        "EXECUTE PERMANENT MOVE",
        "WHAT WORKED",
        "NEXT EDGE",
        "MOVE HISTORY",
        "YOU BEAT THE INDEX",
    ):
        assert phrase in game
    assert ".mobile-tabbar" in styles
    assert 'import "./mobile-polish.css"' in main
    assert "height: auto !important" in mobile_polish
    assert ".mobile-match-card" in mobile_polish
    assert ".mobile-game-hero" in mobile_polish
    assert "font-size: 13px" in mobile_polish
    assert "niftyTarget" in game
    assert ".mobile-stock-feed" in styles
    assert ".mobile-move-sheet" in styles
    assert ".mobile-chase-card" in styles


def test_rapid_blitz_replay_and_newswire_are_full_decision_surfaces() -> None:
    replay = (WEB / "src" / "BlitzRun.tsx").read_text(encoding="utf-8")
    game = (WEB / "src" / "GameBoard.tsx").read_text(encoding="utf-8")
    mobile = (WEB / "src" / "MobileGameBoard.tsx").read_text(encoding="utf-8")
    news = (WEB / "src" / "Newswire.tsx").read_text(encoding="utf-8")
    completed = (WEB / "src" / "CompletedJourney.tsx").read_text(encoding="utf-8")
    styles = (WEB / "src" / "styles.css").read_text(encoding="utf-8")

    for phrase in (
        "PORTFOLIO NAV OHLC",
        "FUTURE MONTHS SEALED",
        "journey-candle",
        "MARKET REPLAY",
        "SIM MONTH",
    ):
        assert phrase in replay
    assert "Care to rebalance?" in game
    assert "Care to rebalance?" in mobile
    for phrase in ("← 1M", "↶ REPLAY", "OPEN FINAL RESULT"):
        assert phrase in replay
    assert "COMPLETED MARKET JOURNEY" in completed
    assert "market-run-terminal svg{width:100%" in styles
    assert "newswire-web-desk" in news
    for phrase in ("MARKET BREADTH", "NEWS SENTIMENT", "SIGNAL MATRIX", "EARNINGS", "QUANT"):
        assert phrase in news
