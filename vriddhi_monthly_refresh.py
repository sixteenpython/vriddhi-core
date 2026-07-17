#!/usr/bin/env python
"""Transactional monthly refresh for the Vriddhi research application.

The refresh is built in an isolated staging directory.  Published artifacts
are not touched until every data invariant and Streamlit smoke test passes.
Run ``python vriddhi_monthly_refresh.py --help`` for operator options.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from vriddhi_validation import ValidationError, artifact_hashes, validate_candidate

ROOT = Path(__file__).resolve().parent
RESEARCH_DIR = ROOT / "research"
DATA_CSV = ROOT / "grand_table_expanded.csv"
ALIAS_FILE = ROOT / "ticker_aliases.json"
BACKUP_DIR = ROOT / "backups"
STAGING_ROOT = ROOT / ".refresh_staging"
KEEP_BACKUPS = 6
HORIZONS = (1, 2, 3, 4, 5)

REQUIRED_PACKAGES = (
    "pandas", "numpy", "yfinance", "statsmodels", "scipy", "matplotlib", "streamlit"
)
REQUIRED_FILES = (
    "build_grand_table.py", "build_research_db.py", "grand_table_expanded.csv",
    "streamlit_app.py", "vriddhi_core.py", "ticker_aliases.json",
)
GENERATED_PATHS = [
    "grand_table_expanded.csv",
    "ticker_aliases.json",
    "research/benchmark.csv",
    "research/universe_health.json",
    *[f"research/portfolio_{h}y.json" for h in HORIZONS],
    *[f"research/portfolio_{h}y_prev.json" for h in HORIZONS],
    "research/validation_report.json",
    "research/refresh_summary.md",
]


def stamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def log(message: str) -> None:
    print(f"[{stamp()}] {message}", flush=True)


def banner(title: str) -> None:
    line = "=" * 76
    print(f"\n{line}\n  {title}\n{line}", flush=True)


def die(message: str, code: int = 1) -> None:
    raise SystemExit(f"\n[{stamp()}] ABORTED: {message}\n")


def run_step(name: str, command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    banner(name)
    log("$ " + " ".join(command))
    started = time.time()
    result = subprocess.run(command, cwd=cwd, env=env)
    elapsed = time.time() - started
    if result.returncode:
        die(f"step {name!r} failed (exit {result.returncode}) after {elapsed:.0f}s")
    log(f"OK - {name!r} finished in {elapsed:.0f}s")


def preflight() -> dict[str, str]:
    banner("Step 1/7 - Pre-flight checks")
    missing_files = [name for name in REQUIRED_FILES if not (ROOT / name).exists()]
    if missing_files:
        die(f"missing required files: {missing_files}")

    versions: dict[str, str] = {}
    missing_packages: list[str] = []
    for package in REQUIRED_PACKAGES:
        try:
            __import__(package)
            versions[package] = importlib.metadata.version(package)
        except (ImportError, importlib.metadata.PackageNotFoundError):
            missing_packages.append(package)
    if missing_packages:
        die(
            f"missing Python packages: {missing_packages}. Install the locked refresh "
            "environment with: uv sync --frozen --all-extras"
        )
    log(f"Python {sys.version.split()[0]} at {sys.executable}")
    log("Dependencies: " + ", ".join(f"{name}={version}" for name, version in versions.items()))
    return versions


def make_backup() -> Path:
    banner("Step 2/7 - Backup current published state")
    BACKUP_DIR.mkdir(exist_ok=True)
    destination = BACKUP_DIR / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    destination.mkdir()
    shutil.copy2(DATA_CSV, destination / DATA_CSV.name)
    if ALIAS_FILE.exists():
        shutil.copy2(ALIAS_FILE, destination / ALIAS_FILE.name)
    if RESEARCH_DIR.exists():
        shutil.copytree(RESEARCH_DIR, destination / "research")
    log(f"Backup created: {destination.relative_to(ROOT)}")
    return destination


def prune_backups() -> None:
    snapshots = sorted(path for path in BACKUP_DIR.iterdir() if path.is_dir())
    for old in snapshots[:-KEEP_BACKUPS]:
        shutil.rmtree(old)
        log(f"Pruned old backup after successful promotion: {old.name}")


def restore_backup(backup: Path) -> None:
    banner(f"Restoring {backup.name}")
    if not backup.is_dir() or backup.parent.resolve() != BACKUP_DIR.resolve():
        die(f"invalid backup path: {backup}")
    shutil.copy2(backup / DATA_CSV.name, DATA_CSV)
    alias = backup / ALIAS_FILE.name
    if alias.exists():
        shutil.copy2(alias, ALIAS_FILE)
    backup_research = backup / "research"
    if backup_research.exists():
        if RESEARCH_DIR.exists():
            shutil.rmtree(RESEARCH_DIR)
        shutil.copytree(backup_research, RESEARCH_DIR)
    log("Published artifacts restored. Review and commit them if a remote rollback is required.")


def create_stage() -> Path:
    STAGING_ROOT.mkdir(exist_ok=True)
    stage = STAGING_ROOT / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    stage.mkdir()
    shutil.copy2(DATA_CSV, stage / DATA_CSV.name)
    shutil.copy2(ALIAS_FILE, stage / ALIAS_FILE.name)
    if RESEARCH_DIR.exists():
        shutil.copytree(RESEARCH_DIR, stage / "research")
    else:
        (stage / "research").mkdir()
    baseline = stage / ".previous_baseline"
    baseline.mkdir()
    for horizon in HORIZONS:
        previous = stage / "research" / f"portfolio_{horizon}y_prev.json"
        if previous.exists():
            shutil.copy2(previous, baseline / previous.name)
    log(f"Candidate workspace: {stage.relative_to(ROOT)}")
    return stage


def preserve_genuine_previous_snapshots(stage: Path) -> None:
    """Keep the older snapshot when a refresh is retried on the same data date."""
    baseline = stage / ".previous_baseline"
    for horizon in HORIZONS:
        current = stage / "research" / f"portfolio_{horizon}y.json"
        previous = stage / "research" / f"portfolio_{horizon}y_prev.json"
        older = baseline / previous.name
        if not (current.exists() and previous.exists() and older.exists()):
            continue
        current_date = json.loads(current.read_text(encoding="utf-8")).get("data_through")
        previous_date = json.loads(previous.read_text(encoding="utf-8")).get("data_through")
        older_date = json.loads(older.read_text(encoding="utf-8")).get("data_through")
        if previous_date and current_date and previous_date >= current_date:
            if older_date and older_date < current_date:
                shutil.copy2(older, previous)
                log(f"{horizon}Y same-date retry: retained previous snapshot from {older_date}")


def build_candidate(stage: Path, *, skip_grand_table: bool, as_of: str | None) -> None:
    if skip_grand_table:
        banner("Step 3/7 - Refresh knowledge asset")
        log("Skipped (--skip-grand-table); copied the current asset into staging")
    else:
        command = [sys.executable, str(ROOT / "build_grand_table.py"), "--out", DATA_CSV.name]
        if as_of:
            command.extend(["--as-of", as_of])
        run_step("Step 3/7 - Build candidate knowledge asset", command, cwd=stage)

    command = [sys.executable, str(ROOT / "build_research_db.py")]
    if as_of:
        command.extend(["--asof", as_of])
    run_step("Step 4/7 - Build candidate research bundles", command, cwd=stage)
    preserve_genuine_previous_snapshots(stage)


def smoke_test(stage: Path) -> None:
    banner("Step 6/7 - Headless Streamlit smoke test (all horizons)")
    code = (
        "from streamlit.testing.v1 import AppTest\n"
        "for horizon in (1, 2, 3, 4, 5):\n"
        f"    app = AppTest.from_file({str(ROOT / 'streamlit_app.py')!r}, default_timeout=90).run()\n"
        "    assert not app.exception, f'initial render: {app.exception}'\n"
        "    app.sidebar.selectbox[0].set_value(horizon).run()\n"
        "    app.sidebar.button[0].click().run()\n"
        "    assert not app.exception, f'horizon={horizon}: {app.exception}'\n"
        "    print(f'  smoke OK horizon={horizon}')\n"
        "print('SMOKE PASSED')\n"
    )
    env = os.environ.copy()
    env["VRIDDHI_RESEARCH_DIR"] = str(stage / "research")
    env["STREAMLIT_SERVER_ENABLE_CORS"] = "true"
    env["STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION"] = "true"
    env["STREAMLIT_LOGGER_LEVEL"] = "error"
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env)
    if result.returncode:
        die("candidate app smoke test failed; published artifacts remain unchanged")
    log("OK - candidate app renders across all five horizons")


def git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=ROOT, text=True, capture_output=True, check=True
        )
        return result.stdout.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def write_release_metadata(
    stage: Path, report: dict, versions: dict[str, str], *, as_of: str | None
) -> None:
    report_path = stage / "research" / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    recommended = [
        item["horizon_years"] for item in report["horizons"] if item["recommended"]
    ]
    first = report["horizons"][0]
    summary = [
        "# Vriddhi monthly refresh summary",
        "",
        f"- Data through: **{report['data_through']}**",
        f"- Universe: **{report['universe_size']} stocks**",
        f"- Recommended horizons: **{recommended or 'none'}**",
        f"- Maximum portfolio turnover: **{report['max_turnover_pct']:.1f}%**",
        f"- Validation: **{report['status'].upper()}**",
        "",
        "## Horizon changes",
        "",
        "| Horizon | Recommended | Stocks | Turnover | Picks | Drops |",
        "|---:|:---:|---:|---:|---|---|",
    ]
    del first
    for item in report["horizons"]:
        summary.append(
            f"| {item['horizon_years']}Y | {'yes' if item['recommended'] else 'no'} | "
            f"{item['num_stocks']} | {item['turnover_pct'] or 0:.1f}% | "
            f"{', '.join(item['picks']) or '-'} | {', '.join(item['drops']) or '-'} |"
        )
    (stage / "research" / "refresh_summary.md").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )

    hash_paths = [path for path in GENERATED_PATHS if path != "research/refresh_summary.md"]
    manifest = {
        "schema_version": 1,
        "release_id": f"refresh-{report['data_through']}",
        "built_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "requested_as_of": as_of or "latest-complete-market-date",
        "data_through": report["data_through"],
        "source_commit": git_value("rev-parse", "HEAD"),
        "source_branch": git_value("branch", "--show-current"),
        "data_provider": "Yahoo Finance via yfinance",
        "benchmark": "Nifty 50 (^NSEI); benchmark methodology review tracked in docs/methodology.md",
        "methodology_version": "2.0",
        "python": sys.version.split()[0],
        "dependencies": versions,
        "validation_status": report["status"],
        "artifacts_sha256": artifact_hashes(stage, hash_paths),
    }
    (stage / "research" / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def validate_and_report(
    stage: Path, *, turnover_limit: float, allow_high_turnover: bool, as_of: str | None
) -> dict:
    banner("Step 5/7 - Validate candidate artifacts")
    try:
        report = validate_candidate(
            stage / DATA_CSV.name,
            stage / "research",
            turnover_limit=turnover_limit,
            allow_high_turnover=allow_high_turnover,
            expected_as_of=as_of,
        )
    except ValidationError as exc:
        die(f"candidate validation failed: {exc}")
    for item in report["horizons"]:
        verdict = "RECOMMENDED" if item["recommended"] else "not recommended"
        log(
            f"{item['horizon_years']}Y {verdict}; {item['num_stocks']} stocks; "
            f"turnover={item['turnover_pct'] or 0:.1f}%; "
            f"picks={item['picks'] or '-'}; drops={item['drops'] or '-'}"
        )
    log(f"Validation passed; data through {report['data_through']}")
    return report


def promote_candidate(stage: Path, backup: Path | None) -> None:
    banner("Step 7/7 - Promote validated candidate")
    sources = [stage / DATA_CSV.name, stage / ALIAS_FILE.name]
    sources.extend(path for path in (stage / "research").iterdir() if path.is_file())
    promoted: list[Path] = []
    try:
        for source in sources:
            destination = ROOT / source.relative_to(stage)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".promoting")
            shutil.copy2(source, temporary)
            os.replace(temporary, destination)
            promoted.append(destination)
    except Exception as exc:
        if backup:
            log(f"Promotion failed after {len(promoted)} files; restoring backup")
            restore_backup(backup)
        die(f"candidate promotion failed: {exc}")
    log(f"Promoted {len(promoted)} validated artifacts")


def git_publish(data_through: str) -> None:
    banner("Publish - commit and push validated artifacts")
    tracked = ["grand_table_expanded.csv", "ticker_aliases.json", "research"]
    try:
        subprocess.run(["git", "add", *tracked], cwd=ROOT, check=True)
        changed = subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=ROOT
        ).returncode
        if changed == 0:
            log("No generated changes to publish")
            return
        message = f"Monthly refresh ({data_through}): validated research release"
        subprocess.run(["git", "commit", "-m", message], cwd=ROOT, check=True)
        subprocess.run(["git", "push", "github", "HEAD:master"], cwd=ROOT, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        die(f"GitHub publication failed: {exc}")
    log("GitHub master updated; Streamlit deployment should now start")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transactional Vriddhi monthly refresh")
    parser.add_argument("--as-of", help="Maximum market-data date (YYYY-MM-DD)")
    parser.add_argument("--skip-grand-table", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--no-smoke", action="store_true")
    parser.add_argument("--candidate-only", action="store_true",
                        help="Validate in staging but do not replace published artifacts")
    parser.add_argument("--push", action="store_true",
                        help="Commit and push generated artifacts after promotion")
    parser.add_argument("--turnover-limit", type=float, default=30.0)
    parser.add_argument("--allow-high-turnover", action="store_true")
    parser.add_argument("--restore-last", action="store_true")
    parser.add_argument("--yes", "-y", action="store_true")
    return parser.parse_args()


def confirm(args: argparse.Namespace) -> None:
    if args.yes or args.candidate_only or not sys.stdin.isatty():
        return
    print("\nThis will build and validate a candidate before replacing published artifacts.")
    answer = input("Proceed with the monthly refresh? [y/N] ").strip().lower()
    if answer not in {"y", "yes"}:
        die("cancelled by user", code=0)


def main() -> None:
    args = parse_args()
    if args.restore_last:
        snapshots = sorted(path for path in BACKUP_DIR.glob("*") if path.is_dir())
        if not snapshots:
            die("no backup snapshots found")
        restore_backup(snapshots[-1])
        return
    if args.push and args.candidate_only:
        die("--push and --candidate-only cannot be combined")
    if args.as_of:
        try:
            datetime.strptime(args.as_of, "%Y-%m-%d")
        except ValueError:
            die("--as-of must use YYYY-MM-DD")

    started = time.time()
    banner(f"VRIDDHI TRANSACTIONAL REFRESH - {datetime.now():%Y-%m-%d %H:%M}")
    confirm(args)
    versions = preflight()
    backup = None if args.no_backup or args.candidate_only else make_backup()
    stage = create_stage()
    build_candidate(stage, skip_grand_table=args.skip_grand_table, as_of=args.as_of)
    report = validate_and_report(
        stage,
        turnover_limit=args.turnover_limit,
        allow_high_turnover=args.allow_high_turnover,
        as_of=args.as_of,
    )
    write_release_metadata(stage, report, versions, as_of=args.as_of)
    if not args.no_smoke:
        smoke_test(stage)
    if args.candidate_only:
        banner(f"CANDIDATE PASSED in {time.time() - started:.0f}s")
        log(f"Candidate retained for inspection: {stage}")
        return

    promote_candidate(stage, backup)
    if backup:
        prune_backups()
    shutil.rmtree(stage)
    if args.push:
        git_publish(report["data_through"])
    banner(f"DONE in {time.time() - started:.0f}s")
    log(f"Published locally: data through {report['data_through']}")
    if not args.push:
        log("Review the generated diff, then publish with git or rerun using --push")


if __name__ == "__main__":
    main()
