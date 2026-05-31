#!/usr/bin/env python
"""
Vriddhi monthly refresh pipeline
================================

One command to refresh the entire Vriddhi back-end at month-end. Run this and
walk away - it rebuilds the "knowledge asset" and every research bundle the app
serves, and it sets up the month-over-month rebalance automatically.

    python vriddhi_monthly_refresh.py                 # full refresh (recommended)
    python vriddhi_monthly_refresh.py --push          # ... then commit + push to GitHub
    python vriddhi_monthly_refresh.py --skip-grand-table   # only rebuild research bundles
    python vriddhi_monthly_refresh.py --no-smoke      # skip the headless app test
    python vriddhi_monthly_refresh.py --yes           # don't ask for confirmation

What it does, in order:
    1. Pre-flight checks (required files + Python packages).
    2. Backup the current CSV + research/ into backups/<timestamp>/.
    3. build_grand_table.py  -> refresh grand_table_expanded.csv
       (yfinance fundamentals + damped-trend time-series forecasts).
    4. build_research_db.py  -> rotate last month's bundles into *_prev.json,
       then rebuild this month's portfolio_Ny.json + benchmark.csv.
    5. Validate every bundle and print the verdicts + the rebalance deltas
       (what changed vs last month) so you can eyeball it before the demo.
    6. (optional) Headless Streamlit smoke test across horizons.
    7. (optional, --push) git commit + push to GitHub to refresh the live beta.

Designed so that anyone - you or a junior - can run it on the last day of the
month and trust the output. It aborts loudly on any failure and never leaves
the live bundles half-written without telling you.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
RESEARCH_DIR = os.path.join(ROOT, "research")
DATA_CSV = os.path.join(ROOT, "grand_table_expanded.csv")
BACKUP_DIR = os.path.join(ROOT, "backups")
HORIZONS = [1, 2, 3, 4, 5]
KEEP_BACKUPS = 6

# Build-time packages each step needs. (streamlit is only for the smoke test.)
REQUIRED_PACKAGES = ["pandas", "numpy", "yfinance", "statsmodels", "scipy"]
REQUIRED_FILES = ["build_grand_table.py", "build_research_db.py", "grand_table_expanded.csv"]


# --------------------------------------------------------------------------- #
# Logging helpers
# --------------------------------------------------------------------------- #
def _stamp():
    return datetime.now().strftime("%H:%M:%S")


def log(msg):
    print(f"[{_stamp()}] {msg}", flush=True)


def banner(title):
    line = "=" * 70
    print(f"\n{line}\n  {title}\n{line}", flush=True)


def die(msg, code=1):
    print(f"\n[{_stamp()}] ABORTED: {msg}\n", flush=True)
    sys.exit(code)


def run_step(name, cmd):
    """Run a subprocess step, streaming output. Abort the pipeline on failure."""
    banner(name)
    log("$ " + " ".join(cmd))
    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    dt = time.time() - t0
    if result.returncode != 0:
        die(f"step '{name}' failed (exit {result.returncode}) after {dt:.0f}s.")
    log(f"OK - '{name}' finished in {dt:.0f}s.")


# --------------------------------------------------------------------------- #
# Pipeline stages
# --------------------------------------------------------------------------- #
def preflight():
    banner("Step 1/6 - Pre-flight checks")
    missing_files = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(ROOT, f))]
    if missing_files:
        die(f"missing required files: {missing_files} (run from the repo root).")
    log(f"Found required files: {', '.join(REQUIRED_FILES)}")

    missing_pkgs = []
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg)
        except ImportError:
            missing_pkgs.append(pkg)
    if missing_pkgs:
        die(f"missing Python packages: {missing_pkgs}. "
            f"Install with: pip install {' '.join(missing_pkgs)}")
    log(f"All build packages importable: {', '.join(REQUIRED_PACKAGES)}")


def backup_current():
    banner("Step 2/6 - Backup current state")
    os.makedirs(BACKUP_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    dest = os.path.join(BACKUP_DIR, stamp)
    os.makedirs(dest, exist_ok=True)

    if os.path.exists(DATA_CSV):
        shutil.copy2(DATA_CSV, os.path.join(dest, "grand_table_expanded.csv"))
    if os.path.isdir(RESEARCH_DIR):
        shutil.copytree(RESEARCH_DIR, os.path.join(dest, "research"), dirs_exist_ok=True)
    log(f"Backed up CSV + research/ -> {os.path.relpath(dest, ROOT)}")

    # Prune old backups, keep the most recent KEEP_BACKUPS.
    snaps = sorted(d for d in os.listdir(BACKUP_DIR)
                   if os.path.isdir(os.path.join(BACKUP_DIR, d)))
    for old in snaps[:-KEEP_BACKUPS]:
        shutil.rmtree(os.path.join(BACKUP_DIR, old), ignore_errors=True)
        log(f"Pruned old backup: {old}")
    return dest


def validate_and_report():
    banner("Step 5/6 - Validate bundles + rebalance report")
    benchmark = os.path.join(RESEARCH_DIR, "benchmark.csv")
    if not os.path.exists(benchmark):
        die("research/benchmark.csv was not produced - research build likely failed.")

    recommended = []
    for hy in HORIZONS:
        cur_path = os.path.join(RESEARCH_DIR, f"portfolio_{hy}y.json")
        if not os.path.exists(cur_path):
            die(f"missing {os.path.relpath(cur_path, ROOT)} - research build failed.")
        with open(cur_path, "r", encoding="utf-8") as f:
            cur = json.load(f)
        if cur.get("num_stocks", 0) <= 0 or not cur.get("stocks"):
            die(f"{hy}y bundle has no stocks - aborting before this goes live.")

        verdict = cur.get("verdict", {})
        rec = bool(verdict.get("recommended"))
        if rec:
            recommended.append(hy)
        through = cur.get("data_through", "?")
        flag = "RECOMMENDED" if rec else "not recommended"
        line = f"  {hy}yr | {flag:15s} | {cur['num_stocks']} stocks | data through {through}"

        # Rebalance delta vs last month, if a previous snapshot exists.
        prev_path = os.path.join(RESEARCH_DIR, f"portfolio_{hy}y_prev.json")
        if os.path.exists(prev_path):
            with open(prev_path, "r", encoding="utf-8") as f:
                prev = json.load(f)
            cur_w = {s["ticker"]: s["weight"] for s in cur["stocks"]}
            prev_w = {s["ticker"]: s["weight"] for s in prev["stocks"]}
            picks = [t for t in cur_w if t not in prev_w]
            drops = [t for t in prev_w if t not in cur_w]
            turnover = sum(abs(cur_w.get(t, 0) - prev_w.get(t, 0))
                           for t in set(cur_w) | set(prev_w)) / 2 * 100
            line += (f" | rebalance: +{len(picks)} pick / -{len(drops)} drop"
                     f" / turnover ~{turnover:.0f}%")
            if picks:
                line += f"  PICK={picks}"
            if drops:
                line += f"  DROP={drops}"
        else:
            line += " | rebalance: (first snapshot, no prior month)"
        print(line, flush=True)

    if recommended:
        log(f"Recommended horizons this month: {recommended}")
    else:
        log("WARNING: no horizon is currently recommended - the app will show "
            "'Not Recommended' for all. Review the gates before the demo.")
    return recommended


def smoke_test():
    banner("Step 6/6 - Headless app smoke test")
    try:
        import streamlit  # noqa: F401
    except ImportError:
        log("streamlit not installed - skipping smoke test (build outputs are still valid).")
        return
    code = (
        "from streamlit.testing.v1 import AppTest\n"
        "for h in (1, 4, 5):\n"
        "    at = AppTest.from_file('streamlit_app.py', default_timeout=60).run()\n"
        "    assert not at.exception, f'init h={h}: {at.exception}'\n"
        "    at.sidebar.selectbox[0].set_value(h).run()\n"
        "    at.sidebar.button[0].click().run()\n"
        "    assert not at.exception, f'gen h={h}: {at.exception}'\n"
        "    print(f'  smoke OK horizon={h}')\n"
        "print('SMOKE PASSED')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT)
    if result.returncode != 0:
        die("smoke test failed - the app errors on the fresh bundles. "
            "Outputs are written but DO NOT push until this is fixed.")
    log("OK - app renders cleanly across horizons on the fresh data.")


def git_push():
    banner("Optional - commit + push to GitHub (live beta refresh)")
    date_str = datetime.now().strftime("%Y-%m-%d")
    files = ["grand_table_expanded.csv", "research"]
    try:
        subprocess.run(["git", "add", *files], cwd=ROOT, check=True)
        msg = f"Monthly refresh ({date_str}): knowledge asset + research bundles"
        commit = subprocess.run(["git", "commit", "-m", msg], cwd=ROOT)
        if commit.returncode != 0:
            log("Nothing to commit (no changes) - skipping push.")
            return
        subprocess.run(["git", "push", "github", "master"], cwd=ROOT, check=True)
        log("Pushed to GitHub. The live beta will redeploy in a minute or two.")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        log(f"git step failed ({e}). Commit/push manually when ready:")
        log("  git add grand_table_expanded.csv research")
        log(f'  git commit -m "Monthly refresh ({date_str})"')
        log("  git push github master")


def confirm(auto_yes):
    if auto_yes or not sys.stdin.isatty():
        return
    print("\nThis will OVERWRITE the live knowledge asset (grand_table_expanded.csv)")
    print("and all research bundles. A backup is taken first.")
    ans = input("Proceed with the monthly refresh? [y/N] ").strip().lower()
    if ans not in ("y", "yes"):
        die("cancelled by user.", code=0)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Vriddhi monthly back-end refresh pipeline.")
    parser.add_argument("--skip-grand-table", action="store_true",
                        help="Skip regenerating the CSV; only rebuild research bundles.")
    parser.add_argument("--no-backup", action="store_true", help="Skip the backup step.")
    parser.add_argument("--no-smoke", action="store_true", help="Skip the headless app test.")
    parser.add_argument("--push", action="store_true",
                        help="Commit and push to GitHub after a successful refresh.")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Run without the interactive confirmation prompt.")
    args = parser.parse_args()

    t_start = time.time()
    banner(f"VRIDDHI MONTHLY REFRESH  -  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log(f"Repo: {ROOT}")
    log(f"Python: {sys.version.split()[0]} ({sys.executable})")

    confirm(args.yes)
    preflight()

    if not args.no_backup:
        backup_current()
    else:
        banner("Step 2/6 - Backup current state")
        log("Skipped (--no-backup).")

    if not args.skip_grand_table:
        run_step("Step 3/6 - Refresh knowledge asset (build_grand_table.py)",
                 [sys.executable, "build_grand_table.py"])
    else:
        banner("Step 3/6 - Refresh knowledge asset")
        log("Skipped (--skip-grand-table); reusing existing grand_table_expanded.csv.")

    run_step("Step 4/6 - Rebuild research bundles (build_research_db.py)",
             [sys.executable, "build_research_db.py"])

    validate_and_report()

    if not args.no_smoke:
        smoke_test()
    else:
        banner("Step 6/6 - Headless app smoke test")
        log("Skipped (--no-smoke).")

    if args.push:
        git_push()

    banner(f"DONE in {time.time() - t_start:.0f}s")
    if not args.push:
        log("Refresh complete. To publish to the live beta, run:")
        log("  git add grand_table_expanded.csv research")
        log('  git commit -m "Monthly refresh"')
        log("  git push github master")
        log("(or re-run this script with --push next time.)")


if __name__ == "__main__":
    main()
