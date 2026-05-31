"""
Vriddhi - self-healing ticker resolver
=======================================

Keeps the universe alive when an NSE symbol changes (rename, demerger, exchange
move, delisting) so the monthly pipeline does NOT need manual CSV edits.

Why this exists: Yahoo occasionally retires a symbol. e.g. after Tata Motors'
demerger, "TATAMOTORS.NS" returns 404 and its 5-year history now lives on the
successor "TMPV.NS". Left alone, the stock silently rots with stale numbers.

How it heals (cheap by design - only the few BROKEN tickers do extra work):
  1. HARVEST: every healthy build caches each ticker's company longName in
     ticker_aliases.json. Names are what make search reliable - searching the
     bare NSE symbol returns junk (e.g. "RELIANCE" -> a US "RS" stock).
  2. RESOLVE (only when a download fails):
       a. try a previously-resolved override symbol,
       b. try the BSE listing (.BO),
       c. search Yahoo by the cached company name and, among NSE/BSE EQUITY
          hits, pick the one with the LONGEST usable price history - after a
          demerger the parent's track record usually survives on one successor.
  3. CACHE the resolved symbol + an audit entry, and write every non-clean
     ticker to research/universe_health.json for a human to glance at.

Nothing here is destructive: an unresolved ticker is just reported and left out
(Nifty 50 leaves plenty), and recommendations already exclude any stock without
fresh price history. So the worst case degrades gracefully instead of breaking.

Build-time only (used by build_grand_table.py / build_research_db.py).
"""

import json
import os
from datetime import datetime

import yfinance as yf

ALIAS_PATH = "ticker_aliases.json"
HEALTH_PATH = os.path.join("research", "universe_health.json")

# Yahoo exchange codes we accept as the "home" listing for an NSE stock.
NSE_BSE_EXCHANGES = {"NSI", "NSE", "BSE", "BO"}


# --------------------------------------------------------------------------- #
# Alias cache (ticker_aliases.json)
# --------------------------------------------------------------------------- #
def load_aliases(path=ALIAS_PATH):
    data = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            data = {}
    data.setdefault("names", {})    # base ticker -> company longName (search query)
    data.setdefault("symbols", {})  # base ticker -> resolved Yahoo symbol override
    data.setdefault("log", {})      # base ticker -> audit record of last resolution
    return data


def save_aliases(data, path=ALIAS_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def symbol_for(base, aliases):
    """Yahoo symbol to use for a base ticker: a resolved override if we have one,
    otherwise the default NSE symbol."""
    return aliases.get("symbols", {}).get(base) or f"{base.strip()}.NS"


def harvest_name(base, name, aliases):
    """Remember a healthy ticker's company name so we can search for its successor
    if it ever breaks."""
    if name and isinstance(name, str) and name.strip():
        aliases.setdefault("names", {})[base] = name.strip()


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #
def _usable_days(symbol, period="5y"):
    """Number of non-empty daily closes a symbol has (0 if it cannot be fetched)."""
    try:
        h = yf.download(symbol, period=period, auto_adjust=True, progress=False)
        if h is None or h.empty:
            return 0
        col = h["Close"] if "Close" in getattr(h, "columns", []) else h.iloc[:, 0]
        # When multiple columns come back (rare), squeeze to the first.
        if hasattr(col, "columns"):
            col = col.iloc[:, 0]
        return int(col.dropna().shape[0])
    except Exception:
        return 0


def _search_candidates(name):
    """NSE/BSE equity symbols Yahoo returns for a company-name query."""
    hits = []
    if not name:
        return hits
    try:
        res = yf.Search(name, max_results=10)
        for q in res.quotes:
            if q.get("quoteType") == "EQUITY" and q.get("exchange") in NSE_BSE_EXCHANGES:
                sym = q.get("symbol")
                if sym:
                    hits.append(sym)
    except Exception:
        pass
    return hits


def resolve_broken(base, aliases, min_days, period="5y"):
    """Find a working Yahoo symbol for a base ticker whose default download failed.

    Returns (symbol_or_None, status). Mutates the alias cache + audit log.
    status is one of: "renamed" (found a different symbol), "ok" (.NS worked on
    retry), or "unresolved".
    """
    default = f"{base.strip()}.NS"
    candidates, tried = [], []
    cached = aliases.get("symbols", {}).get(base)
    if cached:
        candidates.append(cached)
    candidates.append(f"{base.strip()}.BO")
    candidates += _search_candidates(aliases.get("names", {}).get(base))

    best, best_days = None, 0
    for sym in candidates:
        if not sym or sym in tried:
            continue
        tried.append(sym)
        days = _usable_days(sym, period)
        if days > best_days:
            best, best_days = sym, days

    now = datetime.now().strftime("%Y-%m-%d")
    if best and best_days >= min_days:
        aliases.setdefault("symbols", {})[base] = best
        status = "ok" if best == default else "renamed"
        aliases.setdefault("log", {})[base] = {
            "status": status, "resolved_to": best, "history_days": best_days,
            "tried": tried, "resolved_on": now,
        }
        return best, status

    aliases.setdefault("log", {})[base] = {
        "status": "unresolved", "tried": tried, "checked_on": now,
    }
    return None, "unresolved"


# --------------------------------------------------------------------------- #
# Health report
# --------------------------------------------------------------------------- #
def write_health_report(records, out_path=HEALTH_PATH):
    """records: list of {ticker, symbol, status, days}. Writes a summary JSON."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    summary = {}
    for r in records:
        summary[r["status"]] = summary.get(r["status"], 0) + 1
    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "universe_size": len(records),
        "summary": summary,
        "needs_review": [r for r in records if r["status"] in ("renamed", "unresolved", "stale")],
        "tickers": records,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload
