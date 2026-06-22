"""Hermes Next Status — CLI command for real-time system health overview.

Usage:
    hermes-next-status
    hermes-next-status --json
    hermes-next-status --db ~/.hermes-next/cache.db
    hermes-next-status --config ~/.hermes-next.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

from hermes_next.config import HermesNextConfig

logger = logging.getLogger(__name__)


# ── Data models ──────────────────────────────────────────


@dataclass
class DBStats:
    """Local cache database statistics."""

    path: str = ""
    size_mb: float = 0.0
    traces: int = 0
    embedded: int = 0
    policies: int = 0
    skills: int = 0
    concepts: int = 0
    triples: int = 0
    feedback: int = 0
    sessions: int = 0
    open_sessions: int = 0
    wal_size_mb: float = 0.0
    schema_ok: bool = False
    errors: list[str] = field(default_factory=list)


@dataclass
class OVHealth:
    """OpenViking server health."""

    reachable: bool = False
    url: str = ""
    error: str = ""


@dataclass
class GateStatus:
    """Promotion gate status."""

    gate_enabled: bool = False
    current_score: float = 0.0
    best_score: float = 0.0
    best_id: str = ""
    best_step: int = 0
    accept_count: int = 0
    reject_count: int = 0


@dataclass
class StatusReport:
    """Full system status report."""

    timestamp: str = ""
    version: str = "0.5.1"
    config_path: str = ""
    db: DBStats = field(default_factory=DBStats)
    ov: OVHealth = field(default_factory=OVHealth)
    gate: Optional[GateStatus] = None


# ── DB Query ─────────────────────────────────────────────


def _resolve_db_path(cfg_path: str, explicit_db: Optional[str]) -> str:
    """Resolve cache DB path from config or explicit --db."""
    if explicit_db:
        return str(Path(explicit_db).expanduser())

    # Try loading config
    try:
        cfg = HermesNextConfig.load(cfg_path or None)
        db_path = cfg.cache.path
        return str(Path(db_path).expanduser())
    except Exception:
        pass

    # Fallback to default
    return str(Path.home() / ".hermes-next" / "cache.db")


def _query_db(db_path: str) -> DBStats:
    """Query local cache.db for all counts and metadata."""
    stats = DBStats(path=db_path)

    p = Path(db_path)
    if not p.is_file():
        stats.errors.append(f"DB not found: {db_path}")
        return stats

    if p.stat().st_size == 0:
        stats.errors.append("DB file is empty")
        return stats

    stats.size_mb = round(p.stat().st_size / (1024 * 1024), 2)

    # Check WAL
    wal_path = p.with_suffix(".db-wal")
    if wal_path.is_file():
        stats.wal_size_mb = round(wal_path.stat().st_size / (1024 * 1024), 2)

    # Open DB read-only
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Check schema existence
        tables = {
            r["name"]
            for r in cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }

        expected = {"traces", "policies", "skills", "concepts", "triples", "feedback", "session_state"}
        missing = expected - tables
        if missing:
            stats.errors.append(f"Missing tables: {', '.join(sorted(missing))}")
        else:
            stats.schema_ok = True

        # Counts — only if tables exist
        if "traces" in tables:
            stats.traces = cursor.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
            stats.embedded = cursor.execute("SELECT COUNT(*) FROM traces WHERE embedding IS NOT NULL").fetchone()[0]
        if "policies" in tables:
            stats.policies = cursor.execute("SELECT COUNT(*) FROM policies").fetchone()[0]
        if "skills" in tables:
            stats.skills = cursor.execute("SELECT COUNT(*) FROM skills").fetchone()[0]
        if "concepts" in tables:
            stats.concepts = cursor.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]
        if "triples" in tables:
            stats.triples = cursor.execute("SELECT COUNT(*) FROM triples").fetchone()[0]
        if "feedback" in tables:
            stats.feedback = cursor.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        if "session_state" in tables:
            stats.sessions = cursor.execute("SELECT COUNT(*) FROM session_state").fetchone()[0]
            stats.open_sessions = cursor.execute(
                "SELECT COUNT(*) FROM session_state WHERE status='open'"
            ).fetchone()[0]

        conn.close()
    except sqlite3.Error as e:
        stats.errors.append(f"SQLite error: {e}")

    return stats


# ── OV Health ────────────────────────────────────────────


def _check_ov_health(ov_url: str) -> OVHealth:
    """Check if OpenViking server is reachable."""
    health = OVHealth(url=ov_url)
    try:
        resp = httpx.get(f"{ov_url.rstrip('/')}/health", timeout=5.0)
        health.reachable = resp.status_code < 500
        if not health.reachable:
            health.error = f"HTTP {resp.status_code}"
    except httpx.HTTPError as e:
        health.error = str(e)
    except Exception as e:
        health.error = f"Connection failed: {e}"
    return health


# ── Gate State ───────────────────────────────────────────


def _load_gate_status() -> Optional[GateStatus]:
    """Attempt to load promotion gate state."""
    try:
        from hermes_next.memos.promote_gate import gate_state_path, load_gate_state

        path = gate_state_path()
        if not path.is_file():
            return None

        state = load_gate_state()
        return GateStatus(
            gate_enabled=os.environ.get("HERMES_NEXT_GATE_ENABLED", "True") in ("1", "true", "True"),
            current_score=state.current_score,
            best_score=state.best_score,
            best_id=state.best_id,
            best_step=state.best_step,
            accept_count=state.accept_count,
            reject_count=state.reject_count,
        )
    except Exception as e:
        logger.debug("Could not load gate state: %s", e)
        return None


# ── Renderers ────────────────────────────────────────────


def _render_text(report: StatusReport) -> str:
    """Render status report as human-readable text."""
    lines = [
        "╔══════════════════════════════════════╗",
        "║     Hermes Next  System Status       ║",
        "╚══════════════════════════════════════╝",
        "",
        f"  Timestamp : {report.timestamp}",
        f"  Version   : {report.version}",
        f"  Config    : {report.config_path or '(default)'}",
        "",
    ]

    # ── OpenViking ──
    lines.append("── OpenViking ──────────────────────────")
    ov = report.ov
    if ov.reachable:
        lines.append("  Status  : [OK] Connected")
        lines.append(f"  URL     : {ov.url}")
    else:
        lines.append("  Status  : [ERR] Unreachable")
        if ov.error:
            lines.append(f"  Error   : {ov.error}")
        if ov.url:
            lines.append(f"  URL     : {ov.url}")
    lines.append("")

    # ── Cache DB ──
    db = report.db
    lines.append("── Local Cache ─────────────────────────")
    if db.errors:
        for err in db.errors:
            lines.append(f"  [!] {err}")
    lines.append(f"  Path     : {db.path}")
    lines.append(f"  Size     : {db.size_mb} MB")
    if db.wal_size_mb > 0:
        lines.append(f"  WAL      : {db.wal_size_mb} MB")
    lines.append(f"  Schema   : {'[OK] OK' if db.schema_ok else '[ERR] Incomplete'}")
    lines.append(f"  Traces   : {db.traces}")
    embedded = db.embedded
    pct = (embedded / db.traces * 100) if db.traces > 0 else 0
    warn = "  嵌入覆盖率低于 50%" if pct < 50 else ""
    lines.append(f"  嵌入覆盖率:  {embedded} / {db.traces} ({pct:.0f}%){warn}")
    lines.append(f"  Policies : {db.policies}")
    lines.append(f"  Skills   : {db.skills}")
    lines.append(f"  Concepts : {db.concepts}")
    lines.append(f"  Triples  : {db.triples}")
    lines.append(f"  Feedback : {db.feedback}")
    lines.append(f"  Sessions : {db.sessions} ({db.open_sessions} open)")
    lines.append("")

    # ── Promotion Gate ──
    if report.gate:
        g = report.gate
        lines.append("── Promotion Gate ──────────────────────")
        lines.append(f"  Enabled      : {'[OK]' if g.gate_enabled else '[ERR]'}")
        lines.append(f"  Current Score: {g.current_score:.3f}")
        lines.append(f"  Best Score   : {g.best_score:.3f}")
        lines.append(f"  Best ID      : {g.best_id or '-'}")
        lines.append(f"  Best Step    : {g.best_step}")
        lines.append(f"  Accept Count : {g.accept_count}")
        lines.append(f"  Reject Count : {g.reject_count}")
        lines.append("")

    lines.append("══════════════════════════════════════════")
    return "\n".join(lines)


def _render_json(report: StatusReport) -> str:
    """Render status report as JSON."""
    d = asdict(report)
    # Clean up None values
    if d.get("gate") is None:
        d.pop("gate", None)
    return json.dumps(d, indent=2, ensure_ascii=False)


# ── Main ─────────────────────────────────────────────────


def main() -> None:
    """Entry point for `hermes-next-status` CLI."""
    parser = argparse.ArgumentParser(
        description="Hermes Next — system health and statistics overview",
    )
    parser.add_argument(
        "--config",
        default="",
        help="Path to hermes-next.yaml config file",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Path to cache.db (overrides config)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of formatted text",
    )
    parser.add_argument(
        "--ov-url",
        default=None,
        help="OpenViking server URL (overrides config)",
    )

    args = parser.parse_args()

    # Build report
    report = StatusReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        config_path=args.config or "(default)",
    )

    # Resolve DB path and query
    db_path = _resolve_db_path(args.config, args.db)
    report.db = _query_db(db_path)

    # Resolve OV URL
    ov_url = args.ov_url or ""
    if not ov_url:
        try:
            cfg = HermesNextConfig.load(args.config or None)
            ov_url = cfg.openviking.base_url
        except Exception:
            ov_url = "http://localhost:1933"
    report.ov = _check_ov_health(ov_url)

    # Load gate state
    report.gate = _load_gate_status()

    # Output
    if args.json:
        print(_render_json(report))
    else:
        print(_render_text(report))

    # Exit code: non-zero if critical failures
    if not report.ov.reachable and report.db.errors:
        sys.exit(2)
    if not report.ov.reachable:
        sys.exit(1)
    if report.db.errors and not report.db.schema_ok:
        sys.exit(3)


if __name__ == "__main__":
    main()
