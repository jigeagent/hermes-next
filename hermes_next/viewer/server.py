"""Hermes Next Viewer — stdlib-only HTTP server + SPA frontend.

Usage:
    python -m hermes_next.viewer [--port 8080] [--db ~/.hermes-next/cache.db]
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

from hermes_next.cache.connection import CacheConnection
from hermes_next.cache.schema import ensure_schema

logger = logging.getLogger(__name__)

# ── API Handler ───────────────────────────────────────────


class _APIHandler(BaseHTTPRequestHandler):
    """HTTP request handler serving the SPA + JSON API."""

    _cache: Optional[CacheConnection] = None
    _ov_url: str = ""

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.debug(fmt, *args)

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str, status: int = 200) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, msg: str, status: int = 400) -> None:
        self._send_json({"error": msg}, status)

    def _get_db(self) -> sqlite3.Connection:
        if self._cache is None:
            raise RuntimeError("Cache not initialized")
        return self._cache.conn

    # ── Routing ───────────────────────────────────────────

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        params = parse_qs(parsed.query)

        try:
            if path == "" or path == "/" or path == "/index.html":
                self._send_html(_SPA_HTML)
            elif path == "/api/stats":
                self._handle_stats()
            elif path == "/api/traces":
                self._handle_traces(params)
            elif path.startswith("/api/traces/"):
                trace_id = path.split("/api/traces/")[1]
                self._handle_trace_detail(trace_id)
            elif path == "/api/policies":
                self._handle_policies()
            elif path == "/api/skills":
                self._handle_skills()
            elif path == "/api/concepts":
                self._handle_concepts()
            elif path == "/api/triples":
                self._handle_triples()
            elif path == "/api/pipeline":
                self._handle_pipeline()
            elif path == "/api/timeline":
                self._handle_timeline(params)
            elif path == "/api/search":
                self._handle_search(params)
            elif path == "/api/health":
                self._send_json({"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()})
            else:
                self._send_error("Not found", 404)
        except Exception as e:
            logger.exception("API error: %s", e)
            self._send_error(str(e), 500)

    # ── API Handlers ──────────────────────────────────────

    def _handle_stats(self) -> None:
        db = self._get_db()
        trace_count = db.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
        policy_count = db.execute("SELECT COUNT(*) FROM policies").fetchone()[0]
        skill_count = db.execute("SELECT COUNT(*) FROM skills").fetchone()[0]
        concept_count = db.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]
        triple_count = db.execute("SELECT COUNT(*) FROM triples").fetchone()[0]
        synced_count = db.execute("SELECT COUNT(*) FROM traces WHERE synced=1").fetchone()[0]
        self._send_json({
            "traces": trace_count,
            "policies": policy_count,
            "skills": skill_count,
            "concepts": concept_count,
            "triples": triple_count,
            "synced": synced_count,
            "ov_url": self._ov_url,
        })

    def _handle_traces(self, params: dict[str, list[str]]) -> None:
        db = self._get_db()
        limit = min(int(params.get("limit", ["50"])[0]), 200)
        offset = int(params.get("offset", ["0"])[0])
        search = params.get("search", [None])[0]

        if search:
            safe = search.replace('"', '""')
            rows = db.execute(
                """
                SELECT t.* FROM traces t
                JOIN traces_fts fts ON t.rowid = fts.rowid
                WHERE traces_fts MATCH ?
                ORDER BY rank
                LIMIT ? OFFSET ?
                """, (safe, limit, offset),
            ).fetchall()
        else:
            rows = db.execute(
                "SELECT * FROM traces ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()

        traces = [dict(r) for r in rows]
        for t in traces:
            self._serialize_trace(t)
        self._send_json({"traces": traces, "count": len(traces)})

    def _handle_trace_detail(self, trace_id: str) -> None:
        db = self._get_db()
        row = db.execute("SELECT * FROM traces WHERE id = ?", (trace_id,)).fetchone()
        if not row:
            return self._send_error("Trace not found", 404)
        trace = dict(row)
        self._serialize_trace(trace)
        self._send_json(trace)

    def _handle_policies(self) -> None:
        db = self._get_db()
        rows = db.execute(
            "SELECT * FROM policies ORDER BY confidence DESC"
        ).fetchall()
        policies = [dict(r) for r in rows]
        for p in policies:
            self._serialize_json_fields(p)
        self._send_json({"policies": policies, "count": len(policies)})

    def _handle_skills(self) -> None:
        db = self._get_db()
        rows = db.execute("SELECT * FROM skills ORDER BY name ASC").fetchall()
        skills = [dict(r) for r in rows]
        for s in skills:
            self._serialize_json_fields(s)
        self._send_json({"skills": skills, "count": len(skills)})

    def _handle_concepts(self) -> None:
        db = self._get_db()
        try:
            rows = db.execute(
                "SELECT * FROM concepts ORDER BY created_at DESC LIMIT 100"
            ).fetchall()
            concepts = [dict(r) for r in rows]
            for c in concepts:
                self._serialize_concept(c)
            self._send_json({"concepts": concepts, "count": len(concepts)})
        except Exception as e:
            self._send_json({"concepts": [], "count": 0, "error": str(e)})

    def _handle_triples(self) -> None:
        db = self._get_db()
        try:
            rows = db.execute(
                "SELECT * FROM triples ORDER BY created_at DESC LIMIT 200"
            ).fetchall()
            triples = [dict(r) for r in rows]
            self._send_json({"triples": triples, "count": len(triples)})
        except Exception as e:
            self._send_json({"triples": [], "count": 0, "error": str(e)})

    def _handle_pipeline(self) -> None:
        """Return aggregated pipeline promotion data."""
        db = self._get_db()
        trace_count = db.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
        policy_count = db.execute("SELECT COUNT(*) FROM policies").fetchone()[0]
        skill_count = db.execute("SELECT COUNT(*) FROM skills").fetchone()[0]
        concept_count = db.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]
        triple_count = db.execute("SELECT COUNT(*) FROM triples").fetchone()[0]

        # Reward distribution
        reward_stats = db.execute(
            "SELECT ROUND(AVG(reward),3) as avg_reward, "
            "ROUND(MAX(reward),3) as max_reward, "
            "ROUND(MIN(reward),3) as min_reward "
            "FROM traces WHERE reward != 0"
        ).fetchone()

        # Top policies by confidence
        top_policies = db.execute(
            "SELECT id, name, confidence, activation_count "
            "FROM policies ORDER BY confidence DESC LIMIT 5"
        ).fetchall()

        self._send_json({
            "counts": {
                "traces": trace_count,
                "policies": policy_count,
                "skills": skill_count,
                "concepts": concept_count,
                "triples": triple_count,
            },
            "reward_stats": {
                "avg": reward_stats[0] if reward_stats else 0,
                "max": reward_stats[1] if reward_stats else 0,
                "min": reward_stats[2] if reward_stats else 0,
            },
            "top_policies": [dict(r) for r in top_policies],
        })

    def _handle_timeline(self, params: dict[str, list[str]]) -> None:
        db = self._get_db()
        limit = min(int(params.get("limit", ["20"])[0]), 100)
        rows = db.execute(
            "SELECT id, session_id, turn_index, created_at, user_content, reward, synced "
            "FROM traces ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        items = [dict(r) for r in rows]
        self._send_json({"timeline": items, "count": len(items)})

    def _handle_search(self, params: dict[str, list[str]]) -> None:
        db = self._get_db()
        q = params.get("q", [""])[0]
        if not q:
            return self._send_json({"results": [], "count": 0})
        safe = q.replace('"', '""')
        rows = db.execute(
            """
            SELECT t.id, t.session_id, t.turn_index, t.user_content,
                   t.assistant_content, t.reward, t.created_at
            FROM traces t
            JOIN traces_fts fts ON t.rowid = fts.rowid
            WHERE traces_fts MATCH ?
            ORDER BY rank
            LIMIT 20
            """, (safe,),
        ).fetchall()
        results = [dict(r) for r in rows]
        self._send_json({"results": results, "count": len(results)})

    # ── Helpers ───────────────────────────────────────────

    @staticmethod
    def _serialize_trace(t: dict[str, Any]) -> None:
        """Convert binary/complex fields to displayable formats."""
        if t.get("embedding") and isinstance(t["embedding"], str):
            try:
                emb = json.loads(t["embedding"])
                t["embedding"] = f"float[{len(emb)}]"
            except (json.JSONDecodeError, TypeError):
                t["embedding"] = "N/A"
        if t.get("tags") and isinstance(t["tags"], str):
            try:
                t["tags"] = json.loads(t["tags"])
            except (json.JSONDecodeError, TypeError):
                t["tags"] = []
        if t.get("metadata") and isinstance(t["metadata"], str):
            try:
                t["metadata"] = json.loads(t["metadata"])
            except (json.JSONDecodeError, TypeError):
                t["metadata"] = {}

    @staticmethod
    def _serialize_concept(c: dict[str, Any]) -> None:
        """Convert concept JSON string fields to displayable objects."""
        for key in ("member_trace_ids", "member_policy_ids", "metadata"):
            if key in c and isinstance(c[key], str):
                try:
                    c[key] = json.loads(c[key])
                except (json.JSONDecodeError, TypeError):
                    pass
        if c.get("embedding") and isinstance(c["embedding"], str):
            try:
                emb = json.loads(c["embedding"])
                c["embedding"] = f"float[{len(emb)}]"
            except (json.JSONDecodeError, TypeError):
                c["embedding"] = "N/A"

    @staticmethod
    def _serialize_json_fields(row: dict[str, Any]) -> None:
        for key in ("metadata", "source_trace_ids", "source_policy_ids"):
            if key in row and isinstance(row[key], str):
                try:
                    row[key] = json.loads(row[key])
                except (json.JSONDecodeError, TypeError):
                    pass
        if row.get("embedding") and isinstance(row["embedding"], str):
            try:
                emb = json.loads(row["embedding"])
                row["embedding"] = f"float[{len(emb)}]"
            except (json.JSONDecodeError, TypeError):
                row["embedding"] = "N/A"


# ── Server Launcher ───────────────────────────────────────


def serve(
    cache_path: str = "~/.hermes-next/cache.db",
    ov_url: str = "http://localhost:1933",
    port: int = 8080,
    host: str = "127.0.0.1",
) -> None:
    """Start the Hermes Next Viewer server.

    Args:
        cache_path: Path to SQLite cache database.
        ov_url: OpenViking server URL (for display).
        port: HTTP server port.
        host: Bind address.
    """
    # Initialize cache
    cache = CacheConnection(cache_path)
    ensure_schema(cache)

    # Inject into handler class
    _APIHandler._cache = cache
    _APIHandler._ov_url = ov_url

    server = HTTPServer((host, port), _APIHandler)
    print("╔══════════════════════════════════════════╗")
    print("║  Hermes Next Viewer                      ║")
    print("║  ──────────────────────                  ║")
    print(f"║  Local:   http://{host}:{port}              ║")
    print(f"║  Cache:   {cache_path}  ║")
    print(f"║  OpenViking: {ov_url}  ║")
    print("║                                          ║")
    print("║  Ctrl+C to stop                          ║")
    print("╚══════════════════════════════════════════╝")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.server_close()
        cache.close_all()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Hermes Next Viewer")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind address")
    parser.add_argument("--db", type=str, default="~/.hermes-next/cache.db", help="Cache DB path")
    parser.add_argument("--ov-url", type=str, default="http://localhost:1933", help="OpenViking URL")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    serve(
        cache_path=args.db,
        ov_url=args.ov_url,
        port=args.port,
        host=args.host,
    )


if __name__ == "__main__":
    main()


# ═══════════════════════════════════════════════════════════
# SPA HTML — embedded single-page application
# ═══════════════════════════════════════════════════════════

_SPA_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Hermes Next Viewer</title>
<style>
  :root {
    --bg: #0f1117; --surface: #1a1c25; --border: #2a2d3a;
    --text: #e1e4ea; --muted: #8b8fa3; --accent: #6c8cff;
    --green: #4caf7d; --red: #e55555; --yellow: #e5b955;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
         background: var(--bg); color: var(--text); line-height: 1.5; }
  .layout { display: flex; min-height: 100vh; }
  .sidebar { width: 220px; background: var(--surface); border-right: 1px solid var(--border);
             padding: 1.5rem 0; flex-shrink: 0; }
  .sidebar h1 { font-size: 1rem; padding: 0 1.25rem 1rem; border-bottom: 1px solid var(--border);
                margin-bottom: 0.5rem; color: var(--accent); }
  .sidebar nav a { display: block; padding: 0.6rem 1.25rem; color: var(--muted);
                   text-decoration: none; font-size: 0.875rem; transition: 0.15s; }
  .sidebar nav a:hover, .sidebar nav a.active { color: var(--text); background: rgba(108,140,255,0.1); }
  .sidebar nav a.active { border-left: 2px solid var(--accent); }
  .main { flex: 1; padding: 2rem; overflow-y: auto; max-width: calc(100vw - 220px); }
  h2 { font-size: 1.5rem; margin-bottom: 1.5rem; }
  .stats { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
           gap: 1rem; margin-bottom: 2rem; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
          padding: 1.25rem; }
  .card .num { font-size: 2rem; font-weight: 600; color: var(--accent); }
  .card .label { font-size: 0.8rem; color: var(--muted); margin-top: 0.25rem; }
  .search-bar { display: flex; gap: 0.5rem; margin-bottom: 1rem; }
  .search-bar input { flex: 1; padding: 0.6rem 1rem; border-radius: 6px; border: 1px solid var(--border);
                      background: var(--surface); color: var(--text); font-size: 0.875rem; }
  .search-bar input:focus { outline: none; border-color: var(--accent); }
  .search-bar button { padding: 0.6rem 1.2rem; border-radius: 6px; border: none;
                       background: var(--accent); color: #fff; cursor: pointer; font-size: 0.875rem; }
  .search-bar button:hover { opacity: 0.9; }
  table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  th, td { text-align: left; padding: 0.6rem 0.75rem; border-bottom: 1px solid var(--border); }
  th { color: var(--muted); font-weight: 500; font-size: 0.75rem; text-transform: uppercase; }
  tr:hover td { background: rgba(108,140,255,0.05); }
  .badge { display: inline-block; padding: 0.15rem 0.5rem; border-radius: 4px;
           font-size: 0.75rem; font-weight: 500; }
  .badge-success { background: rgba(76,175,125,0.2); color: var(--green); }
  .badge-warning { background: rgba(229,185,85,0.2); color: var(--yellow); }
  .badge-info { background: rgba(108,140,255,0.2); color: var(--accent); }
  .trace-detail { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1.5rem; }
  .trace-detail pre { background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 6px;
                      overflow-x: auto; margin-top: 0.75rem; font-size: 0.825rem; line-height: 1.6; }
  .detail-row { display: flex; gap: 0.5rem; margin-bottom: 0.5rem; }
  .detail-row .key { color: var(--muted); min-width: 120px; }
  .hidden { display: none; }
  .timeline-item { display: flex; gap: 1rem; padding: 0.75rem; border-bottom: 1px solid var(--border); }
  .timeline-item .time { color: var(--muted); min-width: 140px; font-size: 0.8rem; }
  .timeline-item .content { flex: 1; }
  .mt-2 { margin-top: 1rem; }
  .mb-1 { margin-bottom: 0.5rem; }
  .tag { display: inline-block; padding: 0.1rem 0.4rem; border-radius: 3px;
         background: rgba(108,140,255,0.15); color: var(--accent); font-size: 0.75rem; margin: 0.1rem; }
</style>
</head>
<body>
<div class="layout">
  <div class="sidebar">
    <h1>Hermes Next</h1>
    <nav>
      <a href="#" data-view="dashboard" class="active">Dashboard</a>
      <a href="#" data-view="traces">Traces</a>
      <a href="#" data-view="policies">Policies</a>
      <a href="#" data-view="skills">Skills</a>
      <a href="#" data-view="pipeline">Pipeline ⚡</a>
      <a href="#" data-view="concepts">Concepts</a>
      <a href="#" data-view="triples">Triples</a>
      <a href="#" data-view="timeline">Timeline</a>
      <a href="#" data-view="search">Search</a>
    </nav>
  </div>
  <div class="main" id="main-content">
    <div id="view-container">Loading...</div>
  </div>
</div>
<script>
const API = '';
let currentView = 'dashboard';

async function api(path) {
  const r = await fetch(API + path);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

function escape(v) {
  if (v == null) return '<span class="muted">—</span>';
  const s = String(v);
  const d = document.createElement('div');
  d.textContent = s.slice(0, 300);
  return d.innerHTML;
}

function fmtTime(iso) {
  if (!iso) return '—';
  try { return new Date(iso).toLocaleString(); } catch { return iso; }
}

// ── Views ──

async function renderDashboard() {
  const s = await api('/api/stats');
  document.getElementById('view-container').innerHTML = `
    <h2>Dashboard</h2>
    <div class="stats">
      <div class="card"><div class="num">${s.traces}</div><div class="label">Traces</div></div>
      <div class="card"><div class="num">${s.policies}</div><div class="label">Policies</div></div>
      <div class="card"><div class="num">${s.skills}</div><div class="label">Skills</div></div>
      <div class="card"><div class="num">${s.concepts ?? 0}</div><div class="label">Concepts</div></div>
      <div class="card"><div class="num">${s.triples ?? 0}</div><div class="label">Triples</div></div>
      <div class="card"><div class="num">${s.synced}</div><div class="label">Synced</div></div>
    </div>
    <div class="card">
      <div class="label mb-1">OpenViking</div>
      <div>${escape(s.ov_url)}</div>
    </div>
  `;
}

async function renderTraces() {
  const data = await api('/api/traces?limit=100');
  const traces = data.traces || [];
  document.getElementById('view-container').innerHTML = `
    <h2>Traces</h2>
    <p class="muted mb-1">${data.count} traces</p>
    <table>
      <thead><tr>
        <th>ID</th><th>Session</th><th>Turn</th><th>User</th><th>Reward</th><th>Synced</th><th>Created</th>
      </tr></thead>
      <tbody>
        ${traces.map(t => `<tr>
          <td><a href="#" onclick="showTrace('${escape(t.id)}'); return false">${escape(t.id).slice(0,12)}</a></td>
          <td>${escape(t.session_id).slice(0,12)}</td>
          <td>${t.turn_index}</td>
          <td>${escape(t.user_content).slice(0,60)}</td>
          <td>${t.reward != null ? t.reward.toFixed(2) : '0.00'}</td>
          <td>${t.synced ? '<span class="badge badge-success">yes</span>' : '<span class="badge badge-warning">no</span>'}</td>
          <td>${fmtTime(t.created_at)}</td>
        </tr>`).join('')}
      </tbody>
    </table>
    <div id="trace-detail" class="mt-2"></div>
  `;
}

async function showTrace(id) {
  const t = await api('/api/traces/' + id);
  const el = document.getElementById('trace-detail');
  el.innerHTML = `
    <div class="trace-detail">
      <h3 class="mb-1">Trace: ${escape(t.id)}</h3>
      <div class="detail-row"><span class="key">Session</span><span>${escape(t.session_id)}</span></div>
      <div class="detail-row"><span class="key">Turn</span><span>${t.turn_index}</span></div>
      <div class="detail-row"><span class="key">Reward</span><span>${t.reward?.toFixed(3) ?? '0.000'}</span></div>
      <div class="detail-row"><span class="key">Embedding</span><span>${escape(t.embedding)}</span></div>
      <div class="detail-row"><span class="key">Tags</span><span>${Array.isArray(t.tags) ? t.tags.map(x => '<span class="tag">'+escape(x)+'</span>').join('') : escape(t.tags)}</span></div>
      <div class="detail-row"><span class="key">Created</span><span>${fmtTime(t.created_at)}</span></div>
      <div class="mt-2"><strong>User:</strong><pre>${escape(t.user_content)}</pre></div>
      <div class="mt-2"><strong>Assistant:</strong><pre>${escape(t.assistant_content)}</pre></div>
    </div>
  `;
  el.scrollIntoView({ behavior: 'smooth' });
}

async function renderPolicies() {
  const data = await api('/api/policies');
  const policies = data.policies || [];
  document.getElementById('view-container').innerHTML = `
    <h2>Policies</h2>
    <p class="muted mb-1">${data.count} policies</p>
    <table>
      <thead><tr>
        <th>Name</th><th>Confidence</th><th>Activations</th><th>Description</th><th>Created</th>
      </tr></thead>
      <tbody>
        ${policies.map(p => `<tr>
          <td><strong>${escape(p.name)}</strong></td>
          <td><span class="badge ${p.confidence >= 0.5 ? 'badge-success' : 'badge-warning'}">${p.confidence?.toFixed(3)}</span></td>
          <td>${p.activation_count ?? 0}</td>
          <td>${escape(p.description).slice(0,80)}</td>
          <td>${fmtTime(p.created_at)}</td>
        </tr>`).join('')}
      </tbody>
    </table>
  `;
}

async function renderSkills() {
  const data = await api('/api/skills');
  const skills = data.skills || [];
  document.getElementById('view-container').innerHTML = `
    <h2>Skills</h2>
    <p class="muted mb-1">${data.count} skills</p>
    <table>
      <thead><tr>
        <th>Name</th><th>Version</th><th>Description</th><th>Source Policies</th><th>Created</th>
      </tr></thead>
      <tbody>
        ${skills.map(s => `<tr>
          <td><strong>${escape(s.name)}</strong></td>
          <td>v${s.version}</td>
          <td>${escape(s.description).slice(0,80)}</td>
          <td>${Array.isArray(s.source_policy_ids) ? s.source_policy_ids.length : 0}</td>
          <td>${fmtTime(s.created_at)}</td>
        </tr>`).join('')}
      </tbody>
    </table>
  `;
}

async function renderTimeline() {
  const data = await api('/api/timeline?limit=50');
  const items = data.timeline || [];
  document.getElementById('view-container').innerHTML = `
    <h2>Timeline</h2>
    <p class="muted mb-1">${data.count} recent events</p>
    ${items.map(i => `
      <div class="timeline-item">
        <div class="time">${fmtTime(i.created_at)}</div>
        <div class="content">
          <div><a href="#" onclick="switchView('traces'); setTimeout(()=>showTrace('${escape(i.id)}'),100); return false">
            <strong>${escape(i.id).slice(0,12)}</strong></a>
            <span class="badge ${i.synced ? 'badge-success' : 'badge-warning'}">${i.synced ? 'synced' : 'local'}</span>
          </div>
          <div class="muted">${escape(i.user_content).slice(0,100)}</div>
          <div class="muted">reward: ${i.reward?.toFixed(2) ?? '0.00'} · session: ${escape(i.session_id).slice(0,16)}</div>
        </div>
      </div>
    `).join('')}
  `;
}

async function renderSearch() {
  document.getElementById('view-container').innerHTML = `
    <h2>Search</h2>
    <div class="search-bar">
      <input type="text" id="search-input" placeholder="Search traces (FTS5)..." onkeydown="if(event.key==='Enter') doSearch()">
      <button onclick="doSearch()">Search</button>
    </div>
    <div id="search-results"></div>
  `;
}

async function doSearch() {
  const q = document.getElementById('search-input')?.value;
  if (!q) return;
  const data = await api('/api/search?q=' + encodeURIComponent(q));
  const results = data.results || [];
  const el = document.getElementById('search-results');
  if (results.length === 0) {
    el.innerHTML = '<p class="muted">No results found.</p>';
    return;
  }
  el.innerHTML = `
    <p class="muted mb-1">${data.count} results</p>
    <table>
      <thead><tr><th>ID</th><th>User</th><th>Assistant</th><th>Reward</th><th>Time</th></tr></thead>
      <tbody>
        ${results.map(r => `<tr>
          <td><a href="#" onclick="switchView('traces'); setTimeout(()=>showTrace('${escape(r.id)}'),100); return false">${escape(r.id).slice(0,12)}</a></td>
          <td>${escape(r.user_content).slice(0,60)}</td>
          <td>${escape(r.assistant_content).slice(0,60)}</td>
          <td>${r.reward?.toFixed(2) ?? '0.00'}</td>
          <td>${fmtTime(r.created_at)}</td>
        </tr>`).join('')}
      </tbody>
    </table>
  `;
}

// ── Pipeline View ──

async function renderPipeline() {
  const data = await api('/api/pipeline');
  const c = data.counts || {};
  const r = data.reward_stats || {};
  const top = data.top_policies || [];
  document.getElementById('view-container').innerHTML = `
    <h2>Pipeline ⚡</h2>
    <div class="stats">
      <div class="card"><div class="num">${c.traces ?? 0}</div><div class="label">L1 Traces</div></div>
      <div class="card"><div class="num">${c.policies ?? 0}</div><div class="label">L2 Policies</div></div>
      <div class="card"><div class="num">${c.skills ?? 0}</div><div class="label">Skills</div></div>
      <div class="card"><div class="num">${c.concepts ?? 0}</div><div class="label">L3 Concepts</div></div>
      <div class="card"><div class="num">${c.triples ?? 0}</div><div class="label">Triples</div></div>
    </div>
    <div class="card" style="margin-bottom:1rem">
      <div class="label mb-1">Reward Distribution</div>
      <div>Avg: <strong>${r.avg ?? '-'}</strong> · Max: <strong>${r.max ?? '-'}</strong> · Min: <strong>${r.min ?? '-'}</strong></div>
    </div>
    <h3 class="mb-1">Top Policies</h3>
    <table>
      <thead><tr><th>Name</th><th>Confidence</th><th>Activations</th></tr></thead>
      <tbody>
        ${top.map(p => `<tr>
          <td><strong>${escape(p.name)}</strong></td>
          <td><span class="badge ${p.confidence >= 0.5 ? 'badge-success' : 'badge-warning'}">${p.confidence?.toFixed(3)}</span></td>
          <td>${p.activation_count ?? 0}</td>
        </tr>`).join('')}
      </tbody>
    </table>
  `;
}

// ── Concepts View ──

async function renderConcepts() {
  const data = await api('/api/concepts');
  const concepts = data.concepts || [];
  document.getElementById('view-container').innerHTML = `
    <h2>Concepts</h2>
    <p class="muted mb-1">${data.count} concepts</p>
    <table>
      <thead><tr><th>Label</th><th>Description</th><th>Traces</th><th>Policies</th><th>Created</th></tr></thead>
      <tbody>
        ${concepts.map(c => `<tr>
          <td><strong>${escape(c.label)}</strong></td>
          <td>${escape(c.description).slice(0,60)}</td>
          <td>${Array.isArray(c.member_trace_ids) ? c.member_trace_ids.length : 0}</td>
          <td>${Array.isArray(c.member_policy_ids) ? c.member_policy_ids.length : 0}</td>
          <td>${fmtTime(c.created_at)}</td>
        </tr>`).join('')}
      </tbody>
    </table>
  `;
}

// ── Triples View ──

async function renderTriples() {
  const data = await api('/api/triples');
  const triples = data.triples || [];
  document.getElementById('view-container').innerHTML = `
    <h2>Triples</h2>
    <p class="muted mb-1">${data.count} triples</p>
    <table>
      <thead><tr><th>Subject</th><th>Predicate</th><th>Object</th><th>Confidence</th><th>Created</th></tr></thead>
      <tbody>
        ${triples.map(t => `<tr>
          <td>${escape(t.subject).slice(0,30)}</td>
          <td><span class="badge badge-info">${escape(t.predicate)}</span></td>
          <td>${escape(t.object).slice(0,30)}</td>
          <td>${t.confidence?.toFixed(2) ?? '-'}</td>
          <td>${fmtTime(t.created_at)}</td>
        </tr>`).join('')}
      </tbody>
    </table>
  `;
}

// ── Navigation ──

const VIEWS = {
  dashboard: renderDashboard,
  traces: renderTraces,
  policies: renderPolicies,
  skills: renderSkills,
  pipeline: renderPipeline,
  concepts: renderConcepts,
  triples: renderTriples,
  timeline: renderTimeline,
  search: renderSearch,
};

async function switchView(name) {
  currentView = name;
  document.querySelectorAll('.sidebar nav a').forEach(a => {
    a.classList.toggle('active', a.dataset.view === name);
  });
  const renderFn = VIEWS[name];
  if (renderFn) await renderFn();
}

document.querySelectorAll('.sidebar nav a').forEach(a => {
  a.addEventListener('click', e => {
    e.preventDefault();
    switchView(a.dataset.view);
  });
});

// ── Init ──
switchView('dashboard');
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
