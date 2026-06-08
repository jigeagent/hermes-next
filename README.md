# Hermes Next

**A self-evolving memory system for AI agents. Python-native, zero-bridge, brain-like cognition.**

> Hermes Next fuses OpenViking vector storage with a Python-native MemOS cognitive engine, giving Hermes Agent agents persistent, self-evolving memory — no JSON-RPC bridge, no TypeScript dependency, no process overhead.

[![PyPI](https://img.shields.io/pypi/v/hermes-next)](https://pypi.org/project/hermes-next/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-AGPL--3.0-blue)](LICENSE)
[![Tests](https://github.com/jigeagent/hermes-next/actions/workflows/ci.yml/badge.svg)](https://github.com/jigeagent/hermes-next/actions)

---

## Why Hermes Next?

Most agent memory systems today are **TypeScript plugins bolted onto Python agents via JSON-RPC bridges** — every memory operation crosses a process boundary, serializes and deserializes, and introduces a failure point.

Hermes Next is **Python-native from day one**. It runs in-process, shares memory with the agent, and eliminates the bridge entirely.

| Feature | Traditional (memos-local) | Hermes Next |
|---------|--------------------------|-------------|
| Language | TypeScript + JSON-RPC bridge | **Python native** |
| Search | Brute-force cosine | **6-step fusion** (semantic + FTS5 + timeline + RRF + MMR + recency) |
| Lifecycle | None (data grows forever) | **Auto-archive** (90d) + **confidence decay** + pruning |
| Native memory sync | ❌ | ✅ MEMORY.md promotion + session_search fallback |
| Cross-agent sharing | JSON-RPC | ✅ OpenViking namespaces |
| Cognitive pipeline | L1/L2/L3/Skill | **L1→Reward→L2→L3→Skill** fully wired |
| Setup | `npm install` + bridge config | `pip install` |

**Bottom line**: memos-local was built for the TypeScript era. Hermes Next is built for the Python-native era.

---

## Quick Start

```bash
pip install hermes-next
```

Requires Python 3.10+ and a running [OpenViking](https://github.com/bytedance/openviking) server (v0.3.22+).

### Basic Usage

```python
from hermes_next import HermesNextProvider

# Initialize
provider = HermesNextProvider()
provider.initialize(session_id="my-session")

# Every turn — automatic capture + retrieval
context = provider.prefetch("What did we discuss about RAG?")
provider.sync_turn(
    user_content="Tell me about RAG optimization",
    assistant_content="Key techniques: chunk size tuning, embedding selection...",
    session_id="my-session",
    tags=["rag", "optimization"],
)

# At session end — cognitive pipeline runs automatically
provider.on_session_end(messages=[])

# Check pipeline health
status = provider.handle_tool_call("memos_status", {})
print(status)
```

### CLI

```bash
# Start the built-in viewer
hermes-next-viewer --port 8080

# Migrate from legacy memos-local-plugin database
hermes-next-migrate --old-db ~/.hermes/memos-plugin/data/memos.db
```

---

## Architecture

```
                         Hermes Agent (Python)
                              │
                    HermesNextProvider
                     ┌────────┴────────┐
                     │                 │
              RetrievalPipeline   CognitivePipeline
              ┌─────┼───┬───┐    ┌───┼───┬───┬───┐
              │     │   │   │    │   │   │   │   │
           Semantic FTS5 Timel. MMR  L1  L2  L3  Skill
           (OV)   (SQLite)    (Rerank)  (Policy)(WM)(Cryst.)
                     │                 │
                  ┌──┴──┐          ┌───┴───┐
             OpenViking  SQLite  MEMORY.md  state.db
             (Vector)   (Cache)  (Native)  (Fallback)
```

### 4-Layer Retrieval Chain

1. **Hot Memory** — MEMORY.md / USER.md (always in system prompt)
2. **Semantic Search** — OpenViking vector store (primary, 95% of queries)
3. **FTS5 Full-text** — Local SQLite cache (keyword fallback)
4. **session_search** — Hermes Agent native state.db (last resort)

### Cognitive Pipeline

Every session triggers an automatic induction chain:

```
L1 Traces → Reward Backprop → L2 Policy Induction → L3 World Model → Skill Crystallization
   ↓              ↓                  ↓                   ↓                 ↓
 Raw turns    Score each       Extract reusable       Build domain      Package as
              interaction     behavioral patterns     knowledge         invocable skills
```

---

## Configuration

```yaml
# ~/.hermes-next.yaml
openviking:
  base_url: "http://localhost:1933"

cognitive:
  enable_l2_induction: true
  enable_l3_world_model: false   # opt-in, GPT-intensive
  enable_skill_crystallization: false

lifecycle:
  trace_retention_days: 90
  policy_decay_rate: 0.03

integration:
  sync_memory_md: false           # promote to Hermes Agent MEMORY.md
  session_search_fallback: true    # fallback to native state.db FTS5
```

---

## Tools Exposed to the Agent

| Tool | Description |
|------|-------------|
| `memos_search(query, k)` | Semantic search across all memories |
| `memos_get(trace_id)` | Read a specific trace |
| `memos_timeline(limit)` | Recent memory activity |
| `memos_status` | Pipeline health + promotion stats |

---

## Project Structure

```
hermes-next/
├── hermes_next/
│   ├── ov/            # OpenViking REST client
│   ├── memos/         # MemOS cognitive engine
│   ├── cache/         # SQLite local cache (FTS5 + vector)
│   ├── retrieval/     # 6-step fusion pipeline
│   ├── integration/   # Hermes Agent native memory bridge
│   └── viewer/        # Built-in SPA dashboard
├── tests/             # 141 tests, 0 flaky
└── docs/              # Architecture + upgrade guides
```

---

## Comparison with memos-local-plugin

| Aspect | memos-local-plugin | Hermes Next |
|--------|-------------------|-------------|
| Runtime | TypeScript, separate process | **Python, in-process** |
| Communication | JSON-RPC over stdio | **Direct function calls** |
| Vector search | Brute-force cosine (SQLite) | **OpenViking (real vector DB)** |
| Full-text search | ❌ | ✅ **FTS5** |
| Memory lifecycle | ❌ | ✅ **Auto archive + decay** |
| Native Hermes sync | ❌ | ✅ **MEMORY.md + session_search** |
| Cross-agent sharing | Per-process | ✅ **OpenViking namespaces** |
| Viewer | HTTP + SSE | **HTTP SPA** |
| Pipeline | 3-tier retrieval | **6-step fusion + cognitive** |

---

## Roadmap

- **v0.4.0** — Feedback loop (user 👍/👎 → policy update → better retrieval)
- **v0.5.0** — Decision Repair (failure patterns → prevention)
- **v0.6.0** — Hub cross-agent search (native, no OV dependency)

See [docs/strategic-positioning.md](docs/strategic-positioning.md) for the full strategy.

---

## License

AGPL-3.0 — This project is a derivative of [OpenViking](https://github.com/volcengine/openviking) (AGPL-3.0).

---

*Built for agents that learn from every conversation.* 🤛
