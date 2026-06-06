# Hermes Next

**Next-generation memory provider for Hermes Agent.**

Hermes Next fuses [OpenViking](https://github.com/bytedance/openviking) vector storage with a Python-native [MemOS](https://github.com/memtensor/memos) cognitive engine, giving Hermes Agent agents persistent,结构化记忆能力。

## Features

- **OpenViking Backend** — Long-term vector storage, semantic retrieval, session management
- **MemOS Cognitive Pipeline** — L1 Trace capture → reward backpropagation → L2 Policy induction → L3 World Model → Skill crystallization
- **Python Native** — Zero bridging overhead, runs in-process
- **Local SQLite Cache** — FTS5 full-text search + numpy-based cosine similarity, zero dependencies beyond stdlib
- **Fusion Retrieval** — 6-step pipeline combining semantic search, full-text search, policy matching, timeline, recency boost, and MMR diversification

## Installation

```bash
pip install hermes-next
```

Requires Python 3.10+ and a running OpenViking server (v0.3.22+).

## Configuration

Create a `hermes-next.yaml` in your config directory:

```yaml
openviking:
  base_url: "http://localhost:1933"
  api_key: null

cache:
  path: "~/.hermes-next/cache.db"
  enable_fts: true

agent:
  name: "default"
```

Or set environment variables:

```bash
export HERMES_NEXT_OV_URL="http://localhost:1933"
export HERMES_NEXT_CACHE_PATH="~/.hermes-next/cache.db"
```

## Usage with Hermes Agent

```python
from hermes_next import HermesNextProvider

provider = HermesNextProvider()
provider.initialize(session_id="my-session")

# The provider handles prefetching, storage, and retrieval automatically
context = provider.prefetch("What did we discuss about RAG?")
print(context)
```

Or via CLI:

```bash
hermes agent --memory-provider hermes-next
```

## Tools

The provider exposes these tools to the agent:

| Tool | Description |
|------|-------------|
| `memos_search(query, k)` | Semantic search across traces |
| `memos_get(trace_id)` | Read a specific trace |
| `memos_timeline(limit)` | Recent activity timeline |

## Project Structure

```
hermes-next/
├── hermes_next/
│   ├── ov/            # OpenViking REST client
│   ├── memos/         # MemOS cognitive engine
│   ├── cache/         # SQLite local cache
│   └── retrieval/     # Fusion retrieval pipeline
├── tests/
└── plugin.yaml        # Hermes plugin manifest
```

## License

AGPL-3.0 — This project is a derivative of OpenViking (AGPL-3.0).
