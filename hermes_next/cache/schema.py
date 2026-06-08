"""SQLite schema — traces, policies, skills tables with FTS5."""

from __future__ import annotations

from hermes_next.cache.connection import CacheConnection


def ensure_schema(conn_or_cache: CacheConnection) -> None:
    """Create all tables and indexes if they don't exist."""
    conn = conn_or_cache.conn if isinstance(conn_or_cache, CacheConnection) else conn_or_cache

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS traces (
            id          TEXT PRIMARY KEY,
            session_id  TEXT NOT NULL,
            turn_index  INTEGER NOT NULL DEFAULT 0,
            user_content TEXT NOT NULL,
            assistant_content TEXT NOT NULL DEFAULT '',
            embedding   BLOB,
            reward      REAL NOT NULL DEFAULT 0.0,
            tags        TEXT DEFAULT '',
            metadata    TEXT DEFAULT '{}',
            created_at  TEXT NOT NULL,
            synced      INTEGER NOT NULL DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_traces_session
            ON traces(session_id);
        CREATE INDEX IF NOT EXISTS idx_traces_created
            ON traces(created_at);
        CREATE INDEX IF NOT EXISTS idx_traces_synced
            ON traces(synced);

        CREATE TABLE IF NOT EXISTS policies (
            id              TEXT PRIMARY KEY,
            name            TEXT NOT NULL,
            description     TEXT NOT NULL DEFAULT '',
            trigger_pattern TEXT NOT NULL DEFAULT '',
            action_template TEXT NOT NULL DEFAULT '',
            embedding       BLOB,
            confidence      REAL NOT NULL DEFAULT 0.0,
            activation_count INTEGER NOT NULL DEFAULT 0,
            source_trace_ids TEXT DEFAULT '[]',
            metadata        TEXT DEFAULT '{}',
            created_at      TEXT NOT NULL,
            synced          INTEGER NOT NULL DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_policies_confidence
            ON policies(confidence DESC);

        CREATE TABLE IF NOT EXISTS skills (
            name            TEXT PRIMARY KEY,
            description     TEXT NOT NULL DEFAULT '',
            usage_guide     TEXT NOT NULL DEFAULT '',
            source_policy_ids TEXT DEFAULT '[]',
            version         INTEGER NOT NULL DEFAULT 1,
            metadata        TEXT DEFAULT '{}',
            created_at      TEXT NOT NULL
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS traces_fts
            USING fts5(
                user_content,
                assistant_content,
                tags,
                content='traces',
                content_rowid='rowid'
            );

        CREATE TRIGGER IF NOT EXISTS traces_ai AFTER INSERT ON traces BEGIN
            INSERT INTO traces_fts(rowid, user_content, assistant_content, tags)
            VALUES (new.rowid, new.user_content, new.assistant_content, new.tags);
        END;

        CREATE TRIGGER IF NOT EXISTS traces_ad AFTER DELETE ON traces BEGIN
            INSERT INTO traces_fts(traces_fts, rowid, user_content, assistant_content, tags)
            VALUES ('delete', old.rowid, old.user_content, old.assistant_content, old.tags);
        END;

        CREATE TRIGGER IF NOT EXISTS traces_au AFTER UPDATE ON traces BEGIN
            INSERT INTO traces_fts(traces_fts, rowid, user_content, assistant_content, tags)
            VALUES ('delete', old.rowid, old.user_content, old.assistant_content, old.tags);
            INSERT INTO traces_fts(rowid, user_content, assistant_content, tags)
            VALUES (new.rowid, new.user_content, new.assistant_content, new.tags);
        END;

        CREATE TABLE IF NOT EXISTS concepts (
            id          TEXT PRIMARY KEY,
            label       TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            embedding   BLOB,
            member_trace_ids  TEXT DEFAULT '[]',
            member_policy_ids TEXT DEFAULT '[]',
            metadata    TEXT DEFAULT '{}',
            created_at  TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_concepts_label
            ON concepts(label);

        CREATE TABLE IF NOT EXISTS triples (
            id             TEXT PRIMARY KEY,
            subject        TEXT NOT NULL,
            predicate      TEXT NOT NULL,
            object         TEXT NOT NULL,
            confidence     REAL NOT NULL DEFAULT 1.0,
            source_trace_id TEXT,
            metadata       TEXT DEFAULT '{}',
            created_at     TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_triples_subject
            ON triples(subject);
        CREATE INDEX IF NOT EXISTS idx_triples_predicate
            ON triples(predicate);
        CREATE INDEX IF NOT EXISTS idx_triples_object
            ON triples(object);

        CREATE TABLE IF NOT EXISTS feedback (
            id          TEXT PRIMARY KEY,
            episode_id  TEXT NOT NULL,
            trace_id    TEXT,
            polarity    TEXT NOT NULL,
            magnitude   REAL NOT NULL DEFAULT 1.0,
            text        TEXT,
            source      TEXT NOT NULL DEFAULT 'user',
            agent_name  TEXT NOT NULL DEFAULT 'default',
            created_at  TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_feedback_episode
            ON feedback(episode_id);
        CREATE INDEX IF NOT EXISTS idx_feedback_agent
            ON feedback(agent_name);
        CREATE INDEX IF NOT EXISTS idx_feedback_polarity
            ON feedback(polarity);

        CREATE TABLE IF NOT EXISTS session_state (
            session_id    TEXT PRIMARY KEY,
            agent_name    TEXT NOT NULL,
            turn_index    INTEGER NOT NULL DEFAULT 0,
            status        TEXT NOT NULL DEFAULT 'open',
            opened_at     TEXT NOT NULL,
            last_active   TEXT NOT NULL,
            metadata      TEXT DEFAULT '{}'
        );

        CREATE INDEX IF NOT EXISTS idx_session_status
            ON session_state(status);
        CREATE INDEX IF NOT EXISTS idx_session_agent
            ON session_state(agent_name);
    """)
    conn.commit()


def drop_schema(conn_or_cache: CacheConnection) -> None:
    """Drop all tables (for testing)."""
    conn = conn_or_cache.conn if isinstance(conn_or_cache, CacheConnection) else conn_or_cache
    conn.executescript("""
        DROP TABLE IF EXISTS traces_fts;
        DROP TABLE IF EXISTS skills;
        DROP TABLE IF EXISTS policies;
        DROP TABLE IF EXISTS feedback;
        DROP TABLE IF EXISTS session_state;
        DROP TABLE IF EXISTS triples;
        DROP TABLE IF EXISTS concepts;
        DROP TABLE IF EXISTS traces;
    """)
    conn.commit()
