"""Configuration management for Hermes Next."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


@dataclass
class OpenVikingConfig:
    """OpenViking server connection settings."""

    base_url: str = "http://localhost:1933"
    api_key: Optional[str] = None
    timeout: float = 30.0
    max_retries: int = 3


@dataclass
class CacheConfig:
    """Local SQLite cache settings."""

    path: str = "~/.hermes-next/cache.db"
    enable_fts: bool = True
    wal_mode: bool = True


@dataclass
class AgentConfig:
    """Agent identity settings."""

    name: str = "default"
    role: str = "assistant"


@dataclass
class RetrievalConfig:
    """Fusion retrieval pipeline settings."""

    semantic_k: int = 16
    fts_k: int = 8
    policy_k: int = 4
    timeline_k: int = 8
    rrf_k: int = 60
    mmr_lambda: float = 0.7
    mmr_k: int = 8
    recency_decay: float = 0.9


@dataclass
class CognitiveConfig:
    """Cognitive pipeline settings — L1 → Reward → L2 → L3 → Skill."""

    auto_reward_on_session_end: bool = True
    """Automatically apply session-level reward when session ends."""

    enable_l2_induction: bool = True
    """Enable L2 Policy induction from rewarded traces."""

    enable_l3_world_model: bool = False
    """Enable L3 World Model abstraction (opt-in, more expensive)."""

    enable_skill_crystallization: bool = False
    """Enable Skill crystallization from high-confidence policies (opt-in)."""

    min_policies_before_skill: int = 3
    """Minimum policies needed before skill crystallization activates."""

    min_traces_before_l2: int = 5
    """Minimum traces before L2 induction is attempted."""


@dataclass
class LifecycleConfig:
    """Memory lifecycle settings — archive, decay, and cleanup."""

    trace_retention_days: int = 90
    """Traces older than this many days are archived (deleted from local cache)."""

    policy_decay_rate: float = 0.03
    """Confidence decay per day of inactivity (0.0 = no decay)."""

    policy_min_confidence: float = 0.05
    """Policies below this confidence are pruned."""

    cleanup_interval_traces: int = 500
    """Run cleanup every N new traces."""


@dataclass
class IntegrationConfig:
    """Native Hermes Agent memory integration settings."""

    sync_memory_md: bool = True
    """Auto-write high-value summaries to Hermes Agent native MEMORY.md."""

    promote_on_l2_confidence: float = 0.5
    """L2 Policy confidence threshold for MEMORY.md promotion."""

    promote_on_skill_crystallize: bool = True
    """Write crystallized skill summaries to MEMORY.md."""

    session_search_fallback: bool = True
    """Fall back to Hermes Agent state.db FTS5 when hermes-next returns empty."""

    session_search_max_results: int = 5
    """Max results from session_search fallback."""


@dataclass
class HermesNextConfig:
    """Top-level configuration."""

    openviking: OpenVikingConfig = field(default_factory=OpenVikingConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    cognitive: CognitiveConfig = field(default_factory=CognitiveConfig)
    lifecycle: LifecycleConfig = field(default_factory=LifecycleConfig)
    integration: IntegrationConfig = field(default_factory=IntegrationConfig)

    @classmethod
    def load(cls, path: Optional[str] = None) -> "HermesNextConfig":
        """Load config from YAML file, with env var overrides."""
        cfg = cls()

        # Try loading from file
        if path:
            cfg._load_file(path)
        else:
            # Search default locations
            for candidate in cls._default_paths():
                if candidate.exists():
                    cfg._load_file(str(candidate))
                    break

        # Env var overrides
        cfg._apply_env_overrides()

        return cfg

    def _load_file(self, path: str) -> None:
        """Load config from a YAML file."""
        p = Path(path).expanduser()
        if not p.exists():
            return
        raw = p.read_text(encoding="utf-8")
        if yaml is None:
            return
        data = yaml.safe_load(raw) or {}

        if "openviking" in data:
            ov = data["openviking"]
            if "base_url" in ov:
                self.openviking.base_url = ov["base_url"]
            if "api_key" in ov:
                self.openviking.api_key = ov.get("api_key")
            if "timeout" in ov:
                self.openviking.timeout = float(ov["timeout"])

        if "cache" in data:
            c = data["cache"]
            if "path" in c:
                self.cache.path = c["path"]
            if "enable_fts" in c:
                self.cache.enable_fts = bool(c["enable_fts"])

        if "agent" in data:
            a = data["agent"]
            if "name" in a:
                self.agent.name = a["name"]
            if "role" in a:
                self.agent.role = a["role"]

        if "retrieval" in data:
            r = data["retrieval"]
            for key in ("semantic_k", "fts_k", "policy_k", "timeline_k",
                        "rrf_k", "mmr_lambda", "mmr_k", "recency_decay"):
                if key in r:
                    setattr(self.retrieval, key, r[key])

        if "cognitive" in data:
            cog = data["cognitive"]
            if "auto_reward_on_session_end" in cog:
                self.cognitive.auto_reward_on_session_end = bool(cog["auto_reward_on_session_end"])
            if "enable_l2_induction" in cog:
                self.cognitive.enable_l2_induction = bool(cog["enable_l2_induction"])
            if "enable_l3_world_model" in cog:
                self.cognitive.enable_l3_world_model = bool(cog["enable_l3_world_model"])
            if "enable_skill_crystallization" in cog:
                self.cognitive.enable_skill_crystallization = bool(cog["enable_skill_crystallization"])
            if "min_policies_before_skill" in cog:
                self.cognitive.min_policies_before_skill = int(cog["min_policies_before_skill"])
            if "min_traces_before_l2" in cog:
                self.cognitive.min_traces_before_l2 = int(cog["min_traces_before_l2"])

        if "lifecycle" in data:
            lc = data["lifecycle"]
            if "trace_retention_days" in lc:
                self.lifecycle.trace_retention_days = int(lc["trace_retention_days"])
            if "policy_decay_rate" in lc:
                self.lifecycle.policy_decay_rate = float(lc["policy_decay_rate"])
            if "policy_min_confidence" in lc:
                self.lifecycle.policy_min_confidence = float(lc["policy_min_confidence"])
            if "cleanup_interval_traces" in lc:
                self.lifecycle.cleanup_interval_traces = int(lc["cleanup_interval_traces"])

        if "integration" in data:
            ig = data["integration"]
            if "sync_memory_md" in ig:
                self.integration.sync_memory_md = bool(ig["sync_memory_md"])
            if "promote_on_l2_confidence" in ig:
                self.integration.promote_on_l2_confidence = float(ig["promote_on_l2_confidence"])
            if "promote_on_skill_crystallize" in ig:
                self.integration.promote_on_skill_crystallize = bool(ig["promote_on_skill_crystallize"])
            if "session_search_fallback" in ig:
                self.integration.session_search_fallback = bool(ig["session_search_fallback"])
            if "session_search_max_results" in ig:
                self.integration.session_search_max_results = int(ig["session_search_max_results"])

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        env_map = {
            "HERMES_NEXT_OV_URL": ("openviking", "base_url"),
            "HERMES_NEXT_OV_API_KEY": ("openviking", "api_key"),
            "HERMES_NEXT_CACHE_PATH": ("cache", "path"),
            "HERMES_NEXT_AGENT_NAME": ("agent", "name"),
        }
        for env_var, (section, attr) in env_map.items():
            val = os.environ.get(env_var)
            if val is not None:
                getattr(self, section).__setattr__(attr, val)

    @staticmethod
    def _default_paths() -> list[Path]:
        """Return list of default config file paths to search."""
        home = Path.home()
        return [
            home / ".config" / "hermes-next" / "hermes-next.yaml",
            home / ".hermes-next" / "hermes-next.yaml",
            home / ".hermes-next.yaml",
            Path.cwd() / "hermes-next.yaml",
            Path.cwd() / ".hermes-next.yaml",
        ]


def get_config(path: Optional[str] = None) -> HermesNextConfig:
    """Convenience function to load configuration."""
    return HermesNextConfig.load(path)
