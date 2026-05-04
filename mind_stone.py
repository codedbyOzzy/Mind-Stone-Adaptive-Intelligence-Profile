"""
Mind Stone — Adaptive Intelligence Profile
==========================================
A lightweight, self-calibrating module that learns *how* a user communicates
and adjusts an AI assistant's style accordingly — without any explicit setup.

No external dependencies. Pure Python standard library.
Works with any LLM backend (OpenAI, Anthropic, Gemini, Ollama, etc.).

Core idea
---------
Most AI assistants know *what* to say but not *how* to say it for this
specific person. Mind Stone silently observes each conversation turn and
builds a quantified profile of the user's communication preferences:

  • Do they prefer short, punchy answers or detailed explanations?
  • Are they a technical expert or do they need plain language?
  • Do they learn better from examples or from theory first?
  • Are they quickly satisfied or do they always ask follow-ups?

After enough observations, Mind Stone generates a short style directive
that is injected into the system prompt — shaping how the assistant talks,
not what it knows.

Usage (5 lines)
---------------
    from mind_stone import MindStone

    stone = MindStone()                              # loads saved profile

    # After each conversation turn:
    stone.observe(user_message, assistant_message)

    # Before each LLM call:
    directive = stone.get_style_directive()          # "" until enough data
    if directive:
        system_prompt += "\\n\\n" + directive

Profile is auto-saved to `.mind_stone.json` every 5 turns.

Customisation
-------------
Mind Stone ships with English signal sets. To use another language, pass
a custom SignalConfig:

    from mind_stone import MindStone, SignalConfig
    from signals_turkish import TR_CONFIG   # example file in this repo

    stone = MindStone(config=TR_CONFIG)

License: MIT
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class SignalConfig:
    """Language-specific signal sets.

    All values are plain ASCII strings — normalise diacritics before
    passing text to observe() if your language uses them (see _norm helper).

    Attributes
    ----------
    neg_verbosity   Phrases that mean "give me shorter answers"
    pos_verbosity   Phrases that mean "give me more detail"
    example_signals Phrases that mean "show me an example / code"
    theory_signals  Phrases that mean "explain the why / theory"
    satisfied_tokens Short tokens that indicate the user is satisfied with the reply
    tech_words      Vocabulary that indicates technical expertise
    normalise_fn    Optional callable to normalise text before signal matching
                    (e.g. strip diacritics). Receives lowercase str, returns str.
    """
    neg_verbosity:    frozenset = field(default_factory=frozenset)
    pos_verbosity:    frozenset = field(default_factory=frozenset)
    example_signals:  frozenset = field(default_factory=frozenset)
    theory_signals:   frozenset = field(default_factory=frozenset)
    satisfied_tokens: frozenset = field(default_factory=frozenset)
    tech_words:       frozenset = field(default_factory=frozenset)
    normalise_fn:     Optional[object] = None   # callable | None


# ── Default English signal sets ───────────────────────────────────────────────

_EN_NEG_VERBOSITY = frozenset({
    "shorter", "too long", "brief", "summarise", "summarize", "tldr", "tl;dr",
    "cut it", "just tell me", "keep it short", "skip the details",
    "dont need all that", "less words", "concise",
})

_EN_POS_VERBOSITY = frozenset({
    "more detail", "elaborate", "expand", "go deeper", "tell me more",
    "explain further", "what else", "keep going", "continue", "and then",
    "give me more", "in depth", "comprehensive", "thorough",
})

_EN_EXAMPLE_SIGNALS = frozenset({
    "example", "show me", "code", "demo", "in practice", "how does it look",
    "what does it look like", "can you show", "sample", "snippet",
    "practical", "real world", "use case",
})

_EN_THEORY_SIGNALS = frozenset({
    "why", "how does it work", "what is the reason", "explain",
    "behind the scenes", "under the hood", "the concept", "principle",
    "what makes it", "theory", "fundamentals", "philosophy",
})

_EN_SATISFIED_TOKENS = frozenset({
    "ok", "got it", "thanks", "thank you", "perfect", "great",
    "makes sense", "understood", "clear", "nice", "cool", "awesome",
    "exactly", "yep", "yes", "yup",
})

_EN_TECH_WORDS = frozenset({
    # Languages
    "python", "javascript", "typescript", "rust", "golang", "java", "cpp",
    "sql", "bash", "shell", "swift", "kotlin", "scala",
    # Concepts
    "api", "rest", "graphql", "websocket", "async", "await", "thread",
    "queue", "class", "function", "method", "object", "array", "dict",
    "json", "xml", "yaml", "regex", "token", "stream", "buffer", "mutex",
    # AI / ML
    "gpu", "cuda", "cpu", "ram", "embedding", "vector", "model",
    "inference", "finetune", "fine-tune", "rag", "prompt", "llm",
    "neural", "transformer", "gradient", "layer", "attention",
    # Infrastructure
    "docker", "kubernetes", "k8s", "git", "linux", "nginx", "redis",
    "postgres", "mongodb", "kafka", "celery", "ci", "cd", "pipeline",
    # General engineering
    "algorithm", "complexity", "latency", "throughput", "cache", "index",
    "schema", "migration", "refactor", "debug", "profil", "benchmark",
    "concurren", "distributed", "microservice", "monolith",
})

EN_CONFIG = SignalConfig(
    neg_verbosity    = _EN_NEG_VERBOSITY,
    pos_verbosity    = _EN_POS_VERBOSITY,
    example_signals  = _EN_EXAMPLE_SIGNALS,
    theory_signals   = _EN_THEORY_SIGNALS,
    satisfied_tokens = _EN_SATISFIED_TOKENS,
    tech_words       = _EN_TECH_WORDS,
    normalise_fn     = None,   # English ASCII — no normalisation needed
)


# ── Profile data structure ─────────────────────────────────────────────────────

@dataclass
class IntelligenceProfile:
    """Quantified model of a user's communication preferences.

    All float attributes use Exponential Moving Average (EMA) updates.
    Values are bounded to [0.0, 1.0].
    """

    # Core style dimensions
    verbosity:      float = 0.50  # 0 = terse,   1 = verbose
    tech_depth:     float = 0.50  # 0 = plain,   1 = expert
    example_bias:   float = 0.50  # 0 = theory,  1 = examples
    follow_up_rate: float = 0.50  # 0 = satisfied easily, 1 = always wants more

    # Temporal patterns
    peak_hours:  list  = field(default_factory=list)  # top-3 active hours
    hour_counts: dict  = field(default_factory=dict)  # {hour: message_count}

    # Metadata
    total_turns: int   = 0
    created_at:  float = field(default_factory=time.time)
    updated_at:  float = field(default_factory=time.time)

    def confidence(self) -> float:
        """Profile reliability, 0→1.

        Reaches ~50% at 10 turns, ~100% at 55 turns.
        Style directives only activate above a confidence threshold.
        """
        return min(1.0, max(0.0, (self.total_turns - 5) / 50))

    def as_dict(self) -> dict:
        return asdict(self)


# ── Core engine ────────────────────────────────────────────────────────────────

class MindStone:
    """Self-calibrating communication style engine.

    Parameters
    ----------
    path : Path | str
        Where to persist the profile (JSON). Default: ``.mind_stone.json``.
    config : SignalConfig
        Language-specific signal sets. Default: English (EN_CONFIG).
    ema_alpha : float
        Learning rate for EMA updates. Lower = more stable, slower.
        Default: 0.12 (a session of ~40 turns to fully calibrate).
    min_confidence : float
        Minimum profile confidence before directives are generated.
        Default: 0.15 (~12 turns).
    save_every : int
        Persist profile to disk every N turns. Default: 5.
    """

    def __init__(
        self,
        path:           str | Path     = ".mind_stone.json",
        config:         SignalConfig   = EN_CONFIG,
        ema_alpha:      float          = 0.12,
        min_confidence: float          = 0.15,
        save_every:     int            = 5,
    ) -> None:
        self._path           = Path(path)
        self._config         = config
        self._alpha          = ema_alpha
        self._min_conf       = min_confidence
        self._save_every     = save_every
        self.profile         = self._load()

    # ── Persistence ────────────────────────────────────────────────────────────

    def _load(self) -> IntelligenceProfile:
        if not self._path.exists():
            return IntelligenceProfile()
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            p = IntelligenceProfile(
                verbosity      = float(data.get("verbosity",      0.50)),
                tech_depth     = float(data.get("tech_depth",     0.50)),
                example_bias   = float(data.get("example_bias",   0.50)),
                follow_up_rate = float(data.get("follow_up_rate", 0.50)),
                peak_hours     = list(data.get("peak_hours",      [])),
                hour_counts    = {int(k): int(v)
                                  for k, v in data.get("hour_counts", {}).items()},
                total_turns    = int(data.get("total_turns",      0)),
                created_at     = float(data.get("created_at",     time.time())),
                updated_at     = float(data.get("updated_at",     time.time())),
            )
            return p
        except Exception:
            return IntelligenceProfile()

    def _save(self) -> None:
        try:
            self.profile.updated_at = time.time()
            self._path.write_text(
                json.dumps(self.profile.as_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            print(f"[MindStone] Save error: {exc}")

    # ── Core API ───────────────────────────────────────────────────────────────

    def observe(self, user_message: str, assistant_message: str) -> None:
        """Update the profile from one conversation turn.

        Call this after every user → assistant exchange, passing the raw
        text of both messages. Thread-safe if called from a single thread;
        add your own lock for multi-threaded use.

        Parameters
        ----------
        user_message      The user's raw message text.
        assistant_message The assistant's full response text.
        """
        p   = self.profile
        cfg = self._config

        p.total_turns += 1

        # Normalise and lowercase for signal matching
        um = (user_message or "").strip().lower()
        if callable(cfg.normalise_fn):
            um = cfg.normalise_fn(um)

        # ── Hour tracking ──────────────────────────────────────────────────────
        hour = datetime.now().hour
        p.hour_counts[hour] = p.hour_counts.get(hour, 0) + 1
        p.peak_hours = _top_hours(p.hour_counts, n=3)

        # ── Verbosity signals ──────────────────────────────────────────────────
        um_words = set(re.findall(r"\w+", um))
        um_text  = um  # for multi-word substring matching

        if _matches(um_text, um_words, cfg.neg_verbosity):
            # Strong negative signal — faster correction
            p.verbosity = _ema(p.verbosity, 0.0, alpha=0.28)
        elif _matches(um_text, um_words, cfg.pos_verbosity):
            p.verbosity = _ema(p.verbosity, 1.0, alpha=0.22)
        else:
            # Soft signal: user message length as a proxy for verbosity preference
            word_count    = len(um.split())
            length_signal = min(1.0, word_count / 15)
            p.verbosity   = _ema(p.verbosity, length_signal, alpha=self._alpha * 0.4)

        p.verbosity = _clamp(p.verbosity)

        # ── Technical depth ────────────────────────────────────────────────────
        if um_words:
            tech_ratio = len(um_words & cfg.tech_words) / max(len(um_words), 1)
            tech_signal = min(1.0, tech_ratio * 8)   # 12.5% hit ratio → full signal
            p.tech_depth = _ema(p.tech_depth, tech_signal, alpha=self._alpha)
            p.tech_depth = _clamp(p.tech_depth)

        # ── Example vs theory preference ───────────────────────────────────────
        if _matches(um_text, um_words, cfg.example_signals):
            p.example_bias = _ema(p.example_bias, 1.0, alpha=0.20)
        elif _matches(um_text, um_words, cfg.theory_signals):
            p.example_bias = _ema(p.example_bias, 0.0, alpha=0.16)
        p.example_bias = _clamp(p.example_bias)

        # ── Satisfaction rate ──────────────────────────────────────────────────
        # Short, positive follow-up → satisfied; long follow-up → needed more
        word_count   = len(um.split())
        is_satisfied = (
            word_count <= 4
            and bool(um_words & cfg.satisfied_tokens)
        )
        p.follow_up_rate = _ema(
            p.follow_up_rate,
            1.0 if is_satisfied else 0.0,
            alpha=self._alpha,
        )
        p.follow_up_rate = _clamp(p.follow_up_rate)

        # ── Periodic save ──────────────────────────────────────────────────────
        if p.total_turns % self._save_every == 0:
            self._save()

    def get_style_directive(self) -> str:
        """Return a short system-prompt fragment based on the learned profile.

        Returns ``""`` until enough data has been observed (controlled by
        ``min_confidence``). When non-empty, append this to your system
        prompt *before* each LLM call.

        The directive is intentionally short (1–4 sentences) to minimise
        token cost while guiding response style.
        """
        p = self.profile
        if p.confidence() < self._min_conf:
            return ""

        conf = p.confidence()
        lines: list[str] = []

        # Verbosity
        if p.verbosity < 0.30:
            lines.append(
                "This user prefers concise, to-the-point answers — "
                "skip preamble and keep responses tight."
            )
        elif p.verbosity > 0.70:
            lines.append(
                "This user appreciates detailed explanations — "
                "feel free to elaborate when it adds value."
            )

        # Technical depth
        if conf > 0.30:
            if p.tech_depth > 0.72:
                lines.append(
                    "The user is technically proficient — "
                    "use domain terminology without over-explaining basics."
                )
            elif p.tech_depth < 0.28:
                lines.append(
                    "Prefer plain language over jargon; "
                    "explain technical terms when they are unavoidable."
                )

        # Example vs theory
        if conf > 0.25:
            if p.example_bias > 0.68:
                lines.append(
                    "Where possible, lead with a concrete example or code snippet."
                )
            elif p.example_bias < 0.32:
                lines.append(
                    "Lead with the concept or reasoning; add examples only if needed."
                )

        # Low satisfaction → anticipate the follow-up
        if conf > 0.40 and p.follow_up_rate < 0.30:
            lines.append(
                "This user often asks follow-up questions — "
                "aim for completeness and briefly anticipate the obvious next question."
            )

        if not lines:
            return ""

        header = "[Adaptive style — internal directive, do not repeat this to the user]\n"
        return header + "\n".join(f"• {l}" for l in lines)

    def summary(self) -> dict:
        """Human-readable profile summary for debugging or display."""
        p = self.profile
        return {
            "observations":        p.total_turns,
            "confidence":          f"{p.confidence() * 100:.0f}%",
            "verbosity":           _label(p.verbosity,      "terse",   "balanced", "verbose"),
            "tech_depth":          _label(p.tech_depth,     "plain",   "mixed",    "expert"),
            "learning_style":      _label(p.example_bias,   "theory",  "balanced", "examples"),
            "satisfaction_rate":   f"{p.follow_up_rate * 100:.0f}%",
            "peak_hours":          p.peak_hours,
        }

    def reset(self) -> None:
        """Clear the profile and delete the saved file."""
        self.profile = IntelligenceProfile()
        if self._path.exists():
            self._path.unlink()


# ── Internal helpers ───────────────────────────────────────────────────────────

def _ema(current: float, new_val: float, alpha: float) -> float:
    """Exponential moving average — recent observations weighted more."""
    return current * (1 - alpha) + new_val * alpha


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _top_hours(counts: dict, n: int = 3) -> list:
    if not counts:
        return []
    return [h for h, _ in sorted(counts.items(), key=lambda x: -x[1])[:n]]


def _label(value: float, low: str, mid: str, high: str) -> str:
    if value < 0.35:
        return low
    if value > 0.65:
        return high
    return mid


def _matches(text: str, tokens: set, signals: frozenset) -> bool:
    """Check if any signal in the set appears in the text or token set."""
    for sig in signals:
        if " " in sig:           # multi-word: substring match
            if sig in text:
                return True
        else:                    # single word: exact token match
            if sig in tokens:
                return True
    return False
