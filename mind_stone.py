"""Mind Stone — Adaptive Communication Profile.  v1.3.0

Learns *how* a user communicates, not *what* they say.
Builds a quantified style profile from signal detection and EMA updates.
Generates a short directive injected into the system prompt — shaping
tone, depth, and format without the user ever configuring anything.

Profile dimensions:
  verbosity         — preferred response length (0=terse, 1=detailed)
  verbosity_tech    — verbosity for technical topics (v1.3)
  verbosity_general — verbosity for general topics (v1.3)
  tech_depth        — technical vocabulary density (0=plain, 1=expert)
  example_bias      — examples-first vs theory-first (0=theory, 1=example)
  follow_up_rate    — satisfaction with first reply (0=always satisfied)
  peak_hours        — most active hours
  total_turns       — observations recorded (used for confidence scoring)

v1.3 changes:
  - Topic-conditional verbosity: technical and general content tracked separately
  - Explicit override: "keep it short" / "more detail" set the value directly
    instead of nudging it via EMA

How it works:
  - Call observe(user_msg, assistant_msg) after every conversation turn
  - EMA (alpha=0.12) updates the profile slowly — resistant to one-off turns
  - Directives activate after ~12 turns of observation
  - Returned directive is 1-3 sentences, for silent injection into system prompt

Persistence:
  .mind_stone.json — profile survives application restarts

Language:
  Ships with English signal sets. For other languages, override the module-level
  frozensets or pass a normalise_fn to the constructor. See signals_turkish.py
  for a complete Turkish reference.

Quick start:
    from mind_stone import MindStone

    stone = MindStone()

    # After every conversation turn:
    stone.observe(user_message, assistant_message)

    # Before every LLM call:
    directive = stone.get_style_directive()   # "" until ~12 turns
    if directive:
        system_prompt += "\\n\\n" + directive
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional


# ── Persistence ────────────────────────────────────────────────────────────────

_PROFILE_PATH = Path(".mind_stone.json")
_EMA_ALPHA    = 0.12   # learning rate — slow and stable
_MIN_TURNS    = 5      # no directive until this many observations


# ── English signal sets ───────────────────────────────────────────────────────

# NOTE: All signals are compared against lowercased, optionally normalised text.
# Provide pre-lowercased strings. See signals_turkish.py for non-ASCII example.

# User wants a shorter response → verbosity--
_NEG_VERBOSITY = frozenset({
    "too long", "keep it short", "shorter", "shorten", "cut it",
    "too verbose", "skip that", "way too long", "summarise",
    "summarize", "be brief", "just tell me", "enough", "stop",
    "don't need that", "too much",
})

# User wants more detail → verbosity++
_POS_VERBOSITY = frozenset({
    "more detail", "go deeper", "elaborate", "expand on that",
    "explain more", "tell me more", "keep going", "continue",
    "go on", "more context", "full explanation", "in depth",
    "in detail", "detailed", "how exactly", "what do you mean",
    "say more",
})

# User requests a concrete example → example_bias++
_EXAMPLE_SIGNALS = frozenset({
    "example", "show me", "how would that look", "how do you do it",
    "give me an example", "for instance", "code example", "demo",
    "in practice", "concrete", "illustrate", "sample",
})

# User wants theoretical explanation → example_bias--
_THEORY_SIGNALS = frozenset({
    "why", "how does it work", "what's the logic", "what's behind it",
    "why is that", "what is it for", "the principle", "the concept",
    "the idea", "the reason", "what's the purpose", "fundamentally",
})

# Short positive reply → follow_up_rate signal
_SATISFIED_TOKENS = frozenset({
    "ok", "got it", "understood", "thanks", "perfect", "great",
    "makes sense", "clear", "good", "nice", "cool", "alright",
    "yep", "yes", "sure", "cheers", "thank you", "awesome", "noted",
})

# Technical vocabulary — used to detect tech-heavy messages
_TECH_WORDS = frozenset({
    # Programming languages
    "python", "javascript", "typescript", "rust", "golang", "java",
    "kotlin", "swift", "cpp", "csharp", "ruby", "php", "scala",
    "sql", "bash", "powershell",
    # Core concepts
    "api", "rest", "graphql", "websocket", "async", "await", "thread",
    "queue", "class", "function", "method", "object", "array", "dict",
    "json", "xml", "yaml", "regex", "token", "stream", "buffer",
    # Hardware / ML
    "gpu", "cuda", "cpu", "ram", "embedding", "vector", "model",
    "inference", "rag", "prompt", "llm", "transformer", "gradient",
    "neural", "backprop", "layer",
    # Infrastructure
    "docker", "kubernetes", "git", "linux", "nginx", "redis",
    "aws", "gcp", "azure", "postgres", "mongodb",
    # Tools
    "vscode", "pycharm", "vim", "neovim",
})


# ── Profile dataclass ─────────────────────────────────────────────────────────

@dataclass
class IntelligenceProfile:
    """Quantified model of a user's communication style."""

    verbosity:          float      = 0.50   # 0=terse, 1=detailed (global)
    verbosity_tech:     float      = 0.50   # verbosity for technical topics (v1.3)
    verbosity_general:  float      = 0.50   # verbosity for general topics (v1.3)
    tech_depth:         float      = 0.50   # 0=plain language, 1=expert vocabulary
    example_bias:       float      = 0.50   # 0=theory-first, 1=example-first
    follow_up_rate:     float      = 0.50   # 0=always satisfied, 1=always follows up

    peak_hours:         list       = field(default_factory=list)
    hour_counts:        dict       = field(default_factory=dict)

    total_turns:        int        = 0
    created_at:         float      = field(default_factory=time.time)
    updated_at:         float      = field(default_factory=time.time)

    def confidence(self) -> float:
        """Profile reliability, 0 to 1. Grows with observations."""
        # Full confidence at 50 turns; ~20% at 5 turns; 0 before that
        return min(1.0, max(0.0, (self.total_turns - _MIN_TURNS) / 45))


# ── Core class ────────────────────────────────────────────────────────────────

class MindStone:
    """Silently calibrates the assistant's communication style.

    Extracts signals from every conversation turn, updates the profile
    via EMA, and generates a compact directive once enough data exists.

    Parameters
    ----------
    path : str | Path
        Profile persistence file. Default ".mind_stone.json".
    user_name : str
        Display name used inside directives. Default "User".
    normalise_fn : callable, optional
        Text normalisation applied before signal matching.
        Use for non-ASCII languages (see signals_turkish.py).
    """

    def __init__(
        self,
        path: Path = _PROFILE_PATH,
        user_name: str = "User",
        normalise_fn: Optional[Callable[[str], str]] = None,
    ) -> None:
        self._path         = Path(path)
        self._user_name    = user_name
        self._normalise_fn = normalise_fn
        self._lock         = threading.Lock()
        self.profile       = self._load()

    # ── Persistence ────────────────────────────────────────────────────────────

    def _load(self) -> IntelligenceProfile:
        if not self._path.exists():
            return IntelligenceProfile()
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            base_verbosity = float(data.get("verbosity", 0.50))
            p = IntelligenceProfile(
                verbosity         = base_verbosity,
                # v1.3 fields — fall back to global verbosity if loading older profile
                verbosity_tech    = float(data.get("verbosity_tech",    base_verbosity)),
                verbosity_general = float(data.get("verbosity_general", base_verbosity)),
                tech_depth        = float(data.get("tech_depth",     0.50)),
                example_bias      = float(data.get("example_bias",   0.50)),
                follow_up_rate    = float(data.get("follow_up_rate", 0.50)),
                peak_hours        = list(data.get("peak_hours",      [])),
                hour_counts       = {int(k): int(v) for k, v in data.get("hour_counts", {}).items()},
                total_turns       = int(data.get("total_turns",      0)),
                created_at        = float(data.get("created_at",     time.time())),
                updated_at        = float(data.get("updated_at",     time.time())),
            )
            return p
        except Exception as e:
            print(f"[MindStone] Could not load profile, starting fresh: {e}", flush=True)
            return IntelligenceProfile()

    def _save(self) -> None:
        try:
            self.profile.updated_at = time.time()
            data = asdict(self.profile)
            self._path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as e:
            print(f"[MindStone] Save error: {e}", flush=True)

    # ── Observation ────────────────────────────────────────────────────────────

    def observe(self, user_msg: str, assistant_msg: str) -> None:
        """Process one conversation turn and update the profile. Thread-safe."""
        with self._lock:
            self._observe_locked(user_msg, assistant_msg)

    def _observe_locked(self, user_msg: str, assistant_msg: str) -> None:
        p = self.profile
        p.total_turns += 1

        um = (user_msg or "").strip().lower()
        if self._normalise_fn is not None:
            try:
                um = self._normalise_fn(um)
            except Exception:
                pass

        # ── Time tracking ─────────────────────────────────────────────────────
        hour = datetime.now().hour
        p.hour_counts[hour] = p.hour_counts.get(hour, 0) + 1
        p.peak_hours = _top_hours(p.hour_counts, n=3)

        # ── Topic detection: technical vs general? (v1.3) ─────────────────────
        um_words = set(re.findall(r"\w+", um))
        tech_ratio = len(um_words & _TECH_WORDS) / max(len(um_words), 1)
        is_technical = tech_ratio >= 0.05   # ~1 tech word per 20 words

        # ── Verbosity signals ─────────────────────────────────────────────────
        _neg_hit = (um_words & _NEG_VERBOSITY) or any(
            s in um for s in _NEG_VERBOSITY if " " in s
        )
        _pos_hit = (um_words & _POS_VERBOSITY) or any(
            s in um for s in _POS_VERBOSITY if " " in s
        )

        if _neg_hit:
            # Explicit override: direct set, no EMA (v1.3)
            p.verbosity = 0.10
            if is_technical:
                p.verbosity_tech    = 0.10
            else:
                p.verbosity_general = 0.10
        elif _pos_hit:
            # Explicit override: direct set, no EMA (v1.3)
            p.verbosity = 0.90
            if is_technical:
                p.verbosity_tech    = 0.90
            else:
                p.verbosity_general = 0.90
        else:
            # Passive: slow EMA update based on message length
            user_word_count = len(um.split())
            length_signal = min(1.0, user_word_count / 15)
            p.verbosity = _ema(p.verbosity, length_signal, alpha=_EMA_ALPHA * 0.5)
            if is_technical:
                p.verbosity_tech    = _ema(p.verbosity_tech,    length_signal, alpha=_EMA_ALPHA * 0.5)
            else:
                p.verbosity_general = _ema(p.verbosity_general, length_signal, alpha=_EMA_ALPHA * 0.5)

        p.verbosity         = _clamp(p.verbosity)
        p.verbosity_tech    = _clamp(p.verbosity_tech)
        p.verbosity_general = _clamp(p.verbosity_general)

        # ── Technical depth ───────────────────────────────────────────────────
        all_words = set(re.findall(r"\w+", um))
        if all_words:
            tr = len(all_words & _TECH_WORDS) / max(len(all_words), 1)
            tech_signal = min(1.0, tr * 8)   # 0.125 ratio → full signal
            p.tech_depth = _ema(p.tech_depth, tech_signal, alpha=_EMA_ALPHA)
            p.tech_depth = _clamp(p.tech_depth)

        # ── Example vs theory preference ──────────────────────────────────────
        if any(sig in um for sig in _EXAMPLE_SIGNALS):
            p.example_bias = _ema(p.example_bias, 1.0, alpha=0.18)
        elif any(sig in um for sig in _THEORY_SIGNALS):
            p.example_bias = _ema(p.example_bias, 0.0, alpha=0.15)
        p.example_bias = _clamp(p.example_bias)

        # ── Satisfaction rate ─────────────────────────────────────────────────
        um_token_count = len(um.split())
        is_satisfied = (
            um_token_count <= 4
            and bool(set(re.findall(r"\w+", um)) & _SATISFIED_TOKENS)
        )
        satisfaction_signal = 1.0 if is_satisfied else 0.0
        p.follow_up_rate = _ema(p.follow_up_rate, satisfaction_signal, alpha=_EMA_ALPHA)
        p.follow_up_rate = _clamp(p.follow_up_rate)

        # ── Persist every 5 turns (reduce I/O) ───────────────────────────────
        if p.total_turns % 5 == 0:
            self._save()

    # ── Directive generation ──────────────────────────────────────────────────

    def get_style_directive(self) -> str:
        """Return a short style directive based on the current profile. Thread-safe.

        Returns an empty string until enough data exists (~12 turns).
        Intended for silent injection into the system prompt.
        """
        with self._lock:
            return self._style_directive_locked()

    def _style_directive_locked(self) -> str:
        p = self.profile
        if p.confidence() < 0.15:   # ~12 turns minimum
            return ""

        lines: list = []
        conf = p.confidence()
        _n = self._user_name

        # ── Verbosity — topic-conditional when gap is meaningful (v1.3) ───────
        tech_gen_gap = abs(p.verbosity_tech - p.verbosity_general)
        if tech_gen_gap > 0.25 and conf > 0.20:
            if p.verbosity_tech < 0.35 and p.verbosity_general > 0.55:
                lines.append(
                    f"{_n} wants concise answers on technical topics; "
                    "prefers more explanation in general conversation."
                )
            elif p.verbosity_tech > 0.65 and p.verbosity_general < 0.45:
                lines.append(
                    f"{_n} wants detailed explanations on technical topics; "
                    "keep general conversation shorter."
                )
            else:
                if p.verbosity < 0.30:
                    lines.append(
                        f"{_n} prefers concise answers — get to the point, skip the preamble."
                    )
                elif p.verbosity > 0.70:
                    lines.append(
                        f"{_n} appreciates detailed explanations — "
                        "don't hold back when depth is warranted."
                    )
        else:
            if p.verbosity < 0.30:
                lines.append(
                    f"{_n} prefers concise answers — get to the point, skip the preamble."
                )
            elif p.verbosity > 0.70:
                lines.append(
                    f"{_n} appreciates detailed explanations — "
                    "don't hold back when depth is warranted."
                )

        # Tech depth
        if p.tech_depth > 0.72 and conf > 0.30:
            lines.append(
                f"You can use technical terms without explaining them — "
                f"{_n} is proficient in these areas."
            )
        elif p.tech_depth < 0.28 and conf > 0.30:
            lines.append(
                "Use plain language instead of technical jargon; "
                "support with simple examples when needed."
            )

        # Example bias
        if p.example_bias > 0.68 and conf > 0.25:
            lines.append(
                "Illustrate with a concrete example or code snippet whenever possible."
            )
        elif p.example_bias < 0.32 and conf > 0.25:
            lines.append(
                "Explain the reasoning first; only provide an example if necessary."
            )

        # Low satisfaction → user often needs more
        if p.follow_up_rate < 0.30 and conf > 0.40:
            lines.append(
                f"{_n} often asks follow-up questions — keep your answer a bit more complete "
                "and briefly touch on the likely next question."
            )

        if not lines:
            return ""

        header = (
            f"[COMMUNICATION PROFILE for {_n.upper()}"
            " — internal directive, do not repeat this]\n"
        )
        return header + "\n".join(f"- {l}" for l in lines)

    # ── Status ────────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Human-readable profile snapshot."""
        p = self.profile
        return {
            "total_observations": p.total_turns,
            "confidence":         f"{p.confidence()*100:.0f}%",
            "verbosity":          _label(p.verbosity,         "terse",   "balanced", "detailed"),
            "verbosity_tech":     _label(p.verbosity_tech,    "terse",   "balanced", "detailed"),
            "verbosity_general":  _label(p.verbosity_general, "terse",   "balanced", "detailed"),
            "tech_depth":         _label(p.tech_depth,        "plain",   "mixed",    "expert"),
            "example_bias":       _label(p.example_bias,      "theory-first", "balanced", "example-first"),
            "follow_up_rate":     f"{p.follow_up_rate*100:.0f}%",
            "peak_hours":         p.peak_hours,
        }

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset the profile and clear the persistence file."""
        with self._lock:
            self.profile = IntelligenceProfile()
            self._save()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ema(current: float, new_val: float, alpha: float = _EMA_ALPHA) -> float:
    """Exponential moving average — recent observations weighted higher."""
    return current * (1 - alpha) + new_val * alpha


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _top_hours(counts: dict, n: int = 3) -> list:
    """Return the n hours with the most activity."""
    if not counts:
        return []
    return [h for h, _ in sorted(counts.items(), key=lambda x: -x[1])[:n]]


def _label(value: float, low: str, mid: str, high: str) -> str:
    if value < 0.35:
        return low
    if value > 0.65:
        return high
    return mid


# ── Singleton ─────────────────────────────────────────────────────────────────

_instance: Optional[MindStone] = None


def get_mind_stone(path: str = ".mind_stone.json", user_name: str = "User") -> MindStone:
    """Return the global MindStone instance (lazy init)."""
    global _instance
    if _instance is None:
        _instance = MindStone(path=path, user_name=user_name)
    return _instance
