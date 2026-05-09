"""
Bond Stone — Persistent User World Model.  v1.1.0

Builds a structured knowledge graph of who the user is: their projects,
tech stack, constraints, and preferences. Persists across sessions as a
single JSON file. Zero dependencies beyond the standard library. Thread-safe.

The difference between Bond Stone and simply saving chat history is the
difference between a notebook and a map. Chat history stores what was said.
Bond Stone builds what it means.

v1.1 changes:
  - Recency decay (exponential, 90-day half-life) for tech stack and facts
  - Context directive orders items by recency-weighted score
  - Facts now carry first_seen / last_seen timestamps
  - v1.0 profiles are migrated transparently on load

Quick start:
    from bond_stone import BondStone

    stone = BondStone()

    # After every conversation turn:
    stone.observe(user_message, assistant_message)

    # Before every LLM call:
    ctx = stone.get_context_directive()
    if ctx:
        system_prompt += "\\n\\n" + ctx

    # Explicit fact (passive extraction works automatically):
    stone.remember("I work on the Vision project using Python and CUDA")

    # Register a shorthand alias:
    stone.alias("usual setup", "Python + CUDA + Ollama on Windows 11")
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set

__version__ = "1.1.0"

# ── Recency decay ──────────────────────────────────────────────────────────────

_DECAY_HALF_LIFE_DAYS = 90.0   # weight halves every 90 days


def _recency_weight(last_seen: float, half_life: float = _DECAY_HALF_LIFE_DAYS) -> float:
    """Exponential decay: 0 days → 1.0, 90 days → 0.5, 180 days → 0.25."""
    days_ago = (time.time() - last_seen) / 86400.0
    return math.exp(-days_ago * math.log(2) / half_life)


def _weighted_score(mentions: int, last_seen: float) -> float:
    """mention_count × recency_weight — used for context ordering and pruning."""
    return mentions * _recency_weight(last_seen)


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BondConfig:
    """
    Signal sets for Bond Stone.

    Pass a custom instance for non-English use (see signals_turkish.py).
    All signal strings must be pre-normalised to match the output of
    ``normalise_fn``.
    """
    tech_words:          frozenset
    constraint_signals:  frozenset
    preference_signals:  frozenset
    remember_signals:    frozenset
    stopwords:           frozenset
    normalise_fn:        Optional[Callable[[str], str]] = None


EN_CONFIG = BondConfig(
    tech_words=frozenset({
        # Languages
        "python", "javascript", "typescript", "rust", "golang", "java",
        "kotlin", "swift", "cpp", "csharp", "ruby", "php", "scala",
        "sql", "bash", "powershell", "html", "css",
        # Frameworks & libraries
        "react", "vue", "angular", "svelte", "nextjs", "nuxt",
        "django", "flask", "fastapi", "express", "rails",
        "pytorch", "tensorflow", "numpy", "pandas", "sklearn",
        # Infrastructure
        "docker", "kubernetes", "git", "linux", "ubuntu", "windows", "macos",
        "aws", "gcp", "azure", "vercel", "netlify", "cloudflare",
        # Databases
        "postgres", "postgresql", "mysql", "mongodb", "redis", "sqlite",
        "elasticsearch", "cassandra",
        # AI / ML
        "ollama", "openai", "anthropic", "gemini", "groq", "mistral",
        "cuda", "llm", "rag", "embedding", "vector", "transformer",
        "diffusion", "whisper",
        # Protocols / concepts
        "api", "rest", "graphql", "websocket", "grpc", "http", "oauth",
        # Tools
        "vscode", "pycharm", "vim", "neovim",
        "raspberry", "arduino", "esp32",
    }),
    constraint_signals=frozenset({
        "can't use", "cannot use", "don't have", "no access",
        "without", "not allowed", "banned", "blocked",
        "limited to", "only have", "restricted to", "stuck with",
        "no internet", "no gpu", "no api",
    }),
    preference_signals=frozenset({
        "i prefer", "i always", "i usually", "i tend to",
        "i like to", "my preference", "i avoid", "i never",
        "always use", "never use", "i hate using", "i love using",
    }),
    remember_signals=frozenset({
        "remember that", "remember this", "note that", "note this",
        "keep in mind", "don't forget", "save this",
        "my name is", "i am a ", "i work at", "i work on",
        "i'm working on", "i'm building", "i'm developing",
    }),
    stopwords=frozenset({
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "i", "it", "in", "on", "at", "to", "for", "of", "with", "by",
        "and", "or", "but", "so", "this", "that", "my", "your", "its",
        "we", "they", "he", "she", "have", "has", "had", "do", "does", "did",
        "use", "using", "used", "can", "will", "would", "could", "should",
        "just", "also", "me", "us", "them", "our", "their",
        "when", "where", "how", "what", "who", "why", "which",
        "very", "really", "quite", "too", "about", "up", "out", "if",
    }),
    normalise_fn=None,
)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _empty_profile() -> dict:
    return {
        "version": 2,
        "turn_count": 0,
        "tech_stack": {},       # tech_word -> mention_count
        "tech_last_seen": {},   # tech_word -> unix_timestamp  (v1.1)
        "facts": [],            # [{"text", "mentions", "type", "first_seen", "last_seen"}]
        "aliases": {},          # "the usual setup" -> "Python + CUDA"
    }


def _content_words(text: str, stopwords: frozenset) -> Set[str]:
    return set(re.findall(r"\b\w{3,}\b", text.lower())) - stopwords


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _first_sentence_from(text: str, start: int) -> str:
    """Return text from start to end of first sentence (or end of string)."""
    chunk = text[start:]
    for sep in (".", "!", "?", "\n"):
        idx = chunk.find(sep)
        if idx != -1 and idx > 4:
            return chunk[: idx + 1].strip()
    return chunk.strip()


# ── Bond Stone ─────────────────────────────────────────────────────────────────

class BondStone:
    """
    Persistent user world model.

    Silently extracts who the user is — their stack, constraints, and
    preferences — and injects a compact context block into every LLM call.
    Data accumulates across sessions in a single JSON file.

    Parameters
    ----------
    path : str
        Path to the JSON persistence file. Default ".bond_stone.json".
    config : BondConfig
        Language signal sets. Default EN_CONFIG (English).
    save_every : int
        Persist to disk every N turns.
    max_facts : int
        Cap on stored facts (oldest/lowest-confidence pruned first).
    """

    def __init__(
        self,
        path:       str       = ".bond_stone.json",
        config:     BondConfig = EN_CONFIG,
        save_every: int        = 5,
        max_facts:  int        = 60,
    ) -> None:
        self._path       = path
        self._cfg        = config
        self._save_every = save_every
        self._max_facts  = max_facts
        self._lock       = threading.Lock()
        self._profile    = self._load()

    # ── Public API ─────────────────────────────────────────────────────────────

    def observe(
        self,
        user_message:      str,
        assistant_message: str,
        verbose:           bool = False,
    ) -> Optional[dict]:
        """
        Process one conversation turn.

        Call after every user/assistant exchange. Returns None unless
        ``verbose=True``, in which case it returns a dict with extraction
        details.
        """
        with self._lock:
            return self._observe_locked(user_message, assistant_message, verbose)

    def remember(self, fact: str, fact_type: str = "explicit") -> None:
        """
        Explicitly record a fact.

        Use this for information provided outside the normal conversation flow,
        or to manually seed the profile.
        """
        with self._lock:
            self._add_fact(fact.strip(), fact_type)
            self._save_locked()

    def alias(self, shorthand: str, expansion: str) -> None:
        """
        Register a shorthand alias.

            stone.alias("usual setup", "Python + CUDA + Ollama on Windows 11")

        When the user says "the usual setup", the assistant will know what
        it means.
        """
        with self._lock:
            self._profile["aliases"][shorthand.lower().strip()] = expansion.strip()
            self._save_locked()

    def resolve(self, query: str) -> Optional[str]:
        """
        Resolve a query against known aliases.

        Returns the expansion if any alias is found in the query string,
        otherwise None.
        """
        with self._lock:
            q = self._normalise(query)
            for shorthand, expansion in self._profile["aliases"].items():
                if shorthand in q:
                    return expansion
            return None

    def get_context_directive(self, topic_hint: str = "") -> str:
        """
        Return a compact context block for injection into the system prompt.

        Pass ``topic_hint`` (typically the current user message) to
        prioritise facts relevant to the current topic.

        Returns an empty string until there is something worth injecting.
        """
        with self._lock:
            return self._context_directive_locked(topic_hint)

    def summary(self) -> dict:
        """Return a snapshot of the current profile."""
        with self._lock:
            return self._summary_locked()

    def reset(self) -> None:
        """Wipe the profile from memory and disk."""
        with self._lock:
            self._profile = _empty_profile()
            if os.path.exists(self._path):
                os.remove(self._path)

    # ── Internals ──────────────────────────────────────────────────────────────

    def _normalise(self, text: str) -> str:
        fn = self._cfg.normalise_fn
        if fn is not None:
            try:
                return fn(text).lower()
            except Exception:
                pass
        return text.lower()

    def _observe_locked(
        self,
        user_message:      str,
        assistant_message: str,
        verbose:           bool,
    ) -> Optional[dict]:
        um_norm = self._normalise(user_message)
        self._profile["turn_count"] += 1
        extracted: List[tuple] = []

        # ── 1. Tech stack ──────────────────────────────────────────────────────
        words = set(re.findall(r"\b\w+\b", um_norm))
        tech_found = words & self._cfg.tech_words
        now = time.time()
        for w in tech_found:
            self._profile["tech_stack"][w] = (
                self._profile["tech_stack"].get(w, 0) + 1
            )
            self._profile["tech_last_seen"][w] = now

        # ── 2. Explicit remember signals ───────────────────────────────────────
        for sig in self._cfg.remember_signals:
            if sig in um_norm:
                idx = um_norm.find(sig)
                fact_text = _first_sentence_from(user_message, idx)
                if len(fact_text) > len(sig) + 5:
                    self._add_fact(fact_text, "explicit")
                    extracted.append(("explicit", fact_text))
                break

        # ── 3. Constraints ─────────────────────────────────────────────────────
        for sig in self._cfg.constraint_signals:
            if sig in um_norm:
                idx = um_norm.find(sig)
                fact_text = _first_sentence_from(user_message, idx)
                if len(fact_text) > len(sig) + 3:
                    self._add_fact(fact_text, "constraint")
                    extracted.append(("constraint", fact_text))
                break

        # ── 4. Preferences ─────────────────────────────────────────────────────
        for sig in self._cfg.preference_signals:
            if sig in um_norm:
                idx = um_norm.find(sig)
                fact_text = _first_sentence_from(user_message, idx)
                if len(fact_text) > len(sig) + 3:
                    self._add_fact(fact_text, "preference")
                    extracted.append(("preference", fact_text))
                break

        # ── Persist ────────────────────────────────────────────────────────────
        if self._profile["turn_count"] % self._save_every == 0:
            self._save_locked()

        if verbose:
            return {
                "turn":       self._profile["turn_count"],
                "tech_found": sorted(tech_found),
                "extracted":  extracted,
                "profile":    self._summary_locked(),
            }
        return None

    def _add_fact(self, text: str, fact_type: str) -> None:
        """Add a fact, merging with an existing one if sufficiently similar."""
        words = _content_words(text, self._cfg.stopwords)
        now = time.time()

        for fact in self._profile["facts"]:
            existing_words = _content_words(fact["text"], self._cfg.stopwords)
            if _jaccard(words, existing_words) >= 0.52:
                fact["mentions"] += 1
                fact["last_seen"] = now
                # Keep the more informative (longer) version
                if len(text) > len(fact["text"]):
                    fact["text"] = text
                return

        self._profile["facts"].append({
            "text":       text,
            "mentions":   1,
            "type":       fact_type,
            "first_seen": now,
            "last_seen":  now,
        })

        # Prune when over cap — drop lowest recency-weighted score
        if len(self._profile["facts"]) > self._max_facts:
            self._profile["facts"].sort(
                key=lambda f: -_weighted_score(f["mentions"], f.get("last_seen", now))
            )
            self._profile["facts"] = self._profile["facts"][: self._max_facts]

    def _context_directive_locked(self, topic_hint: str) -> str:
        tech    = self._profile["tech_stack"]
        facts   = self._profile["facts"]
        aliases = self._profile["aliases"]

        if not tech and not facts and not aliases:
            return ""

        parts = ["User context (Bond Stone):"]

        # Top 8 tech terms by recency-weighted score
        if tech:
            last_seen = self._profile.get("tech_last_seen", {})
            now = time.time()
            top = sorted(
                tech.items(),
                key=lambda x: -_weighted_score(x[1], last_seen.get(x[0], now))
            )[:8]
            parts.append("- Stack: " + ", ".join(t[0].capitalize() for t in top))

        # Registered aliases
        for shorthand, expansion in aliases.items():
            parts.append(f'- "{shorthand}" = {expansion}')

        # Facts — sorted by recency-weighted score, filtered by topic relevance
        if facts:
            now = time.time()
            sorted_facts = sorted(
                facts,
                key=lambda f: -_weighted_score(f["mentions"], f.get("last_seen", now))
            )

            if topic_hint:
                hint_words = _content_words(
                    self._normalise(topic_hint), self._cfg.stopwords
                )
                relevant = [
                    f for f in sorted_facts
                    if _jaccard(
                        hint_words,
                        _content_words(f["text"], self._cfg.stopwords),
                    ) > 0.08
                ]
                others  = [f for f in sorted_facts if f not in relevant]
                ordered = (relevant[:4] + others[:2])[:6]
            else:
                ordered = sorted_facts[:6]

            for fact in ordered:
                label = (
                    "[constraint] " if fact["type"] == "constraint"
                    else "[preference] " if fact["type"] == "preference"
                    else ""
                )
                parts.append(f"- {label}{fact['text']}")

        return "\n".join(parts) if len(parts) > 1 else ""

    def _summary_locked(self) -> dict:
        return {
            "turn_count": self._profile["turn_count"],
            "tech_stack": dict(self._profile["tech_stack"]),
            "fact_count": len(self._profile["facts"]),
            "facts":      list(self._profile["facts"]),
            "aliases":    dict(self._profile["aliases"]),
        }

    # ── Persistence ────────────────────────────────────────────────────────────

    def _load(self) -> dict:
        if os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                p = _empty_profile()
                p.update(data)

                # v1.0 → v1.1 migration ──────────────────────────────────────
                # If tech_last_seen is missing, seed from current time
                # (old data treated as "recent" to avoid immediate decay).
                if not p.get("tech_last_seen"):
                    now = time.time()
                    p["tech_last_seen"] = {w: now for w in p.get("tech_stack", {})}

                # Add first_seen / last_seen to facts that lack them.
                now = time.time()
                for fact in p.get("facts", []):
                    fact.setdefault("first_seen", now)
                    fact.setdefault("last_seen", now)

                p["version"] = 2
                return p
            except Exception:
                pass
        return _empty_profile()

    def _save_locked(self) -> None:
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._profile, f, indent=2, ensure_ascii=False)
        except Exception:
            pass


# ── Turkish config (reference implementation) ─────────────────────────────────

_TR_MAP = str.maketrans("çğışöüÇĞİŞÖÜ", "cgisouCGISOu")

def _norm_tr(text: str) -> str:
    return text.translate(_TR_MAP)

_TR_TECH_WORDS = frozenset({
    "python", "javascript", "typescript", "rust", "golang", "java",
    "sql", "bash", "powershell",
    "api", "async", "await", "thread", "queue", "class", "function",
    "json", "yaml", "regex", "token", "stream", "buffer",
    "gpu", "cuda", "cpu", "ram", "embedding", "vector", "model",
    "inference", "rag", "prompt", "llm",
    "docker", "kubernetes", "git", "linux", "windows",
    "neural", "transformer", "gradient", "ollama", "openai", "gemini",
    "whisper", "pytorch", "numpy", "pandas",
    # Turkish technical terms (diacritics stripped)
    "fonksiyon", "degisken", "dongu", "sinif", "nesne", "dizi",
    "veritabani", "sorgu", "sunucu", "istemci", "protokol",
    "algoritma", "bellek", "islemci", "hata", "debug", "test",
})

TR_CONFIG = BondConfig(
    tech_words=_TR_TECH_WORDS,
    constraint_signals=frozenset({
        "kullanamiyorum", "kullanamiyor", "yok elimde", "erisemiyorum",
        "izin yok", "yasak", "kisitli", "sadece var", "olmadan",
        "internet yok", "gpu yok", "api yok", "erisim yok",
    }),
    preference_signals=frozenset({
        "tercih ederim", "her zaman kullanirim", "genellikle", "hep kullanirim",
        "seviyorum", "sevmiyorum", "kacinirim", "kullanmam",
        "tercihim", "her zaman",
    }),
    remember_signals=frozenset({
        "bunu hatirla", "bunu not et", "aklinda tut", "unutma",
        "kaydet bunu", "adim", "uzerinde calisiyorum", "gelistiriyorum",
        "projemde", "sistemimde", "remember that", "note that",
    }),
    stopwords=frozenset({
        "bir", "bu", "su", "ve", "ile", "ama", "ya", "da", "de",
        "icin", "ben", "sen", "biz", "siz", "var", "yok", "oldu",
        "daha", "cok", "az", "tam", "sadece", "bile", "hic",
        "nasil", "neden", "ne", "nerede", "kim", "hangi",
        "gibi", "gore", "kadar", "beri", "sonra", "once",
        "a", "an", "the", "is", "are", "in", "on", "at", "to",
        "for", "of", "with", "and", "or", "but", "this", "that",
    }),
    normalise_fn=_norm_tr,
)


# ── Singleton ─────────────────────────────────────────────────────────────────

import threading as _threading

_bond_stone_instance: Optional[BondStone] = None
_bond_stone_lock = _threading.Lock()


def get_bond_stone(
    path: str = ".bond_stone.json",
    config: BondConfig = EN_CONFIG,
) -> BondStone:
    """Return the global BondStone instance (lazy init, thread-safe)."""
    global _bond_stone_instance
    if _bond_stone_instance is None:
        with _bond_stone_lock:
            if _bond_stone_instance is None:
                _bond_stone_instance = BondStone(path=path, config=config)
    return _bond_stone_instance
