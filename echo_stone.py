"""
Echo Stone — Comprehension Pattern Detector.  v1.1.0
=====================================================
The companion module to Mind Stone.
Mind Stone teaches *how to speak*. Echo Stone answers *whether it worked*.

Analyses the user's reaction to each response and detects comprehension
patterns that neither user nor assistant would explicitly flag. Translates
these into a directive that shapes *how* the assistant explains — not just
what it says.

Detected patterns:
  - "ok got it" then asks the same thing two turns later  → false confirmation
  - Rephrases the same question three different ways       → genuine confusion
  - Long response followed by a one-word reply             → cognitive overload
  - Digs deeper after an explanation                       → real understanding

Profile dimensions:
  comprehension_rate   0=rarely understands first try, 1=always gets it
  false_confirm_rate   0=confirmations genuine, 1=often confirms without understanding
  overload_rate        0=handles complexity well, 1=easily overwhelmed
  depth_rate           0=stays surface level, 1=digs deeper every time

v1.1 changes:
  - 5-turn rolling window (was 2) for overload detection
  - Window-average overload: sustained long responses + deflection token triggers
  - Deflection token set (_deflection_tokens) for short non-signal replies

Quick start:
    from echo_stone import EchoStone

    stone = EchoStone()

    # Same interface as Mind Stone:
    stone.observe(user_message, assistant_message)

    # Before every LLM call:
    directive = stone.get_comprehension_directive()   # "" until ~8 turns
    if directive:
        system_prompt += "\\n\\n" + directive
"""

from __future__ import annotations

__version__ = "1.1.0"

import json
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional


# ── Signal configuration ──────────────────────────────────────────────────────

@dataclass
class EchoConfig:
    """Language-specific signal sets.

    All strings must be pre-normalised to match the output of normalise_fn.
    See signals_turkish.py for a Turkish reference implementation.
    """
    confusion_signals:   frozenset  # "I don't understand", "say that again"
    confirmation_tokens: frozenset  # "ok", "got it", "understood"
    deepen_signals:      frozenset  # "so that means", "what if", "building on that"
    normalise_fn:        Optional[Callable[[str], str]] = None


# ── English default ───────────────────────────────────────────────────────────

EN_CONFIG = EchoConfig(
    confusion_signals = frozenset({
        "i dont understand", "i don't understand", "didn't get it",
        "dont get it", "what do you mean", "what does that mean",
        "can you repeat", "say that again", "explain again",
        "explain differently", "simpler", "in simpler terms",
        "i'm lost", "im lost", "confused", "makes no sense",
        "what", "huh",
    }),
    confirmation_tokens = frozenset({
        "ok", "got it", "makes sense", "understood", "thanks",
        "thank you", "perfect", "great", "clear", "alright",
        "sure", "yep", "yes", "nice", "cool",
    }),
    deepen_signals = frozenset({
        "so that means", "so if", "what if", "does that mean",
        "in that case", "building on that", "following that logic",
        "so then", "and what about", "what about", "one more question",
        "to take it further", "going deeper",
    }),
    normalise_fn = None,
)

# ── Turkish config (also included for reference) ──────────────────────────────

_TR_MAP = str.maketrans("çğışöüÇĞİŞÖÜ", "cgisouCGISOu")

def _norm_tr(text: str) -> str:
    return text.translate(_TR_MAP)

TR_CONFIG = EchoConfig(
    confusion_signals = frozenset({
        "anlamadim", "anlayamadim", "anlasilmadi", "anlamiyorum",
        "tekrar", "tekrar anlatir misin", "tekrar aciklar misin",
        "baska turlu", "farkli anlatir misin", "daha basit",
        "ne demek", "ne anlama geliyor", "kafam karisti",
        "neden boyle", "nasil yani", "yani ne",
        "peki ama", "ama nasil", "ama neden",
    }),
    confirmation_tokens = frozenset({
        "tamam", "anladim", "ok", "oldu", "tamamdir", "tmm",
        "peki", "iyi", "guzel", "super", "harika", "anlasild",
        "mantikli", "mantikli geldi", "evet", "yes",
    }),
    deepen_signals = frozenset({
        "peki ya", "ya da", "bir de", "ya su durumda",
        "yani demek ki", "demek ki", "o zaman",
        "su anlama mi geliyor", "soyle mi anlayacagiz",
        "bu da mi", "bu durum icin de", "benzer sekilde",
        "mantikli cunku", "mantikli, cunku",
        "daha ileri gidersek", "daha da",
        "onu da sorayim", "bir sorum daha",
        "peki o zaman",
    }),
    normalise_fn = _norm_tr,
)


# ── Stop words ────────────────────────────────────────────────────────────────

_EN_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "i", "you", "he", "she", "it", "we", "they", "my", "your",
    "this", "that", "what", "how", "why", "when", "where", "which",
    "and", "or", "but", "for", "not", "so", "at", "in", "on", "to",
    "of", "with", "can", "could", "just", "also", "very",
})

_TR_STOPWORDS = frozenset({
    "bir", "bu", "su", "o", "ve", "ile", "icin", "de", "da", "ki",
    "mi", "mu", "ama", "cunku", "ise", "ya", "veya",
    "ne", "gibi", "kadar", "daha", "en", "cok", "az", "hic", "her",
    "ben", "sen", "biz", "siz", "bana", "sana", "beni", "seni",
    "onu", "onlari", "bunu", "bizi", "sizin",
    "nasil", "neden", "nerede", "ne", "hangi", "kim",
    "evet", "hayir", "tamam", "ok", "iyi",
})


# ── Profile dataclass ─────────────────────────────────────────────────────────

@dataclass
class ComprehensionProfile:
    """Numerical model of how a user comprehends explanations.

    All float values are updated via EMA, clamped to [0.0, 1.0].
    """

    comprehension_rate: float = 0.50  # 0=rarely gets it first try, 1=always does
    false_confirm_rate: float = 0.50  # 0=confirmations genuine, 1=often false
    overload_rate:      float = 0.50  # 0=handles complexity, 1=easily overwhelmed
    depth_rate:         float = 0.50  # 0=stays surface, 1=digs deeper every time

    total_turns:  int   = 0
    created_at:   float = field(default_factory=time.time)
    updated_at:   float = field(default_factory=time.time)

    def confidence(self) -> float:
        """0 to 1 reliability score. Active around 8 turns, full around 40."""
        return min(1.0, max(0.0, (self.total_turns - 3) / 37))

    def as_dict(self) -> dict:
        return asdict(self)


# ── Core class ────────────────────────────────────────────────────────────────

class EchoStone:
    """Comprehension pattern detector.

    Same interface as Mind Stone: observe(user_msg, assistant_msg).
    Internally tracks the previous turn as well — real analysis happens
    on the turn after, by examining how the user responded.

    Thread-safe.

    Parameters
    ----------
    path : str | Path
        Profile persistence file. Default ".echo_stone.json".
    config : EchoConfig
        Language signal sets. Default EN_CONFIG (English).
    ema_alpha : float
        Learning rate. Default 0.15.
    min_confidence : float
        Threshold before directives activate. Default 0.12 (~8 turns).
    save_every : int
        Persist to disk every N turns.
    rephrase_threshold : float
        Jaccard similarity score above which a message is a rephrase.
    overload_word_count : int
        Response word count that triggers overload checking.
    """

    def __init__(
        self,
        path:                str | Path  = ".echo_stone.json",
        config:              EchoConfig  = EN_CONFIG,
        ema_alpha:           float       = 0.15,
        min_confidence:      float       = 0.12,
        save_every:          int         = 5,
        rephrase_threshold:  float       = 0.38,
        overload_word_count: int         = 120,
    ) -> None:
        self._path               = Path(path)
        self._config             = config
        self._alpha              = ema_alpha
        self._min_conf           = min_confidence
        self._save_every         = save_every
        self._rephrase_threshold = rephrase_threshold
        self._overload_wc        = overload_word_count
        self._lock               = threading.Lock()
        self.profile             = self._load()

        # Multi-turn rolling window (v1.1: 2 → 5 turns) — not persisted
        _win = 5
        self._user_window:      list = []   # normalised, last 5 turns
        self._assist_window:    list = []   # raw, last 5 turns
        self._topic_window:     list = []   # content-word frozensets
        self._window_size:      int  = _win

        # Single-turn backward references
        self._prev_user:        Optional[str]       = None
        self._prev_assistant:   Optional[str]       = None
        self._prev_was_confirm: bool                = False
        self._prev_topic_words: frozenset           = frozenset()
        self._confirmed_topic:  frozenset           = frozenset()

        # Overload deflection tokens (v1.1)
        self._deflection_tokens: frozenset = frozenset({
            "ok", "hmm", "neyse", "tamam", "anladim", "peki",
            "whatever", "sure", "fine", "got it", "okay",
        })

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> ComprehensionProfile:
        if not self._path.exists():
            return ComprehensionProfile()
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return ComprehensionProfile(
                comprehension_rate = float(data.get("comprehension_rate", 0.50)),
                false_confirm_rate = float(data.get("false_confirm_rate", 0.50)),
                overload_rate      = float(data.get("overload_rate",      0.50)),
                depth_rate         = float(data.get("depth_rate",         0.50)),
                total_turns        = int(data.get("total_turns",          0)),
                created_at         = float(data.get("created_at",         time.time())),
                updated_at         = float(data.get("updated_at",         time.time())),
            )
        except Exception:
            return ComprehensionProfile()

    def _save(self) -> None:
        """Called while lock is held."""
        try:
            self.profile.updated_at = time.time()
            self._path.write_text(
                json.dumps(self.profile.as_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            print(f"[EchoStone] Save error: {exc}")

    # ── Public API ────────────────────────────────────────────────────────────

    def observe(
        self,
        user_message:      str,
        assistant_message: str,
        verbose:           bool = False,
    ) -> Optional[dict]:
        """Process one conversation turn and update the comprehension profile.

        Same signature as Mind Stone — call after every user/assistant exchange.
        Internally, the real analysis runs on the turn *after*, by examining
        how the user reacted to the previous assistant response.

        verbose=True returns a dict with detected signal and profile snapshot.
        """
        with self._lock:
            return self._observe_locked(user_message, assistant_message, verbose)

    def _observe_locked(
        self,
        user_message:      str,
        assistant_message: str,
        verbose:           bool,
    ) -> Optional[dict]:
        p   = self.profile
        cfg = self._config

        # ── Normalise text ─────────────────────────────────────────────────────
        um_raw = (user_message or "").strip().lower()
        um = um_raw
        if callable(cfg.normalise_fn):
            try:
                um = cfg.normalise_fn(um)
            except Exception:
                pass

        um_words = frozenset(re.findall(r"\w+", um))
        stopwords = _TR_STOPWORDS if cfg.normalise_fn else _EN_STOPWORDS
        um_content_words = um_words - stopwords

        # ── Signal detection (needs a previous turn) ──────────────────────────
        signal: Optional[str] = None

        if self._prev_user is not None:
            # 1. Explicit confusion — highest priority
            if _matches(um, um_words, cfg.confusion_signals):
                p.comprehension_rate = _ema(p.comprehension_rate, 0.0, 0.25)
                p.overload_rate      = _ema(p.overload_rate,      1.0, 0.18)
                signal = "explicit_confusion"

            # 2. Cognitive overload: long response → short/deflection reply (v1.1)
            # Condition A: immediately previous response was long + reply is short
            # Condition B: rolling window average long + deflection token present
            _assist_long   = (self._prev_assistant and
                              len(self._prev_assistant.split()) > self._overload_wc)
            _reply_short   = len(um.split()) <= 3
            _has_deflect   = bool(um_words & self._deflection_tokens)
            _window_avg_long = (
                len(self._assist_window) >= 3 and
                sum(len(a.split()) for a in self._assist_window[-3:]) / 3
                > self._overload_wc * 0.7
            )
            if (_assist_long and _reply_short) or (
                _window_avg_long and _has_deflect and _reply_short
            ):
                p.overload_rate = _ema(p.overload_rate, 1.0, 0.20)
                signal = "overload_deflect"

            # 3. Deepening → genuine understanding
            elif _matches(um, um_words, cfg.deepen_signals):
                p.comprehension_rate = _ema(p.comprehension_rate, 1.0, 0.20)
                p.depth_rate         = _ema(p.depth_rate,         1.0, 0.22)
                signal = "deepening"

            # 4. Rephrase detection
            elif _rephrase_score(um_content_words, self._prev_topic_words) >= self._rephrase_threshold:
                p.comprehension_rate = _ema(p.comprehension_rate, 0.0, 0.18)
                signal = "rephrase"
                if self._prev_was_confirm:
                    p.false_confirm_rate = _ema(p.false_confirm_rate, 1.0, 0.28)
                    signal = "false_confirmation"

            # 5. Post-confirmation topic overlap → soft false confirm
            elif self._prev_was_confirm:
                overlap = _rephrase_score(um_content_words, self._confirmed_topic)
                if overlap >= 0.18:
                    p.false_confirm_rate = _ema(p.false_confirm_rate, 1.0, 0.22)
                    p.comprehension_rate = _ema(p.comprehension_rate, 0.0, 0.14)
                    signal = "false_confirmation_soft"
                else:
                    # Genuine confirmation — moved to a different topic
                    p.false_confirm_rate = _ema(p.false_confirm_rate, 0.0, 0.14)
                    p.comprehension_rate = _ema(p.comprehension_rate, 1.0, 0.14)
                    signal = "genuine_confirmation"

            # 6. Neutral continuation
            else:
                p.comprehension_rate = _ema(p.comprehension_rate, 0.65, 0.07)
                signal = "neutral"

            # Clamp all values
            p.comprehension_rate = _clamp(p.comprehension_rate)
            p.false_confirm_rate = _clamp(p.false_confirm_rate)
            p.overload_rate      = _clamp(p.overload_rate)
            p.depth_rate         = _clamp(p.depth_rate)

        # ── Update state for next turn ────────────────────────────────────────
        wc = len(um.split())
        is_confirm = (
            wc <= 6
            and bool(um_words & cfg.confirmation_tokens)
            and not um.rstrip().endswith("?")
            and not bool(um_words & frozenset({"but", "however", "ama", "peki"}))
        )
        if is_confirm:
            self._confirmed_topic = self._prev_topic_words
        else:
            self._confirmed_topic = frozenset()

        self._prev_was_confirm  = is_confirm
        self._prev_user         = um
        self._prev_assistant    = assistant_message
        self._prev_topic_words  = um_content_words

        # ── Update 5-turn rolling window (v1.1) ──────────────────────────────
        self._user_window.append(um)
        self._assist_window.append(assistant_message or "")
        self._topic_window.append(um_content_words)
        if len(self._user_window)   > self._window_size: self._user_window.pop(0)
        if len(self._assist_window) > self._window_size: self._assist_window.pop(0)
        if len(self._topic_window)  > self._window_size: self._topic_window.pop(0)

        # ── Turn counter and save ─────────────────────────────────────────────
        p.total_turns += 1
        if p.total_turns % self._save_every == 0:
            self._save()

        if verbose:
            return {
                "signal":     signal,
                "is_confirm": is_confirm,
                "profile":    self._summary_locked(),
            }
        return None

    def get_comprehension_directive(self) -> str:
        """Return a comprehension directive for injection into the system prompt.

        Returns an empty string until enough data exists (min_confidence threshold).
        """
        with self._lock:
            p = self.profile
            if p.confidence() < self._min_conf:
                return ""

            conf  = p.confidence()
            lines: list = []

            # High false confirmation rate
            if p.false_confirm_rate > 0.58 and conf > 0.20:
                lines.append(
                    "This user sometimes says they understand when they don't. "
                    "After complex explanations, add a brief check-in "
                    "(e.g. 'Does that make sense?' or a one-sentence recap)."
                )

            # Low comprehension rate
            if p.comprehension_rate < 0.35 and conf > 0.25:
                lines.append(
                    "This user usually needs more than one explanation. "
                    "Use concrete analogies, avoid abstract descriptions, "
                    "and break multi-step answers into numbered steps."
                )

            # Cognitive overload
            if p.overload_rate > 0.62 and conf > 0.20:
                lines.append(
                    "Long responses overwhelm this user. "
                    "Keep answers short and focused; let the user ask for more. "
                    "Don't try to cover everything in one go."
                )

            # Deep learner
            if p.depth_rate > 0.65 and conf > 0.30:
                lines.append(
                    "This user digs deeper after every explanation — you can give "
                    "dense, information-rich answers. Preemptively address the "
                    "obvious next question."
                )

            if not lines:
                return ""

            header = "[Comprehension guide — internal directive, do not repeat this to the user]\n"
            return header + "\n".join(f"* {l}" for l in lines)

    def summary(self) -> dict:
        """Human-readable profile snapshot. Thread-safe."""
        with self._lock:
            return self._summary_locked()

    def _summary_locked(self) -> dict:
        p = self.profile
        return {
            "version":            __version__,
            "observations":       p.total_turns,
            "confidence":         f"{p.confidence() * 100:.0f}%",
            "comprehension_rate": _label(p.comprehension_rate, "weak",     "moderate", "strong"),
            "false_confirm_rate": _label(p.false_confirm_rate, "rare",     "moderate", "frequent"),
            "overload_rate":      _label(p.overload_rate,      "resilient","moderate", "overloads easily"),
            "depth_rate":         _label(p.depth_rate,         "surface",  "moderate", "deep"),
        }

    def reset(self) -> None:
        """Reset the profile and clear the persistence file. Thread-safe."""
        with self._lock:
            self.profile            = ComprehensionProfile()
            self._prev_user         = None
            self._prev_assistant    = None
            self._prev_was_confirm  = False
            self._prev_topic_words  = frozenset()
            self._confirmed_topic   = frozenset()
            if self._path.exists():
                self._path.unlink()


# ── Singleton ─────────────────────────────────────────────────────────────────

_echo_stone_instance: Optional[EchoStone] = None
_echo_stone_lock = threading.Lock()


def get_echo_stone(
    path: str = ".echo_stone.json",
    config: EchoConfig = EN_CONFIG,
) -> EchoStone:
    """Return the global EchoStone instance (lazy init, thread-safe)."""
    global _echo_stone_instance
    if _echo_stone_instance is None:
        with _echo_stone_lock:
            if _echo_stone_instance is None:
                _echo_stone_instance = EchoStone(path=path, config=config)
    return _echo_stone_instance


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ema(current: float, new_val: float, alpha: float) -> float:
    return current * (1 - alpha) + new_val * alpha


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _rephrase_score(words_a: frozenset, words_b: frozenset) -> float:
    """Jaccard similarity — overlap between two content-word sets."""
    if not words_a or not words_b:
        return 0.0
    intersection = len(words_a & words_b)
    union        = len(words_a | words_b)
    return intersection / union if union else 0.0


def _matches(text: str, tokens: frozenset, signals: frozenset) -> bool:
    """Check whether any signal appears in the text or token set."""
    for sig in signals:
        if " " in sig:
            if sig in text:
                return True
        else:
            if sig in tokens:
                return True
    return False


def _label(value: float, low: str, mid: str, high: str) -> str:
    if value < 0.35:
        return low
    if value > 0.65:
        return high
    return mid
