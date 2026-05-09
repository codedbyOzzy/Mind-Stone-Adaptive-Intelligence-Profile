"""
Intuition Stone — Conversation Arc Predictor.  v1.1.0

Learns which topics consistently lead to which follow-up topics. After
enough conversations, it can predict where this one is going before the
user has finished asking — and prompt the assistant to address it first.

A capable human assistant doesn't just answer the question in front of them.
They recognise patterns. Intuition Stone gives an AI that same capability.

v1.1 changes:
  - 2-step chain prediction: tracks A→B→C topic transitions (stored with _2_ prefix)
  - Recency timestamps: word_follows entries carry last_seen timestamps
  - Transition expiry: edges not seen in 90 days are pruned automatically
  - v1.0 profiles are migrated transparently on load

Quick start:
    from intuition_stone import IntuitionStone

    stone = IntuitionStone()

    # After every conversation turn:
    stone.observe(user_message, assistant_message)

    # Before every LLM call:
    hint = stone.get_prediction_directive(current_user_message)
    if hint:
        system_prompt += "\\n\\n" + hint
"""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, List, Optional, Set, Tuple

__version__ = "1.1.0"


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class IntuitionConfig:
    """
    Signal sets for Intuition Stone.

    Pass a custom instance for non-English use (see signals_turkish.py).
    """
    stopwords:    frozenset
    normalise_fn: Optional[Callable[[str], str]] = None


EN_CONFIG = IntuitionConfig(
    stopwords=frozenset({
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "i", "it", "in", "on", "at", "to", "for", "of", "with", "by",
        "and", "or", "but", "so", "this", "that", "my", "your", "its",
        "we", "they", "he", "she", "have", "has", "had", "do", "does", "did",
        "use", "using", "used", "can", "will", "would", "could", "should",
        "just", "also", "me", "us", "them", "our", "their",
        "when", "where", "how", "what", "who", "why", "which",
        "very", "really", "quite", "too", "about", "up", "out", "if",
        "get", "make", "need", "want", "know", "think", "try", "like",
        "now", "then", "here", "there", "way", "time", "work",
    }),
    normalise_fn=None,
)


# ── Internal helpers ───────────────────────────────────────────────────────────

_TOPIC_SIZE = 5    # max content words per topic vector
_MIN_WORD_LEN = 3  # minimum word length to count as a topic word

_TRANSITION_EXPIRE_DAYS = 90   # edges older than this are pruned (v1.1)


def _empty_profile() -> dict:
    return {
        "version": 2,
        "turn_count": 0,
        "transitions": {},       # topic_key -> {next_topic_key: count}
        "word_follows": {},      # word -> {word: count}
        "word_last_seen": {},    # word -> {word: unix_timestamp}  (v1.1)
    }


def _topic_key(words: FrozenSet[str]) -> str:
    return "|".join(sorted(words))


def _top_words(
    text: str,
    stopwords: frozenset,
    normalise_fn: Optional[Callable[[str], str]],
    n: int = _TOPIC_SIZE,
) -> FrozenSet[str]:
    """Extract the N most informative content words from text."""
    normalised = normalise_fn(text).lower() if normalise_fn else text.lower()
    words = [
        w for w in re.findall(r"\b\w{%d,}\b" % _MIN_WORD_LEN, normalised)
        if w not in stopwords
    ]
    # Rank by length as a proxy for specificity (longer = more specific)
    words.sort(key=len, reverse=True)
    return frozenset(words[:n])


# ── Intuition Stone ────────────────────────────────────────────────────────────

class IntuitionStone:
    """
    Conversation arc predictor.

    Tracks which topics consistently follow which other topics across all
    conversations with this user. When a strong pattern emerges, injects a
    directive so the assistant can address the follow-up before being asked.

    Parameters
    ----------
    path : str
        Path to the JSON persistence file. Default ".intuition_stone.json".
    config : IntuitionConfig
        Language signal sets (stopwords, normalise_fn). Default EN_CONFIG.
    user_name : str
        Display name used inside directives. Default "User".
    save_every : int
        Persist to disk every N turns.
    min_observations : int
        Minimum transitions needed before any prediction is emitted.
    min_confidence : float
        Minimum P(B|A) needed to include a prediction.
    max_predictions : int
        Maximum number of follow-up topics to mention in one directive.
    expire_days : float
        Transitions not seen within this many days are pruned.
    """

    def __init__(
        self,
        path:             str             = ".intuition_stone.json",
        config:           IntuitionConfig = EN_CONFIG,
        user_name:        str             = "User",
        save_every:       int             = 5,
        min_observations: int             = 3,
        min_confidence:   float           = 0.40,
        max_predictions:  int             = 2,
        expire_days:      float           = _TRANSITION_EXPIRE_DAYS,
    ) -> None:
        self._path       = path
        self._cfg        = config
        self._user_name  = user_name
        self._save_every = save_every
        self._min_obs    = min_observations
        self._min_conf   = min_confidence
        self._max_pred   = max_predictions
        self._expire_days = expire_days
        self._lock       = threading.Lock()
        self._profile    = self._load()
        # Runtime only: last 2 topic vectors (2-step prediction, v1.1)
        self._prev_topic:  FrozenSet[str] = frozenset()
        self._prev2_topic: FrozenSet[str] = frozenset()

    # ── Public API ─────────────────────────────────────────────────────────────

    def observe(
        self,
        user_message:      str,
        assistant_message: str,
        verbose:           bool = False,
    ) -> Optional[dict]:
        """
        Process one conversation turn.

        Call after every user/assistant exchange.
        """
        with self._lock:
            return self._observe_locked(user_message, assistant_message, verbose)

    def get_prediction_directive(self, current_user_message: str = "") -> str:
        """
        Return a directive predicting likely follow-up topics.

        Pass the current user message so the prediction is anchored to the
        present topic. Returns an empty string until enough data exists.
        """
        with self._lock:
            return self._prediction_directive_locked(current_user_message)

    def summary(self) -> dict:
        """Return a snapshot of the current transition model."""
        with self._lock:
            return self._summary_locked()

    def reset(self) -> None:
        """Wipe the profile from memory and disk."""
        with self._lock:
            self._profile    = _empty_profile()
            self._prev_topic = frozenset()
            if os.path.exists(self._path):
                os.remove(self._path)

    # ── Internals ──────────────────────────────────────────────────────────────

    def _extract_topic(self, text: str) -> FrozenSet[str]:
        return _top_words(text, self._cfg.stopwords, self._cfg.normalise_fn)

    def _observe_locked(
        self,
        user_message:      str,
        assistant_message: str,
        verbose:           bool,
    ) -> Optional[dict]:
        self._profile["turn_count"] += 1
        current_topic = self._extract_topic(user_message)

        recorded_transitions: List[Tuple[str, str]] = []

        import time as _time
        now = _time.time()

        if self._prev_topic and current_topic:
            # Coarse topic-level transition
            prev_key = _topic_key(self._prev_topic)
            curr_key = _topic_key(current_topic)
            if prev_key != curr_key:
                t = self._profile["transitions"]
                if prev_key not in t:
                    t[prev_key] = {}
                t[prev_key][curr_key] = t[prev_key].get(curr_key, 0) + 1
                recorded_transitions.append((prev_key, curr_key))

            # Fine word-level transitions + timestamp (v1.1)
            wf  = self._profile["word_follows"]
            wls = self._profile["word_last_seen"]
            for pw in self._prev_topic:
                if pw not in wf:  wf[pw]  = {}
                if pw not in wls: wls[pw] = {}
                for cw in current_topic:
                    if pw != cw:
                        wf[pw][cw]  = wf[pw].get(cw, 0) + 1
                        wls[pw][cw] = now

        # 2-step chain: record A→B→C transition (v1.1)
        if self._prev2_topic and self._prev_topic and current_topic:
            wf  = self._profile["word_follows"]
            wls = self._profile["word_last_seen"]
            for pw in self._prev2_topic:
                if pw not in wf:  wf[pw]  = {}
                if pw not in wls: wls[pw] = {}
                for cw in current_topic:
                    if cw not in self._prev_topic and pw != cw:
                        chain_key = f"_2_{cw}"   # 2-step transition marker
                        wf[pw][chain_key]  = wf[pw].get(chain_key, 0) + 1
                        wls[pw][chain_key] = now

        self._prev2_topic = self._prev_topic
        self._prev_topic  = current_topic

        if self._profile["turn_count"] % self._save_every == 0:
            self._expire_old_transitions()   # prune stale edges (v1.1)
            self._save_locked()

        if verbose:
            return {
                "turn":                  self._profile["turn_count"],
                "current_topic":         sorted(current_topic),
                "recorded_transitions":  recorded_transitions,
                "profile":               self._summary_locked(),
            }
        return None

    def _expire_old_transitions(self) -> None:
        """Remove word_follows entries older than expire_days (v1.1). Lock held."""
        import time as _time
        cutoff = _time.time() - self._expire_days * 86400
        wf  = self._profile["word_follows"]
        wls = self._profile["word_last_seen"]
        to_delete = []
        for pw, follows in wf.items():
            for cw in list(follows.keys()):
                last = wls.get(pw, {}).get(cw, 0)
                if last > 0 and last < cutoff:
                    to_delete.append((pw, cw))
        for pw, cw in to_delete:
            wf[pw].pop(cw, None)
            wls.get(pw, {}).pop(cw, None)
        # Remove empty dicts
        for pw in list(wf.keys()):
            if not wf[pw]:
                wf.pop(pw, None)
                wls.pop(pw, None)

    def _predict_from_words(
        self, topic_words: FrozenSet[str]
    ) -> List[Tuple[str, float]]:
        """
        Return predicted follow-up words with confidence scores.

        Primary gate: absolute count >= min_observations.
        Secondary gate: P(B|A) >= min_confidence.
        """
        wf = self._profile["word_follows"]
        scores: Dict[str, float] = {}

        for pw in topic_words:
            if pw not in wf:
                continue
            follows = wf[pw]
            total = sum(follows.values())
            for cw, count in follows.items():
                if count < self._min_obs:
                    continue
                if cw in topic_words:
                    continue
                conf = count / total
                if conf >= self._min_conf:
                    scores[cw] = max(scores.get(cw, 0.0), conf)
                else:
                    # Count gate met but diluted — emit with count-based score
                    pseudo_conf = min(0.99, count / self._min_obs * self._min_conf)
                    scores[cw] = max(scores.get(cw, 0.0), pseudo_conf)

        return sorted(scores.items(), key=lambda x: -x[1])

    def _prediction_directive_locked(self, current_user_message: str) -> str:
        if not current_user_message:
            return ""

        topic = self._extract_topic(current_user_message)
        if not topic:
            return ""

        predictions = self._predict_from_words(topic)
        if not predictions:
            return ""

        top = predictions[: self._max_pred]

        pred_words = [w for w, _ in top]
        if len(pred_words) == 1:
            topics_str = f'"{pred_words[0]}"'
        else:
            topics_str = " and ".join(f'"{w}"' for w in pred_words)

        topic_sample = ", ".join(sorted(topic)[:3])
        _uname = self._user_name

        return (
            f"[Intuition Stone — internal directive, do not repeat this]\n"
            f"Based on {_uname}'s conversation patterns, topics around "
            f"{topic_sample} are often followed by questions about {topics_str}. "
            f"If your response will naturally lead there, consider addressing it proactively."
        )

    def _summary_locked(self) -> dict:
        transitions  = self._profile["transitions"]
        word_follows = self._profile["word_follows"]

        top_pairs: List[Tuple[str, str, int]] = []
        for pw, follows in word_follows.items():
            for cw, count in follows.items():
                if count >= self._min_obs:
                    top_pairs.append((pw, cw, count))
        top_pairs.sort(key=lambda x: -x[2])

        return {
            "turn_count": self._profile["turn_count"],
            "unique_topic_transitions": sum(len(v) for v in transitions.values()),
            "unique_word_pairs_tracked": sum(len(v) for v in word_follows.values()),
            "top_word_transitions": [
                {"from": pw, "to": cw, "count": cnt}
                for pw, cw, cnt in top_pairs[:5]
            ],
        }

    # ── Persistence ────────────────────────────────────────────────────────────

    def _load(self) -> dict:
        if os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                p = _empty_profile()
                p.update(data)
                # v1.0 migration: add word_last_seen if missing
                if "word_last_seen" not in p:
                    p["word_last_seen"] = {}
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

import threading as _threading

_TR_MAP = str.maketrans("çğışöüÇĞİŞÖÜ", "cgisouCGISOu")

def _norm_tr(text: str) -> str:
    return text.translate(_TR_MAP)

TR_CONFIG = IntuitionConfig(
    stopwords=frozenset({
        # Turkish stop words (diacritics stripped)
        "bir", "bu", "su", "ve", "ile", "ama", "ya", "da", "de",
        "icin", "ben", "sen", "biz", "siz", "var", "yok", "oldu",
        "daha", "cok", "az", "tam", "sadece", "bile", "hic",
        "nasil", "neden", "ne", "nerede", "kim", "hangi",
        "gibi", "gore", "kadar", "beri", "sonra", "once",
        "suan", "simdi", "zaman", "her", "butun", "tum",
        "evet", "hayir", "tamam", "peki", "ok",
        # English stop words (mixed-language support)
        "a", "an", "the", "is", "are", "in", "on", "at", "to",
        "for", "of", "with", "and", "or", "but", "this", "that",
        "how", "what", "why", "when", "where", "who", "which",
        "can", "will", "do", "does", "use", "get", "make",
        "work", "need", "want", "just", "also", "now", "then",
    }),
    normalise_fn=_norm_tr,
)


# ── Singleton ─────────────────────────────────────────────────────────────────

_intuition_stone_instance: Optional[IntuitionStone] = None
_intuition_stone_lock = _threading.Lock()


def get_intuition_stone(
    path: str = ".intuition_stone.json",
    config: IntuitionConfig = EN_CONFIG,
    user_name: str = "User",
) -> IntuitionStone:
    """Return the global IntuitionStone instance (lazy init, thread-safe)."""
    global _intuition_stone_instance
    if _intuition_stone_instance is None:
        with _intuition_stone_lock:
            if _intuition_stone_instance is None:
                _intuition_stone_instance = IntuitionStone(
                    path=path, config=config, user_name=user_name
                )
    return _intuition_stone_instance
