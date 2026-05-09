"""Smoke test for all four public standalone stones."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from mind_stone import MindStone
from echo_stone import EchoStone
from bond_stone import BondStone
from intuition_stone import IntuitionStone


def test_mind_stone():
    ms = MindStone(path=".test_mind.json", user_name="Alice")
    # Non-tech "too long" → sets verbosity_general
    ms.observe("show me an example", "Long response here...")
    ms.observe("too long keep it short", "")
    assert ms.profile.verbosity == 0.10, "verbosity should be 0.10"
    assert ms.profile.verbosity_general == 0.10, "verbosity_general should be 0.10"
    # Technical "too long" → sets verbosity_tech
    ms.observe("this python function is too long", "")
    assert ms.profile.verbosity_tech == 0.10, "verbosity_tech should be 0.10"
    # Technical positive → sets verbosity_tech back up
    ms.observe("go deeper on this python api", "")
    assert ms.profile.verbosity_tech == 0.90, "verbosity_tech should be 0.90 after pos override"
    # summary must not crash
    s = ms.summary()
    assert "total_observations" in s
    print("[MindStone] PASS")


def test_echo_stone():
    es = EchoStone(path=".test_echo.json")
    long_response = "word " * 130   # 130 words > overload_word_count default of 120
    es.observe("explain async/await", long_response)
    es.observe("ok", "")  # short deflection after long response
    assert es.profile.overload_rate > 0.50, "overload_rate should rise"
    s = es.summary()
    assert "comprehension_rate" in s
    print("[EchoStone] PASS")


def test_bond_stone():
    bs = BondStone(path=".test_bond.json")
    bs.observe("I am working on a project using python and docker", "")
    assert "python" in bs._profile["tech_stack"], "python should be in stack"
    assert "docker" in bs._profile["tech_stack"], "docker should be in stack"
    bs.remember("I always prefer typed Python over raw dicts")
    assert len(bs._profile["facts"]) >= 1, "no facts stored"
    ctx = bs.get_context_directive()
    assert "python" in ctx.lower() or "docker" in ctx.lower(), "stack missing from directive"
    bs.alias("usual setup", "Python + CUDA on Windows 11")
    resolved = bs.resolve("use the usual setup please")
    assert resolved is not None, "alias should resolve"
    print("[BondStone] PASS")


def test_intuition_stone():
    ist = IntuitionStone(path=".test_intuition.json", user_name="Bob")
    for _ in range(4):
        ist.observe("python async error", "Here is the fix...")
        ist.observe("event loop blocked", "Here is the solution...")
    s = ist.summary()
    assert s["turn_count"] == 8, "expected 8 turns"
    assert s["unique_word_pairs_tracked"] > 0, "no word pairs tracked"
    d = ist.get_prediction_directive("python async problem")
    # Directive may be empty if transitions haven't crossed threshold yet
    print("[IntuitionStone] directive:", repr(d[:80]) if d else "(empty — below threshold)")
    print("[IntuitionStone] PASS")


def cleanup():
    for f in [".test_mind.json", ".test_echo.json", ".test_bond.json", ".test_intuition.json"]:
        if os.path.exists(f):
            os.remove(f)


if __name__ == "__main__":
    try:
        test_mind_stone()
        test_echo_stone()
        test_bond_stone()
        test_intuition_stone()
        print()
        print("All 4 standalone stones — smoke tests PASSED")
    finally:
        cleanup()
