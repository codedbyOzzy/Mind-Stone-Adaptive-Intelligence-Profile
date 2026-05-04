"""
Mind Stone — Usage Examples
===========================

Three examples, increasing complexity:
  1. Basic usage (no LLM required)
  2. OpenAI integration
  3. Turkish language configuration (shows multilingual support)

All code, comments, and output are in English.
Example 3 contains Turkish conversation strings — these are the actual
user messages in a Turkish-language assistant, included to demonstrate
that signal detection works across languages. The profile output is
always in English regardless of input language.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Example 1 — Basic usage (stdlib only, no LLM needed)
# ──────────────────────────────────────────────────────────────────────────────

from mind_stone import MindStone

def example_basic():
    """Observe turns and watch the profile evolve.

    Simulates a technical user who prefers concise answers and code examples.
    After 10 turns the profile reflects this and the directive activates.
    """
    stone = MindStone(path=".demo_profile.json")

    print("=== Example 1: Basic observation ===\n")

    turns = [
        ("How do I parse JSON in Python?",
         "Use `json.loads(text)` to parse a string, or `json.load(file)` for a file object."),

        ("ok",
         "Let me know if you need anything else."),

        ("Show me a real example with error handling",
         "```python\nimport json\ntry:\n    data = json.loads(text)\nexcept json.JSONDecodeError as e:\n    print(f'Invalid JSON: {e}')\n```"),

        ("nice thanks",
         "Glad it helped!"),

        ("How does async/await work with the requests library?",
         "requests is synchronous — for async HTTP use httpx or aiohttp instead."),

        ("too long, just show code",
         "```python\nimport httpx\nasync def fetch(url):\n    async with httpx.AsyncClient() as c:\n        return await c.get(url)\n```"),

        ("got it",
         "Perfect."),

        ("example for threading queue please",
         "```python\nimport queue, threading\nq = queue.Queue()\nthreading.Thread(target=worker, args=(q,)).start()\n```"),

        ("perfect",
         "Done."),

        ("difference between process and thread?",
         "Threads share memory, processes don't. Use threads for I/O-bound, "
         "processes for CPU-bound work."),
    ]

    for user_msg, assistant_msg in turns:
        stone.observe(user_msg, assistant_msg)

    print("Profile after 10 turns:")
    for key, val in stone.summary().items():
        print(f"  {key:<22} {val}")

    print("\nStyle directive:")
    directive = stone.get_style_directive()
    print(directive if directive else "  (not enough data yet — need ~12 turns)")

    stone.reset()
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Example 2 — OpenAI integration
# ──────────────────────────────────────────────────────────────────────────────

def example_openai():
    """Minimal OpenAI integration showing where Mind Stone hooks in.

    The only change to a standard chat loop is:
      - one observe() call after each turn
      - one get_style_directive() call before each LLM call

    Everything else is unchanged.
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("=== Example 2: OpenAI (skipped — openai not installed) ===\n")
        return

    import os
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("=== Example 2: OpenAI (skipped — OPENAI_API_KEY not set) ===\n")
        return

    print("=== Example 2: OpenAI integration ===\n")

    client = OpenAI(api_key=api_key)
    stone  = MindStone(path=".openai_demo_profile.json")

    BASE_SYSTEM_PROMPT = "You are a helpful AI assistant."
    history = []

    def chat(user_message: str) -> str:
        """One turn of conversation with adaptive style."""

        # 1. Inject adaptive style directive (empty until ~12 turns of data)
        directive = stone.get_style_directive()
        system    = BASE_SYSTEM_PROMPT
        if directive:
            system += "\n\n" + directive

        # 2. Call the LLM
        messages = [{"role": "system", "content": system}] + history
        messages.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=300,
        )
        assistant_message = response.choices[0].message.content

        # 3. Update conversation history
        history.append({"role": "user",      "content": user_message})
        history.append({"role": "assistant", "content": assistant_message})

        # 4. Observe — Mind Stone learns silently from this turn
        stone.observe(user_message, assistant_message)

        return assistant_message

    questions = [
        "Explain what a transformer model is.",
        "too long, shorter please",
        "show me a python example of tokenisation",
        "got it thanks",
        "what is attention mechanism",
    ]

    for q in questions:
        print(f"User:      {q}")
        answer = chat(q)
        print(f"Assistant: {answer[:120]}{'...' if len(answer) > 120 else ''}\n")

    print("Profile after conversation:")
    for k, v in stone.summary().items():
        print(f"  {k:<22} {v}")

    stone.reset()
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Example 3 — Turkish language configuration
# ──────────────────────────────────────────────────────────────────────────────

def example_turkish():
    """Demonstrate Mind Stone with Turkish signals (TR_CONFIG).

    The conversation turns below are in Turkish — they are realistic messages
    from a Turkish-language AI assistant interaction. Each turn is annotated
    with an English translation so you can follow the signal detection logic
    without knowing Turkish.

    Key observation: the *profile output and style directive are always in
    English*, regardless of the language of user input. Only the signal
    detection (which words trigger which profile update) is language-specific.

    To adapt Mind Stone for any language, you only need to provide a
    SignalConfig with translated signal sets — see signals_turkish.py.
    """
    print("=== Example 3: Turkish language configuration ===")
    print("    (Turkish inputs, English profile output)\n")

    from signals_turkish import TR_CONFIG
    stone = MindStone(path=".turkish_demo_profile.json", config=TR_CONFIG)

    turns = [
        # [EN: "how do you write a cuda gpu kernel?"]  → tech signal
        ("cuda ile gpu kernel nasil yazilir?",           "CUDA kernel yazmak icin..."),
        # [EN: "shorten"]                               → neg_verbosity signal
        ("kisalt",                                        "Tamam."),
        # [EN: "give me a python async queue example"]  → example + tech signal
        ("python async queue ornek ver",                  "Iste ornek:"),
        # [EN: "ok understood"]                         → satisfaction signal
        ("tamam anladim",                                 "Guzel."),
        # [EN: "how does attention work in transformers?"] → tech + theory signal
        ("transformer modelde attention nasil calisir?",  "Attention mekanizmasi..."),
        # [EN: "too long, just code"]                   → neg_verbosity signal
        ("cok uzun, sadece kod",                          "```python\n...```"),
        # [EN: "great"]                                 → satisfaction signal
        ("harika",                                        "Guzel."),
        # [EN: "show me a gpu memory management example"] → example + tech signal
        ("gpu bellek yonetimi icin ornek goster",         "```python\nimport torch\n...```"),
        # [EN: "done, thanks"]                          → satisfaction signal
        ("oldu tesekkurler",                              "Kolay gelsin."),
        # [EN: "why does embedding size matter?"]       → theory signal
        ("embedding boyutu neden onemli?",                "Embedding boyutu..."),
    ]

    for user_msg, assistant_msg in turns:
        stone.observe(user_msg, assistant_msg)

    print("Profile after 10 Turkish turns (always English output):")
    for key, val in stone.summary().items():
        print(f"  {key:<22} {val}")

    directive = stone.get_style_directive()
    if directive:
        print("\nStyle directive (English, regardless of input language):")
        print(directive)

    stone.reset()
    print()


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    example_basic()
    example_openai()
    example_turkish()
