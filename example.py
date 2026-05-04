"""
Mind Stone — Usage Examples
===========================

Three examples, increasing complexity:
  1. Basic usage (no LLM required)
  2. OpenAI integration
  3. Turkish language configuration
"""

# ──────────────────────────────────────────────────────────────────────────────
# Example 1 — Basic usage (stdlib only, no LLM needed)
# ──────────────────────────────────────────────────────────────────────────────

from mind_stone import MindStone

def example_basic():
    """Observe turns and watch the profile evolve."""
    stone = MindStone(path=".demo_profile.json")

    print("=== Example 1: Basic observation ===\n")

    # Simulate a technical, example-seeking user who likes concise answers
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
         "Threads share memory, processes don't. Use threads for I/O-bound, processes for CPU-bound work."),
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
    """Minimal OpenAI integration showing where Mind Stone hooks in."""

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
        # 1. Build style-aware system prompt
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

        # 3. Update history
        history.append({"role": "user",      "content": user_message})
        history.append({"role": "assistant", "content": assistant_message})

        # 4. Observe the turn — Mind Stone learns silently
        stone.observe(user_message, assistant_message)

        return assistant_message

    # Run a small demo conversation
    questions = [
        "Explain what a transformer model is.",
        "too long, shorter please",
        "show me a python example of tokenisation",
        "got it thanks",
        "what is attention mechanism",
    ]

    for q in questions:
        print(f"User: {q}")
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
    """Use the Turkish signal sets instead of the English defaults."""
    print("=== Example 3: Turkish configuration ===\n")

    from signals_turkish import TR_CONFIG
    stone = MindStone(path=".turkish_demo_profile.json", config=TR_CONFIG)

    turns = [
        ("cuda ile gpu kernel nasıl yazılır?",           "CUDA kernel yazmak için..."),
        ("kısalt",                                        "Tamam."),
        ("python async queue örnek ver",                  "İşte örnek:"),
        ("tamam anladım",                                 "Güzel."),
        ("transformer modelde attention nasıl çalışır?",  "Attention mekanizması..."),
        ("çok uzun, sadece kod",                          "```python\n...```"),
        ("harika",                                        "Rica ederim."),
        ("gpu bellek yönetimi için örnek göster",         "```python\nimport torch\n...```"),
        ("oldu teşekkürler",                              "Kolay gelsin."),
        ("embedding boyutu neden önemli?",                "Embedding boyutu..."),
    ]

    for user_msg, assistant_msg in turns:
        stone.observe(user_msg, assistant_msg)

    print("Profil (10 tur gözlem):")
    for key, val in stone.summary().items():
        print(f"  {key:<22} {val}")

    stone.reset()
    print()


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    example_basic()
    example_openai()
    example_turkish()
