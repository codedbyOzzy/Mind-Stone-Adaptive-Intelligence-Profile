# Intelligence Stones

> *"Most AI assistants know what to say. These teach them how to say it — and whether it actually landed."*

A collection of lightweight, zero-dependency Python modules that give AI assistants genuine understanding of the people they talk to.

Each stone is a standalone drop-in. No framework. No configuration. No external dependencies.  
Together, they build something more complete.

---

## The Collection

| Stone | Status | What it learns |
|-------|--------|----------------|
| [**Mind Stone**](#-mind-stone) | `v1.3.0` ✅ | *How* the user communicates — style, depth, pace |
| [**Echo Stone**](#-echo-stone) | `v1.1.0` ✅ | *Whether* the user actually understood |
| [**Bond Stone**](#bond-stone) | `v1.1.0` ✅ | *Who* the user is — their world, context, history |
| [**Intuition Stone**](#intuition-stone) | `v1.1.0` ✅ | *Where* the conversation is going |

All four stones are production-tested inside [FRIDAY Synapse](https://github.com/codedbyOzzy/ProjectFRIDAY) and available as standalone modules.

---

## Used in Production

The complete four-stone system powers **[FRIDAY Synapse](https://github.com/codedbyOzzy/ProjectFRIDAY)** — a Windows-native desktop AI system with persistent memory, real voice interaction, and deep OS-level integration.

All four stones run on every conversation turn inside FRIDAY Synapse, forming the adaptive cognitive layer that makes the system feel different from a standard LLM wrapper.

---

## How the stones fit together

Most AI assistants fail at communication in four distinct ways.  
Each stone addresses exactly one of them.

```
The problem                         The stone that solves it
─────────────────────────────────   ──────────────────────────────────────
The assistant speaks the same way   Mind Stone — learns your style,
to everyone, regardless of who      depth preference, and pace. Adjusts
you are or how you communicate.     the delivery to fit you specifically.

The assistant never knows if its    Echo Stone — detects false
explanation actually worked. It     confirmations, confusion patterns,
moves on whether you understood     and cognitive overload. Knows the
or not.                             difference between "got it" and
                                    "got it" (but didn't).

Every session starts from zero.     Bond Stone — builds a persistent
You re-explain your project, your   model of your world across sessions.
stack, your constraints — every     Remembers without being told to
single time.                        remember.

The assistant only sees the         Intuition Stone — learns the shape
current question. It has no idea    of conversations. Knows where this
where the conversation is going.    is going before you finish asking.
```

### The feedback loop between Mind Stone and Echo Stone

Mind Stone and Echo Stone are designed to work as a pair. Neither is complete without the other.

**Mind Stone** adjusts *how* the assistant speaks — but it can't tell if those adjustments are actually helping. It observes signals like "too long" or "show me an example" and updates the style profile accordingly. What it cannot see is whether its calibrated responses are landing.

**Echo Stone** fills that gap. It watches the turn *after* the assistant speaks and measures the reaction. A user who says "got it" and then asks the exact same question again didn't get it — and Echo Stone knows the difference. That feedback then shapes how the assistant explains in future turns.

Together, they form a self-correcting communication loop:

```
Mind Stone calibrates the delivery
         │
         ▼
  Assistant responds
         │
         ▼
Echo Stone measures whether it worked
         │
         ├── understood → reinforces the approach
         │
         └── confused   → flags for adjustment next turn
```

Without Mind Stone, Echo Stone has nothing to improve upon.  
Without Echo Stone, Mind Stone is adjusting blindly.

### The full architecture — all four stones

When all four stones are active, the assistant operates on four layers simultaneously:

```
  ┌──────────────────────────────────────────────────────────┐
  │  Layer 1 — Expression                                    │
  │                                                          │
  │  Mind Stone      How to speak to you                     │
  │  Echo Stone      Whether it worked                       │
  │                                                          │
  │  Together: a delivery system that calibrates and         │
  │  validates itself with every turn.                       │
  ├──────────────────────────────────────────────────────────┤
  │  Layer 2 — Context                                       │
  │                                                          │
  │  Bond Stone      Who you are — your world, projects,     │
  │                  people, constraints, history            │
  │                                                          │
  │  Without this layer, every session starts from zero.     │
  │  With it, the assistant already knows what you mean      │
  │  when you say "the usual setup" or "that API problem".   │
  ├──────────────────────────────────────────────────────────┤
  │  Layer 3 — Prediction                                    │
  │                                                          │
  │  Intuition Stone Where this conversation is going        │
  │                                                          │
  │  Without this layer, the assistant answers the current   │
  │  question. With it, the assistant is already preparing   │
  │  the answer you haven't asked yet.                       │
  └──────────────────────────────────────────────────────────┘
```

The complete picture — all four working together:

```
  User types a message
       │
       ├─ Intuition Stone  "I've seen this pattern before.
       │                    They'll need X next."
       │
       ├─ Bond Stone       "They're working on the Vision project.
       │                    They use Python + CUDA. Last time
       │                    this came up, the issue was memory."
       │
       ├─ Mind Stone       "Keep it short. Lead with code.
       │                    Skip the theory."
       │
       ▼
  Assistant responds
       │
       ▼
  Echo Stone             "They said 'ok' — but they asked
                          the same thing three turns ago.
                          This explanation isn't landing.
                          Flag for next turn."
```

No magic. No large models. No embeddings.  
Each stone is a small, auditable Python module running in microseconds.  
The intelligence comes from careful observation accumulated over time.

---

## 🧠 Mind Stone

> *The user's communication fingerprint — learned silently, applied automatically.*

Observes every conversation turn and builds a quantified model of how a user communicates. Generates a short style directive injected into the system prompt — shaping tone, depth, and format without the user ever configuring anything.

### How it works

```
Each conversation turn:
  user_message + assistant_message
         │
         ▼
  ┌─────────────────────────────────┐
  │        Mind Stone Engine        │
  │                                 │
  │  Signal detection               │
  │    "too long" → verbosity ↓     │
  │    "show me"  → example_bias ↑  │
  │    "cuda gpu" → tech_depth ↑    │
  │    "got it"   → satisfaction ↑  │
  │                                 │
  │  EMA update (α = 0.12)          │
  │  Persistence every 5 turns      │
  └──────────────┬──────────────────┘
                 │
                 ▼
  ┌─────────────────────────────────┐
  │     Intelligence Profile        │
  │                                 │
  │  verbosity:      0.21  (terse)  │
  │  tech_depth:     0.87  (expert) │
  │  example_bias:   0.74  (examp.) │
  │  follow_up_rate: 0.62           │
  │  confidence:     68%            │
  └──────────────┬──────────────────┘
                 │
                 ▼
  get_style_directive()
  ─────────────────────────────────────────
  "This user prefers concise answers.
   Use domain terminology freely.
   Lead with a code example."
  ─────────────────────────────────────────
```

### Profile dimensions

| Dimension | Low (0) | High (1) | Learned from |
|-----------|---------|----------|-------------|
| `verbosity` | Terse, direct | Detailed, thorough | Explicit signals + message length |
| `tech_depth` | Plain language | Expert vocabulary | Technical word density |
| `example_bias` | Theory first | Examples first | "show me" / "why does it" signals |
| `follow_up_rate` | Satisfied by first reply | Always asks more | Short confirmations vs long follow-ups |

### Learning curve

```
Turns:      0    5   12   25   40   55+
Confidence: 0%  0%  14%  40%  70% 100%
Directive:  ───────── ON ──────────────▶
```

### Quick start

```python
from mind_stone import MindStone

stone = MindStone()

# After every conversation turn:
stone.observe(user_message, assistant_message)

# Before every LLM call:
directive = stone.get_style_directive()   # "" until ~12 turns
if directive:
    system_prompt += "\n\n" + directive
```

### API

#### `MindStone(path, user_name, normalise_fn)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `path` | `.mind_stone.json` | Profile persistence path |
| `user_name` | `"User"` | Display name used in directives |
| `normalise_fn` | `None` | Text normalisation for non-ASCII languages |

#### Methods

```python
stone.observe(user_message, assistant_message)
stone.get_style_directive() -> str
stone.summary()             -> dict
stone.reset()
```

#### Profile dimensions (v1.3)

| Dimension | What changed in v1.3 |
|-----------|---------------------|
| `verbosity` | Global verbosity (unchanged) |
| `verbosity_tech` | **New** — verbosity tracked separately for technical topics |
| `verbosity_general` | **New** — verbosity tracked separately for general topics |
| `tech_depth` | Unchanged |
| `example_bias` | Unchanged |
| `follow_up_rate` | Unchanged |

When `verbosity_tech` and `verbosity_general` diverge by more than 0.25, the directive
emits a topic-specific instruction instead of a global one. Explicit override signals
("keep it short", "more detail") now **directly set** the value instead of nudging via
EMA — immediate effect guaranteed.

---

## 📡 Echo Stone

> *The assistant spoke. But did the user understand?*

Analyses the user's reaction to each response and detects comprehension patterns that neither the user nor the assistant would explicitly flag. Translates these into a directive that shapes *how* the assistant explains — not just what it says.

### The problem

```
Standard flow:
  Assistant explains X  →  User says "ok got it"  →  Assistant moves on

What actually happened:
  User says "ok got it"  →  Two minutes later: "wait, how does X work again?"
                                                              ↑
                                              Echo Stone caught this.
                                              It's called a false confirmation.
```

### Detected patterns

| Signal | What happened |
|--------|--------------|
| `explicit_confusion` | User directly says they didn't understand |
| `overload_deflect` | Long response → 1–3 word reply (cognitive shutdown) |
| `deepening` | User asks a deeper question — they understood and want more |
| `rephrase` | User asks the same thing in different words |
| `false_confirmation` | User confirms, then returns to the same topic |
| `genuine_confirmation` | User confirms and moves to a genuinely different topic |

### Comprehension profile

| Dimension | Low (0) | High (1) |
|-----------|---------|----------|
| `comprehension_rate` | Rarely understands first try | Always gets it first try |
| `false_confirm_rate` | Confirmations are genuine | Often confirms without understanding |
| `overload_rate` | Handles complexity well | Easily overwhelmed by long responses |
| `depth_rate` | Stays surface level | Digs deeper every time |

### Quick start

```python
from echo_stone import EchoStone

stone = EchoStone()

# Same interface as Mind Stone:
stone.observe(user_message, assistant_message)

# Before every LLM call:
directive = stone.get_comprehension_directive()   # "" until ~8 turns
if directive:
    system_prompt += "\n\n" + directive
```

### Running both stones together

```python
from mind_stone import MindStone
from echo_stone import EchoStone

mind = MindStone()
echo = EchoStone()

def after_each_turn(user_message, assistant_message):
    mind.observe(user_message, assistant_message)
    echo.observe(user_message, assistant_message)

def build_system_prompt(base_prompt, user_message):
    directives = []
    d = mind.get_style_directive()
    if d: directives.append(d)
    d = echo.get_comprehension_directive()
    if d: directives.append(d)
    return base_prompt + ("\n\n" + "\n\n".join(directives) if directives else "")
```

### API

#### `EchoStone(path, config, ema_alpha, min_confidence, save_every, rephrase_threshold, overload_word_count)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `path` | `.echo_stone.json` | Profile persistence path |
| `config` | `EN_CONFIG` | Language signal sets |
| `ema_alpha` | `0.15` | Learning rate |
| `min_confidence` | `0.12` | Threshold before directives activate (~8 turns) |
| `save_every` | `5` | Persist to disk every N turns |
| `rephrase_threshold` | `0.38` | Jaccard similarity for rephrase detection |
| `overload_word_count` | `120` | Response length that triggers overload check |

#### Methods

```python
stone.observe(user_message, assistant_message, verbose=False)
stone.get_comprehension_directive() -> str
stone.summary()                     -> dict
stone.reset()
```

#### `observe(verbose=True)` report

```python
{
  "signal":     str | None,   # detected comprehension signal
  "is_confirm": bool,
  "profile":    { ... }
}
```

---

## Language customisation

Both stones ship with English signal sets. Any language is supported by passing a config object:

```python
from mind_stone import MindStone, SignalConfig
from echo_stone import EchoStone, EchoConfig

mind = MindStone(config=SignalConfig(
    neg_verbosity    = frozenset({"kurzer", "zu lang"}),
    pos_verbosity    = frozenset({"mehr details", "erklar mir"}),
    example_signals  = frozenset({"beispiel", "zeig mir"}),
    theory_signals   = frozenset({"warum", "wie funktioniert"}),
    satisfied_tokens = frozenset({"ok", "danke", "verstanden"}),
    tech_words       = frozenset({"python", "api", "docker"}),
    normalise_fn     = None,
))

echo = EchoStone(config=EchoConfig(
    confusion_signals   = frozenset({"verstehe nicht", "nochmal", "was meinst du"}),
    confirmation_tokens = frozenset({"ok", "verstanden", "danke", "gut"}),
    deepen_signals      = frozenset({"also wenn", "bedeutet das", "was ware wenn"}),
    normalise_fn        = None,
))
```

For non-ASCII languages — provide a `normalise_fn` that strips diacritics.  
See [`signals_turkish.py`](signals_turkish.py) for a complete reference implementation covering both stones.

---

## Bond Stone

Right now, every session starts from zero. You mention "the Vision project" and the assistant has no idea what that is — even though you explained it three sessions ago. You say "the usual setup" and it asks you to clarify. You state your constraints once and then have to repeat them forever.

Bond Stone solves this by building a persistent, structured model of the user's world — not as raw conversation logs, but as a live knowledge graph: entities, relationships, and context that accumulates across every session.

The difference between Bond Stone and simply saving chat history is the difference between a notebook and a map. Chat history stores what was said. Bond Stone builds what it means — who the people are, what the projects involve, how the pieces connect. When you say "same thing as before", it actually knows what that means.

### Quick start

```python
from bond_stone import BondStone

stone = BondStone()

# After every conversation turn:
stone.observe(user_message, assistant_message)

# Before every LLM call:
ctx = stone.get_context_directive()
if ctx:
    system_prompt += "\n\n" + ctx

# Explicit fact (passive extraction works automatically):
stone.remember("I work on the Vision project using Python and CUDA")

# Register a shorthand alias:
stone.alias("usual setup", "Python + CUDA + Ollama on Windows 11")
```

### What gets captured automatically

| Signal type | Trigger | Example |
|-------------|---------|---------|
| Tech stack  | Technical term found in message | `python`, `docker`, `cuda` |
| Explicit fact | Remember/note signal | "remember that I work on the Vision project" |
| Constraint | Can't-use signal | "I can't use Docker in this environment" |
| Preference | Preference signal | "I always prefer typed Python over raw dicts" |
| Alias | `stone.alias()` call | "usual setup" → "Python + CUDA + Ollama" |

### API

#### `BondStone(path, config, save_every, max_facts)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `path` | `.bond_stone.json` | Profile persistence path |
| `config` | `EN_CONFIG` | Language signal sets |
| `save_every` | `5` | Persist to disk every N turns |
| `max_facts` | `60` | Maximum stored facts (oldest/lowest-confidence pruned first) |

#### Methods

```python
stone.observe(user_message, assistant_message, verbose=False)
stone.remember(fact, fact_type="explicit")
stone.alias(shorthand, expansion)
stone.resolve(query)           -> Optional[str]
stone.get_context_directive(topic_hint="")  -> str
stone.summary()                -> dict
stone.reset()
```

---

## Intuition Stone

A capable human assistant doesn't just answer the question in front of them. They recognise patterns. They know that when a developer asks *this kind of question*, they'll need *this* next — and they quietly prepare for it before being asked.

Intuition Stone learns the shape of conversations. It tracks which questions tend to lead to which follow-ups, which topics always resurface, which paths a conversation typically takes once it starts down a certain road. Over time, it builds a model of where this specific user tends to go.

This isn't about predicting the future. It's about recognising that most conversations follow familiar arcs — and that an assistant who has seen enough of them can stop waiting to be asked.

### Quick start

```python
from intuition_stone import IntuitionStone

stone = IntuitionStone()

# After every conversation turn:
stone.observe(user_message, assistant_message)

# Before every LLM call:
hint = stone.get_prediction_directive(current_user_message)
if hint:
    system_prompt += "\n\n" + hint
```

### How prediction works

```
Observations accumulate:
  turn 1:  "python async error" → "event loop"
  turn 2:  "fastapi endpoint"   → "async handler"
  turn 3:  "asyncio task"       → "event loop"
  turn 4:  "aiohttp request"    → "event loop"

After turn 4, min_observations (3) reached:
  cuda → memory  (seen 4×, P=0.71)

Next time the user asks about "asyncio task":
  directive → "this user usually follows up about event loop next"
```

### API

#### `IntuitionStone(path, config, user_name, save_every, min_observations, min_confidence, max_predictions, expire_days)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `path` | `.intuition_stone.json` | Profile persistence path |
| `config` | `EN_CONFIG` | Language signal sets |
| `user_name` | `"User"` | Display name used in directives |
| `save_every` | `5` | Persist every N turns |
| `min_observations` | `3` | Minimum times a transition must be seen |
| `min_confidence` | `0.40` | Minimum P(B\|A) to emit a prediction |
| `max_predictions` | `2` | Max follow-up topics in one directive |
| `expire_days` | `90` | Prune transitions not seen within this many days |

#### Methods

```python
stone.observe(user_message, assistant_message, verbose=False)
stone.get_prediction_directive(current_user_message)  -> str
stone.summary()  -> dict
stone.reset()
```

---

## The complete picture

```
Mind Stone      →  the assistant speaks in a way that fits you
Echo Stone      →  the assistant knows whether it worked
Bond Stone      →  the assistant knows your world
Intuition Stone →  the assistant knows where you're going
```

Each stone adds one layer of genuine understanding that is currently missing from every AI assistant — not because the technology doesn't exist, but because no one has built it this way.

Four modules. Zero dependencies. One JSON file per stone.

---

## Design decisions

**Why EMA instead of a counter?**  
A simple counter weights day-1 behaviour forever. EMA ensures recent turns matter more — if preferences shift, the profile adapts within ~15 turns.

**Why 0.12 as the default alpha for Mind Stone?**  
At α=0.12, a single strong signal moves the profile ~12%. Stable enough to ignore one-off turns, fast enough to reflect genuine shifts.

**Why session-dampened EMA in Mind Stone?**  
The first few turns of a new session often don't represent real preferences — the user may be rushed or testing something. Dampening alpha by 50% for the first 3 turns of a new session prevents one atypical session from overriding months of data.

**Why Jaccard similarity for rephrase detection in Echo Stone?**  
Zero-dependency constraint rules out embeddings. Jaccard on content words (stopwords removed) is fast, interpretable, and accurate enough for the short messages that typically trigger rephrase detection.

**Why check overload before confirmation in Echo Stone?**  
A very short reply after a 120-word response signals cognitive shutdown regardless of the words used. Checking overload first prevents a one-word deflection from being misclassified as a confirmation.

**Why not use embeddings or ML?**  
Zero-dependency is a hard design constraint. Both stones run in microseconds, produce auditable profiles, and require no model downloads or API calls. The signal-detection approach is sufficient — and more transparent.

---

## Files

```
mind_stone.py          Mind Stone core module (v1.3.0)
echo_stone.py          Echo Stone core module (v1.1.0)
bond_stone.py          Bond Stone core module (v1.1.0)
intuition_stone.py     Intuition Stone core module (v1.1.0)
signals_turkish.py     Turkish signal sets for all stones
example.py             Usage examples
test_v11.py            Mind Stone v1.1 test suite (16 tests)
test_v12.py            Mind Stone v1.2 test suite (30 tests)
```

---

## Requirements

Python 3.9+. No third-party packages.

---

## Contributing

Signal sets for more languages are very welcome.  
Copy `signals_turkish.py`, adapt the frozensets for your language, and open a pull request.

---

## License

MIT — free to use, modify, and distribute.
