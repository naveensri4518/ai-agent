# ============================================================
# Challenge 4 - Full Agent using Strands SDK + Ollama
# Model    : llama3.2:3b  (runs 100% locally, no cloud)
# Tools    : calculator | weather | age_calculator
#            remember_info | recall_info | show_memories
# Memory   : Mem0 + FAISS  (persistent, survives restarts)
# Embedder : nomic-embed-text via Ollama
# ============================================================
#
# This is the "Full Agent" — it combines everything from
# Challenge 2 (tools) and Challenge 3 (memory) into a single
# powerful agent that can:
#   ✓ Do math
#   ✓ Check weather
#   ✓ Calculate ages
#   ✓ Remember your name, preferences, and personal info
#   ✓ Recall that information across sessions (persistent!)
#
# ============================================================
# SETUP — run these commands ONCE before this script:
#
#   pip install strands-agents mem0ai faiss-cpu sentence-transformers
#   ollama pull llama3.2:3b
#   ollama pull nomic-embed-text
#   ollama serve                  ← keep this running in bg
# ============================================================


# ============================================================
# SECTION 1 — IMPORTS
# ============================================================
# We import everything we need upfront so it's easy to see
# all dependencies at a glance.
# ============================================================

import os                   # File path management
import warnings             # Suppress noisy library logs
from datetime import date   # Used in age_calculator tool

# Suppress unimportant deprecation / info warnings
warnings.filterwarnings("ignore")

# mem0 — the persistent memory layer
# Memory.from_config() wires together FAISS + Ollama embedder
from mem0 import Memory

# strands — the agent framework
# Agent   : orchestrates the full reasoning + tool-call loop
# tool    : decorator that registers a Python function as an agent tool
from strands import Agent, tool
from strands.models.ollama import OllamaModel


# ============================================================
# SECTION 2 — PERSISTENT MEMORY SETUP (Mem0 + FAISS)
# ============================================================
# FAISS stores memories as numeric vectors in a local file.
# Mem0 manages the full pipeline:
#   Input text
#     → Ollama LLM extracts key facts
#     → nomic-embed-text converts facts to vectors
#     → FAISS saves vectors to disk
# On recall, the same embedding model is used to search.
# ============================================================

# Directory and file paths for memory storage
MEMORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory_store")
FAISS_PATH = os.path.join(MEMORY_DIR, "memory.index")  # Vector index file
HISTORY_DB = os.path.join(MEMORY_DIR, "history.db")    # SQLite operation log

# Ensure the directory exists before Mem0 tries to write to it
os.makedirs(MEMORY_DIR, exist_ok=True)

# Full Mem0 configuration — all local, zero cloud dependencies
MEM0_CONFIG = {

    # ── VECTOR STORE ─────────────────────────────────────────
    # FAISS = Facebook AI Similarity Search.
    # Saves an index file to disk. Fast, lightweight, no server.
    "vector_store": {
        "provider": "faiss",
        "config": {
            "index_path": FAISS_PATH,
        },
    },

    # ── EMBEDDER ─────────────────────────────────────────────
    # nomic-embed-text is a small, fast embedding model.
    # It converts text → dense vectors (e.g. 768 numbers)
    # so that FAISS can find "similar" memories by distance.
    # Pull once with: ollama pull nomic-embed-text
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text",
            "ollama_base_url": "http://localhost:11434",
        },
    },

    # ── LLM (for memory extraction) ──────────────────────────
    # Before storing, Mem0 uses this LLM to extract structured
    # facts from your raw text.
    # "My name is Thamarai and I love chess"
    #   → extracted: ["Name: Thamarai", "Hobby: chess"]
    # temperature=0 keeps extraction deterministic.
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "llama3.2:3b",
            "temperature": 0,
            "max_tokens": 1000,
            "ollama_base_url": "http://localhost:11434",
        },
    },

    # ── HISTORY LOG ──────────────────────────────────────────
    # SQLite database that records every add/search/delete.
    # Open with DB Browser for SQLite to inspect your memories.
    "history_db_path": HISTORY_DB,
}

# Initialise memory — this is done once at startup
print("  Initialising memory system (FAISS + Ollama)...")
try:
    memory = Memory.from_config(MEM0_CONFIG)
    print("  Memory system ready.")
except Exception as e:
    print(f"  Memory init warning: {e}")
    print("  Continuing without persistence (in-memory only).")
    memory = None

# The user ID ties all memories to one "person".
# Change this string to support multiple users.
USER_ID = "challenge4_user"


# ============================================================
# SECTION 3 — MEMORY HELPER FUNCTIONS
# ============================================================
# These plain Python functions wrap Mem0's API.
# They are called inside the @tool functions below.
# Keeping them separate makes the tool code clean and readable.
# ============================================================

def _save_to_memory(content: str) -> str:
    """Internal helper: add content to FAISS memory via Mem0."""
    if memory is None:
        return "Memory system unavailable."
    try:
        result = memory.add(content, user_id=USER_ID)
        stored = [r.get("memory", "") for r in result.get("results", [])] \
                 if isinstance(result, dict) else []
        return f"Stored: {'; '.join(stored)}" if stored else "Information stored."
    except Exception as e:
        return f"Memory save error: {e}"


def _search_memory(query: str, limit: int = 5) -> str:
    """Internal helper: search FAISS memory by semantic similarity."""
    if memory is None:
        return "Memory system unavailable."
    try:
        results = memory.search(query, user_id=USER_ID, limit=limit)
        items = results.get("results", []) if isinstance(results, dict) else []
        if not items:
            return "No relevant memories found."
        lines = [f"  - {item.get('memory', str(item))}" for item in items]
        return "Relevant memories:\n" + "\n".join(lines)
    except Exception as e:
        return f"Memory search error: {e}"


def _list_all_memories() -> str:
    """Internal helper: return all stored memories for this user."""
    if memory is None:
        return "Memory system unavailable."
    try:
        results = memory.get_all(user_id=USER_ID)
        items = results.get("results", []) if isinstance(results, dict) else []
        if not items:
            return "No memories stored yet. Tell me something about yourself!"
        lines = [f"  {i+1}. {item.get('memory', str(item))}"
                 for i, item in enumerate(items)]
        return "Everything I remember about you:\n" + "\n".join(lines)
    except Exception as e:
        return f"Memory list error: {e}"


# ============================================================
# SECTION 4 — TOOL DEFINITIONS
# ============================================================
# Each function decorated with @tool becomes a callable skill
# for the agent. The docstring is the description the LLM reads
# to decide WHEN to call each tool.
#
# We have 6 tools total:
#   Utility tools  → calculator, get_weather, age_calculator
#   Memory tools   → remember_info, recall_info, show_memories
# ============================================================


# ── UTILITY TOOL 1: CALCULATOR ────────────────────────────
@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result.
    Use this for ANY arithmetic: addition, subtraction,
    multiplication, division, percentages, powers, etc.
    Pass the expression as a string, e.g. '25 * 4 + 10'.
    """
    try:
        # eval() with empty builtins prevents code injection
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {expression} = {result}"
    except ZeroDivisionError:
        return "Error: Cannot divide by zero."
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


# ── UTILITY TOOL 2: WEATHER ───────────────────────────────
@tool
def get_weather(city: str) -> str:
    """
    Get the current weather for a given city.
    Returns temperature (°F), condition, humidity, and wind speed.
    Use this whenever the user asks about weather or temperature.
    """
    # Simulated local data — swap in a real API call for production
    weather_data = {
        "new york":    {"temp": 72,  "condition": "Partly Cloudy", "humidity": 65, "wind": 12},
        "london":      {"temp": 59,  "condition": "Rainy",         "humidity": 80, "wind": 15},
        "tokyo":       {"temp": 77,  "condition": "Sunny",         "humidity": 70, "wind": 8},
        "sydney":      {"temp": 68,  "condition": "Clear",         "humidity": 55, "wind": 10},
        "paris":       {"temp": 63,  "condition": "Cloudy",        "humidity": 75, "wind": 9},
        "dubai":       {"temp": 95,  "condition": "Hot and Sunny", "humidity": 40, "wind": 14},
        "mumbai":      {"temp": 88,  "condition": "Humid",         "humidity": 85, "wind": 11},
        "hyderabad":   {"temp": 91,  "condition": "Hot and Clear", "humidity": 45, "wind": 7},
        "bangalore":   {"temp": 78,  "condition": "Pleasant",      "humidity": 60, "wind": 6},
        "chicago":     {"temp": 65,  "condition": "Windy",         "humidity": 58, "wind": 20},
        "los angeles": {"temp": 80,  "condition": "Sunny",         "humidity": 35, "wind": 5},
        "toronto":     {"temp": 55,  "condition": "Overcast",      "humidity": 72, "wind": 13},
        "delhi":       {"temp": 98,  "condition": "Hot and Hazy",  "humidity": 50, "wind": 9},
        "chennai":     {"temp": 93,  "condition": "Hot and Humid", "humidity": 80, "wind": 10},
    }
    data = weather_data.get(city.lower().strip())
    if data:
        return (
            f"Weather in {city.title()}:\n"
            f"  Temperature : {data['temp']}°F\n"
            f"  Condition   : {data['condition']}\n"
            f"  Humidity    : {data['humidity']}%\n"
            f"  Wind Speed  : {data['wind']} mph"
        )
    return (
        f"No data for '{city}'.\n"
        f"Available cities: {', '.join(c.title() for c in weather_data)}"
    )


# ── UTILITY TOOL 3: AGE CALCULATOR ────────────────────────
@tool
def age_calculator(birth_year: int, birth_month: int = 1, birth_day: int = 1) -> str:
    """
    Calculate someone's exact age from their birth date.
    Returns age in years, months, and days from today.
    Use this when the user asks how old they are or
    wants to calculate an age from a birth year/date.
    birth_year  : 4-digit year (required), e.g. 1995
    birth_month : month 1-12 (optional, default 1)
    birth_day   : day 1-31   (optional, default 1)
    """
    try:
        today     = date.today()
        birthdate = date(birth_year, birth_month, birth_day)
        if birthdate > today:
            return "Error: Birth date cannot be in the future."

        years = today.year - birthdate.year
        if (today.month, today.day) < (birthdate.month, birthdate.day):
            years -= 1

        months = today.month - birthdate.month
        if today.day < birthdate.day:
            months -= 1
        if months < 0:
            months += 12

        if today.day >= birthdate.day:
            period_start = date(today.year, today.month, birthdate.day)
        else:
            if today.month == 1:
                period_start = date(today.year - 1, 12, birthdate.day)
            else:
                period_start = date(today.year, today.month - 1, birthdate.day)
        days = (today - period_start).days

        return (
            f"Age for {birth_day:02d}/{birth_month:02d}/{birth_year}:\n"
            f"  Today      : {today.strftime('%d %B %Y')}\n"
            f"  Exact Age  : {years} years, {months} months, {days} days\n"
            f"  In Years   : {years}"
        )
    except ValueError as e:
        return f"Invalid date — {e}"


# ── MEMORY TOOL 1: REMEMBER ───────────────────────────────
@tool
def remember_info(information: str) -> str:
    """
    Save important information about the user to persistent memory.
    Call this IMMEDIATELY when the user shares:
      - Their name ("My name is ...")
      - Age, birthday, location, job, hobbies, preferences
      - Any personal fact they want remembered in future sessions.
    The information persists between sessions via FAISS on disk.
    """
    result = _save_to_memory(information)
    print(f"\n  [MEMORY SAVED] {information[:70]}")
    return result


# ── MEMORY TOOL 2: RECALL ─────────────────────────────────
@tool
def recall_info(query: str) -> str:
    """
    Search persistent memory for information relevant to the query.
    Call this BEFORE answering personal questions such as:
      - "What is my name?"
      - "What do I like?"
      - "Where am I from?"
      - "Do you remember what I told you?"
    Returns the most semantically similar stored memories.
    """
    result = _search_memory(query)
    print(f"\n  [MEMORY RECALL] Query: '{query}'")
    print(f"  [MEMORY RECALL] Found: {result[:80]}")
    return result


# ── MEMORY TOOL 3: SHOW ALL ───────────────────────────────
@tool
def show_memories(placeholder: str = "") -> str:
    """
    List ALL memories stored for the current user.
    Call this when the user asks:
      - "What do you know about me?"
      - "Show me all my memories"
      - "What have I told you?"
    The placeholder argument is not used — pass an empty string.
    """
    result = _list_all_memories()
    print("\n  [MEMORY LIST] Listing all stored memories...")
    return result


# ============================================================
# SECTION 5 — MODEL SETUP
# ============================================================
# temperature=0.3 balances accuracy (for tool calls & memory)
# with natural conversational tone (for free-form chat).
# ============================================================

model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3.2:3b",
    temperature=0.3,
)


# ============================================================
# SECTION 6 — SYSTEM PROMPT
# ============================================================
# This is the agent's "job description" — it tells the model:
#   1. What tools are available and when to use each one
#   2. Strict rules so it doesn't guess or skip memory lookups
# A good system prompt is the most important part of any agent.
# ============================================================

SYSTEM_PROMPT = """
You are a capable, friendly AI assistant with both utility tools
and a persistent memory system. You remember information across
conversations — even if the user restarts the session.

=== UTILITY TOOLS ===

1. calculator(expression)
   → Use for ANY math: "What is 15% of 2500?", "What is 99 * 88?"
   → Pass the full expression as a string: '15 / 100 * 2500'

2. get_weather(city)
   → Use when asked about weather, temperature, or conditions.
   → Pass the city name as a string: 'Hyderabad'

3. age_calculator(birth_year, birth_month, birth_day)
   → Use when asked about someone's age or birth date.
   → birth_year is required. month and day are optional integers.

=== MEMORY TOOLS ===

4. remember_info(information)
   → Call IMMEDIATELY when the user shares personal facts:
     name, age, location, job, hobbies, favourite things, etc.
   → Also call when they say "remember that..." or "note that..."

5. recall_info(query)
   → Call BEFORE answering any personal question.
   → Use a descriptive query: "user's name", "user's hobbies"

6. show_memories(placeholder="")
   → Call when asked: "what do you know about me?",
     "show my memories", "what have I told you?"

=== STRICT RULES ===
- NEVER guess personal information — always use recall_info first.
- ALWAYS call remember_info when the user shares personal details.
- ALWAYS use calculator for math instead of computing in your head.
- ALWAYS use get_weather for weather questions.
- After any tool call, respond naturally and conversationally.
- If recall_info returns "No relevant memories found", be honest
  and ask the user to share that information.
"""


# ============================================================
# SECTION 7 — AGENT CREATION
# ============================================================
# All 6 tools are registered here.
# Strands passes the tool list to the model so it knows
# which functions are available to call during the loop.
# ============================================================

agent = Agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[
        # Utility tools
        calculator,
        get_weather,
        age_calculator,
        # Memory tools
        remember_info,
        recall_info,
        show_memories,
    ],
)


# ============================================================
# SECTION 8 — INTERACTIVE CHAT LOOP
# ============================================================
# The loop keeps the conversation going until the user quits.
# Two built-in shortcuts bypass the agent for quick ops:
#   'memory' → print everything stored in FAISS
#   'clear'  → wipe all memories after confirmation
# All other input goes to the agent, which decides which
# tool(s) to call and returns a natural language response.
# ============================================================

def main():
    print()
    print("=" * 65)
    print("  Challenge 4 - Full Agent  |  Strands SDK + Ollama")
    print("  Model    : llama3.2:3b  (local)")
    print("  Tools    : calculator | weather | age calculator")
    print("  Memory   : Mem0 + FAISS  (persistent across sessions)")
    print("  Embedder : nomic-embed-text via Ollama")
    print("-" * 65)
    print("  Try asking:")
    print("    > My name is Thamarai and I am from Hyderabad")
    print("    > I love playing chess and reading sci-fi")
    print("    > What is my name?")
    print("    > What is the weather in Hyderabad?")
    print("    > What is 25% of 8400?")
    print("    > How old is someone born on 10 June 1995?")
    print("    > What do you know about me?")
    print("-" * 65)
    print("  Commands: 'memory' = list all | 'clear' = wipe memory")
    print("  Type 'exit' or 'quit' to stop.")
    print("=" * 65)

    while True:
        # ── Get user input ───────────────────────────────────
        user_input = input("\nYou: ").strip()

        # ── Exit ─────────────────────────────────────────────
        if user_input.lower() in ("exit", "quit", ""):
            print("\nGoodbye! Your memories are saved and will be")
            print("available next time you run this agent.")
            break

        # ── Shortcut: list all memories ──────────────────────
        # Bypass the agent and print directly from FAISS
        if user_input.lower() == "memory":
            print("\n" + _list_all_memories())
            continue

        # ── Shortcut: clear all memories ─────────────────────
        # Ask for confirmation before wiping the index
        if user_input.lower() == "clear":
            confirm = input("  Clear ALL memories? This cannot be undone. (yes/no): ")
            if confirm.strip().lower() == "yes" and memory is not None:
                try:
                    all_mems = memory.get_all(user_id=USER_ID)
                    items = all_mems.get("results", []) \
                            if isinstance(all_mems, dict) else []
                    for item in items:
                        memory.delete(item["id"])
                    print(f"  Cleared {len(items)} memories.")
                except Exception as e:
                    print(f"  Clear error: {e}")
            else:
                print("  Clear cancelled.")
            continue

        # ── Normal turn: send to agent ────────────────────────
        # The agent:
        #   1. Reads the user message
        #   2. Decides which tool(s) to call (if any)
        #   3. Calls the tool and gets the result
        #   4. Passes the result back to the model
        #   5. Returns a final natural-language response
        response = agent(user_input)
        print(f"\nAgent: {response}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
