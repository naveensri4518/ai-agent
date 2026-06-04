# ============================================================
# Challenge 3 - Memory Agent using Strands SDK + Ollama
# Model    : llama3.2:3b  (runs 100% locally)
# Memory   : Mem0  +  FAISS  (persistent, local vector store)
# Embedder : nomic-embed-text via Ollama  (local embeddings)
# ============================================================
#
# HOW MEMORY WORKS HERE:
#   1. You say something → agent replies AND stores key facts
#      in a FAISS index file on your disk (memory.index)
#   2. Next time you ask a question → agent searches the FAISS
#      index for relevant past facts and injects them as context
#   3. This works across sessions — restart the script and it
#      still remembers what you told it before!
# ============================================================

# ============================================================
# SETUP — run these commands ONCE before running this script:
#
#   pip install mem0ai faiss-cpu sentence-transformers
#   ollama pull llama3.2:3b
#   ollama pull nomic-embed-text       <-- local embedding model
#   ollama serve                       <-- keep running in bg
# ============================================================


# --- STANDARD LIBRARY IMPORTS ---
import os           # For building file paths
import json         # For saving/loading memory to disk
import warnings     # For suppressing noisy library warnings

# Suppress unimportant warnings from underlying libraries
warnings.filterwarnings("ignore")

# --- MEM0 IMPORT ---
# Memory is the core Mem0 class.
# Memory.from_config() lets us wire up:
#   - which vector store to use (FAISS = local file)
#   - which LLM to use for memory extraction (Ollama)
#   - which embedder to use (Ollama nomic-embed-text)
from mem0 import Memory

# --- STRANDS IMPORTS ---
from strands import Agent, tool
from strands.models.ollama import OllamaModel


# ============================================================
# SECTION 1 — MEM0 CONFIGURATION
# ============================================================
# This config tells Mem0 exactly how to store and retrieve
# memories — all locally, no cloud, no API keys needed.
# ============================================================

# Where to save the persistent memory files on disk
MEMORY_DIR  = os.path.join(os.path.dirname(__file__), "memory_store")
FAISS_PATH  = os.path.join(MEMORY_DIR, "memory.index")   # FAISS vector index
HISTORY_DB  = os.path.join(MEMORY_DIR, "history.db")     # SQLite history log

# Create the folder if it doesn't exist yet
os.makedirs(MEMORY_DIR, exist_ok=True)

MEM0_CONFIG = {
    # ── VECTOR STORE ─────────────────────────────────────────
    # FAISS stores memories as embedding vectors in a local
    # .index file. No server needed — it's just a file.
    "vector_store": {
        "provider": "faiss",
        "config": {
            "index_path": FAISS_PATH,   # Where to save the index
        },
    },

    # ── EMBEDDER ─────────────────────────────────────────────
    # nomic-embed-text converts text → vectors (numbers).
    # These vectors are stored in FAISS so similar text can
    # be found later via semantic search.
    # Pull it once with: ollama pull nomic-embed-text
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text",
            "ollama_base_url": "http://localhost:11434",
        },
    },

    # ── LLM ──────────────────────────────────────────────────
    # Mem0 uses the LLM internally to intelligently *extract*
    # facts from your messages before storing them.
    # Example: "My name SLag and I love Python"
    #  → Mem0 extracts: ["Name: Thamarai", "Loves Python"]
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "llama3.2:3b",
            "temperature": 0,              # 0 = deterministic extraction
            "max_tokens": 1000,
            "ollama_base_url": "http://localhost:11434",
        },
    },

    # ── HISTORY DATABASE ─────────────────────────────────────
    # SQLite file that logs every add/search operation.
    # Useful for debugging — you can open it with any SQLite
    # browser to see all stored memories.
    "history_db_path": HISTORY_DB,
}


# ============================================================
# SECTION 2 — INITIALISE MEM0 MEMORY
# ============================================================
# Memory.from_config() reads the config above and sets up:
#   - the FAISS index (creates it if it doesn't exist)
#   - the Ollama embedder connection
#   - the Ollama LLM connection
# ============================================================

print("Initialising memory system...")
try:
    memory = Memory.from_config(MEM0_CONFIG)
    print("Memory system ready.")
except Exception as e:
    print(f"Memory init error: {e}")
    print("Falling back to simple in-memory storage (no persistence).")
    memory = None


# ============================================================
# SECTION 3 — MEMORY HELPER FUNCTIONS
# ============================================================
# These thin wrappers keep the main code clean and add
# readable print statements so you can see memory in action.
# ============================================================

# Fixed user ID — change this to support multiple users
USER_ID = "challenge3_user"

def save_to_memory(content: str) -> str:
    """
    Store a new piece of information in the FAISS memory.
    Mem0 extracts key facts from 'content' before storing.
    """
    if memory is None:
        return "Memory unavailable."
    try:
        result = memory.add(content, user_id=USER_ID)
        # result is a list of stored memory objects
        stored = [r.get("memory", "") for r in result.get("results", [])] if isinstance(result, dict) else []
        if stored:
            return f"Stored: {'; '.join(stored)}"
        return "Information processed and stored."
    except Exception as e:
        return f"Memory save error: {e}"


def search_memory(query: str, limit: int = 5) -> str:
    """
    Search FAISS for memories relevant to the query.
    Returns a formatted string of matching facts.
    """
    if memory is None:
        return "No memories available."
    try:
        results = memory.search(query, user_id=USER_ID, limit=limit)
        # results is a dict with a "results" key
        items = results.get("results", []) if isinstance(results, dict) else []
        if not items:
            return "No relevant memories found."
        memories_text = "\n".join(
            f"  - {item.get('memory', str(item))}"
            for item in items
        )
        return f"Relevant memories:\n{memories_text}"
    except Exception as e:
        return f"Memory search error: {e}"


def list_all_memories() -> str:
    """
    List every memory stored for this user.
    """
    if memory is None:
        return "No memories available."
    try:
        results = memory.get_all(user_id=USER_ID)
        items = results.get("results", []) if isinstance(results, dict) else []
        if not items:
            return "No memories stored yet."
        lines = [f"  {i+1}. {item.get('memory', str(item))}" for i, item in enumerate(items)]
        return "All stored memories:\n" + "\n".join(lines)
    except Exception as e:
        return f"Memory list error: {e}"


# ============================================================
# SECTION 4 — STRANDS TOOL DEFINITIONS
# ============================================================
# Three tools the agent can call:
#   remember_info  → save something to memory
#   recall_info    → search memory for something
#   show_memories  → list everything in memory
# ============================================================

@tool
def remember_info(information: str) -> str:
    """
    Save important information about the user to persistent memory.
    Use this whenever the user shares personal details, preferences,
    or facts they want the agent to remember.
    Examples: name, age, location, hobbies, favourite things.
    """
    result = save_to_memory(information)
    print(f"  [MEMORY SAVED] {information[:60]}...")
    return result


@tool
def recall_info(query: str) -> str:
    """
    Search persistent memory for information relevant to the query.
    Use this before answering personal questions to check if the
    agent already knows the answer from a previous conversation.
    Examples: 'What is the user's name?', 'What does the user like?'
    """
    result = search_memory(query)
    print(f"  [MEMORY SEARCH] '{query}' → {result[:80]}...")
    return result


@tool
def show_memories(placeholder: str = "") -> str:
    """
    List all memories stored for the current user.
    Use this when the user asks 'what do you remember about me?'
    or 'show all memories' or 'what do you know about me?'.
    The placeholder argument is unused — just pass an empty string.
    """
    result = list_all_memories()
    print(f"  [MEMORY LIST] Listing all memories...")
    return result


# ============================================================
# SECTION 5 — MODEL SETUP
# ============================================================

model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3.2:3b",
    temperature=0.3,    # Low for reliable, consistent responses
)


# ============================================================
# SECTION 6 — SYSTEM PROMPT
# ============================================================
# We instruct the agent to ALWAYS check memory before answering
# personal questions, and ALWAYS save new personal information.
# ============================================================

SYSTEM_PROMPT = """
You are a helpful, friendly AI assistant with persistent memory.
You have three memory tools available:

1. remember_info(information) 
   → Call this IMMEDIATELY when the user shares:
     - Their name ("My name is ...")
     - Age, location, job, hobbies, preferences
     - Any personal fact they want you to remember

2. recall_info(query)
   → Call this BEFORE answering any personal question like:
     - "What is my name?"
     - "What do I like?"
     - "Do you remember what I told you?"

3. show_memories(placeholder="")
   → Call this when the user asks you to list what you know about them.

STRICT RULES:
- NEVER guess personal information — always use recall_info first.
- ALWAYS call remember_info when the user tells you something personal.
- After storing or recalling, respond naturally and conversationally.
- If memory returns "No relevant memories found", say you don't know yet
  and ask them to tell you.
"""


# ============================================================
# SECTION 7 — AGENT CREATION
# ============================================================

agent = Agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[remember_info, recall_info, show_memories],
)


# ============================================================
# SECTION 8 — INTERACTIVE CHAT LOOP
# ============================================================

def main():
    print()
    print("=" * 62)
    print("  Challenge 3 - Memory Agent  |  Strands + Mem0 + FAISS")
    print("  Model    : llama3.2:3b  (local)")
    print("  Memory   : FAISS  (persists to disk between sessions)")
    print("  Embedder : nomic-embed-text via Ollama")
    print("-" * 62)
    print("  Try saying:")
    print("    > My name is Thamarai")
    print("    > I love playing chess and reading sci-fi books")
    print("    > What is my name?")
    print("    > What are my hobbies?")
    print("    > What do you know about me?")
    print("  Commands: 'memory' = list all | 'clear' = wipe memory")
    print("  Type 'exit' or 'quit' to stop.")
    print("=" * 62)

    while True:
        user_input = input("\nYou: ").strip()

        # ── Exit ────────────────────────────────────────────
        if user_input.lower() in ("exit", "quit", ""):
            print("Goodbye! Your memories are saved for next time.")
            break

        # ── Quick command: list all memories ───────────────
        if user_input.lower() == "memory":
            print("\n" + list_all_memories())
            continue

        # ── Quick command: clear all memories ──────────────
        if user_input.lower() == "clear":
            confirm = input("  Are you sure you want to clear all memories? (yes/no): ")
            if confirm.lower() == "yes" and memory is not None:
                try:
                    all_mems = memory.get_all(user_id=USER_ID)
                    items = all_mems.get("results", []) if isinstance(all_mems, dict) else []
                    for item in items:
                        memory.delete(item["id"])
                    print("  All memories cleared.")
                except Exception as e:
                    print(f"  Clear error: {e}")
            else:
                print("  Clear cancelled.")
            continue

        # ── Normal conversation — pass to agent ────────────
        # The agent will decide whether to call remember_info,
        # recall_info, or show_memories based on the message,
        # then return a natural language reply.
        response = agent(user_input)
        print(f"\nAgent: {response}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
