# ============================================================
# Challenge 5 - MCP Chatbot using Strands SDK + Ollama
# Model    : llama3.2:3b  (runs 100% locally)
# Protocol : Model Context Protocol (MCP) over stdio
# MCP Tools: mcp_calculator | mcp_weather | mcp_datetime
#            mcp_note_manager
# ============================================================
#
# WHAT IS MCP?
# ────────────
# MCP (Model Context Protocol) is an open standard for
# connecting AI agents to external tool servers.
# Instead of hardcoding tools into your agent script,
# you run a separate MCP server process that *advertises*
# its tools. The agent discovers them automatically at startup.
#
# This is the industry pattern used in production AI systems:
#
#   ┌─────────────────────────────────────────────────────┐
#   │  starter.py  (MCP Client + Strands Agent)           │
#   │      ↕  JSON-RPC over stdin/stdout (stdio)          │
#   │  mcp_server.py  (MCP Server — tool provider)        │
#   └─────────────────────────────────────────────────────┘
#
# HOW STRANDS USES MCP:
#   1. MCPClient launches mcp_server.py as a subprocess
#   2. They exchange an MCP "handshake" — the server sends
#      a list of all available tools and their schemas
#   3. Strands wraps each MCP tool so the Agent can call it
#      exactly like a regular @tool function
#   4. When you ask a question, the agent calls the right
#      MCP tool, gets the result, and replies naturally
#
# ============================================================
# SETUP — run these commands ONCE:
#
#   pip install strands-agents mcp
#   ollama pull llama3.2:3b
#   ollama serve                 ← keep this running in bg
#
# Then run:
#   python starter.py            ← this file only; it auto-
#                                   launches mcp_server.py
# ============================================================


# ============================================================
# SECTION 1 — IMPORTS
# ============================================================

import sys          # sys.executable — path to current Python
import os           # File path building
import warnings
warnings.filterwarnings("ignore")

# ── Strands imports ───────────────────────────────────────
from strands import Agent
from strands.models.ollama import OllamaModel

# ── MCP client imports ────────────────────────────────────
# MCPClient        : manages the MCP server process lifecycle
#                    and the JSON-RPC communication channel.
#                    Requires a *callable* that returns the
#                    transport — not a params object directly.
# stdio_client     : async context manager that spawns the
#                    subprocess and wires up stdin/stdout pipes.
# StdioServerParameters: dataclass describing the subprocess
#                    command, args, and environment.
from strands.tools.mcp import MCPClient
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client


# ============================================================
# SECTION 2 — MCP SERVER PARAMETERS + TRANSPORT CALLABLE
# ============================================================
# MCPClient requires a *callable* (a zero-argument function)
# that returns the async transport context manager.
# This is because Strands may need to reconnect the transport,
# so it calls the function rather than using a static object.
#
# Pattern:
#   server_params  →  StdioServerParameters (what to launch)
#   transport_fn   →  lambda that returns stdio_client(params)
#
# Strands calls: transport_fn()  →  stdio_client(server_params)
#   which spawns: python mcp_server.py
#   and wires up stdin/stdout pipes for JSON-RPC.
# ============================================================

# Absolute path to mcp_server.py (same folder as this file)
MCP_SERVER_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "mcp_server.py"
)

# StdioServerParameters: describes which subprocess to launch
server_params = StdioServerParameters(
    command=sys.executable,         # Full path to python.exe
    args=[MCP_SERVER_PATH],         # Script to run as MCP server
    env=dict(os.environ),           # Pass current environment vars
)

# transport_fn: the callable MCPClient actually needs.
# It wraps stdio_client so Strands can call it like a function.
# lambda: stdio_client(server_params)
# means "when called, return the stdio transport for server_params"
transport_fn = lambda: stdio_client(server_params)


# ============================================================
# SECTION 3 — MODEL SETUP
# ============================================================

model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3.2:3b",
    temperature=0.3,    # Low = precise tool calls + reliable output
)


# ============================================================
# SECTION 4 — SYSTEM PROMPT
# ============================================================
# We list all 4 MCP tools so the model knows exactly when
# and how to use each one.
# ============================================================

SYSTEM_PROMPT = """
You are an innovative MCP-powered AI chatbot built with the
Strands SDK and Ollama. You connect to a local MCP server
that provides 4 tools:

1. mcp_calculator(expression)
   → Use for ANY math question.
   → Pass expression as string: '25 * 4', 'sqrt(144)', '10 ** 3'

2. mcp_weather(city)
   → Use when asked about weather in a city.
   → Pass city name: 'Tokyo', 'Hyderabad', 'London'

3. mcp_datetime(info_type)
   → Use for date/time questions.
   → info_type: 'all' | 'date' | 'time' | 'day' | 'year'

4. mcp_note_manager(action, content)
   → Use for saving and listing notes.
   → action: 'save' (requires content) | 'list' | 'clear'

RULES:
- ALWAYS use the correct tool instead of guessing.
- For math, pass the full expression as a string.
- For date/time, use info_type='all' unless asked for specific info.
- After a tool call, explain the result in a friendly way.
- If a city isn't available for weather, say so and list options.
"""


# ============================================================
# SECTION 5 — AGENT + MCP WIRING
# ============================================================
# This is the key pattern for MCP in Strands:
#
#   with MCPClient(server_params) as mcp_client:
#       tools = mcp_client.list_tools_sync()
#       agent = Agent(tools=tools, ...)
#       agent("your question")
#
# MCPClient is a context manager:
#   __enter__ : launches mcp_server.py subprocess,
#               performs the MCP handshake,
#               returns a client with discovered tools
#   __exit__  : terminates the subprocess cleanly
#
# mcp_client.list_tools_sync() returns a list of Strands-
# compatible tool wrappers — one per tool exposed by the server.
# These work identically to @tool-decorated functions.
# ============================================================

def main():
    print()
    print("=" * 65)
    print("  Challenge 5 - MCP Chatbot  |  Strands SDK + Ollama")
    print("  Model    : llama3.2:3b  (local)")
    print("  Protocol : Model Context Protocol (MCP) over stdio")
    print("  Server   : mcp_server.py  (launched automatically)")
    print("-" * 65)
    print("  MCP Tools available:")
    print("    mcp_calculator   — math and arithmetic")
    print("    mcp_weather      — weather by city")
    print("    mcp_datetime     — current date and time")
    print("    mcp_note_manager — save, list, clear notes")
    print("-" * 65)
    print("  Try asking:")
    print("    > What is sqrt(225) + 10 * 3?")
    print("    > What is the weather in Tokyo?")
    print("    > What day is today?")
    print("    > Save a note: Buy milk and eggs")
    print("    > Show my notes")
    print("    > What is today's date and what is 2 ** 10?")
    print("  Type 'exit' or 'quit' to stop.")
    print("=" * 65)

    # ── Launch MCP server and connect ─────────────────────
    # The 'with' block starts mcp_server.py as a subprocess.
    # The agent lives inside this block so the subprocess
    # stays alive for the entire chat session.
    print("\n  Connecting to MCP server...")

    try:
        with MCPClient(transport_fn) as mcp_client:

            # ── Discover tools from the MCP server ────────
            # The server sends back its full tool catalogue
            # during the MCP handshake. Strands wraps each
            # tool so the Agent can call them transparently.
            mcp_tools = mcp_client.list_tools_sync()
            tool_names = [t.tool_name if hasattr(t, 'tool_name') else str(t)
                          for t in mcp_tools]
            print(f"  MCP server connected. Tools discovered: {len(mcp_tools)}")
            print(f"  Tool names: {', '.join(tool_names)}")

            # ── Create the agent with MCP tools ───────────
            # mcp_tools is a plain Python list — you pass it
            # to Agent(tools=...) exactly like @tool functions.
            agent = Agent(
                model=model,
                system_prompt=SYSTEM_PROMPT,
                tools=mcp_tools,   # ← MCP tools injected here
            )

            print("\n  Agent ready. Start chatting!\n")

            # ── Chat loop ──────────────────────────────────
            while True:
                user_input = input("You: ").strip()

                # Exit
                if user_input.lower() in ("exit", "quit", ""):
                    print("\nGoodbye! MCP server shutting down.")
                    break

                # Help command — remind the user of available tools
                if user_input.lower() == "help":
                    print()
                    print("  Available MCP tools:")
                    print("    mcp_calculator(expression)")
                    print("      e.g. 'What is 15% of 5000?'")
                    print("    mcp_weather(city)")
                    print("      e.g. 'Weather in Hyderabad?'")
                    print("    mcp_datetime(info_type)")
                    print("      e.g. 'What time is it?', 'What day is today?'")
                    print("    mcp_note_manager(action, content)")
                    print("      e.g. 'Save a note: Call mom'")
                    print("      e.g. 'Show my notes'")
                    print()
                    continue

                # Tools command — list discovered MCP tools
                if user_input.lower() == "tools":
                    print(f"\n  Discovered {len(mcp_tools)} MCP tool(s):")
                    for t in mcp_tools:
                        name = t.tool_name if hasattr(t, 'tool_name') else str(t)
                        print(f"    - {name}")
                    print()
                    continue

                # Normal turn — send to agent
                # The agent internally calls the MCP server
                # via JSON-RPC over stdin/stdout when a tool
                # is needed, then returns a plain text reply.
                response = agent(user_input)
                print(f"\nAgent: {response}\n")

    except FileNotFoundError:
        print(f"\n  ERROR: Could not find mcp_server.py at:")
        print(f"  {MCP_SERVER_PATH}")
        print("  Make sure mcp_server.py is in the same folder as starter.py.")

    except Exception as e:
        print(f"\n  ERROR connecting to MCP server: {e}")
        print("  Make sure 'mcp' is installed:  pip install mcp")
        print("  Make sure Ollama is running:   ollama serve")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
