# ============================================================
# Challenge 5 - MCP Server
# File     : mcp_server.py
# Protocol : Model Context Protocol (MCP) over stdio
# ============================================================
#
# WHAT IS AN MCP SERVER?
# ─────────────────────
# MCP (Model Context Protocol) is an open standard that lets
# AI agents discover and call tools hosted by any server,
# using a common JSON-RPC wire format.
#
# This server exposes 4 tools:
#   1. mcp_calculator    — evaluate math expressions
#   2. mcp_weather       — get simulated city weather
#   3. mcp_datetime      — current date, time, day, year
#   4. mcp_note_manager  — save / list / clear text notes
#
# TRANSPORT:
# The server communicates over STDIO (stdin / stdout).
# The Strands SDK launches this file as a subprocess and
# talks to it through pipes — no ports, no HTTP needed.
#
# HOW THE AGENT USES THIS:
#   starter.py  →  launches  →  mcp_server.py  (subprocess)
#       ↕  JSON-RPC over stdin/stdout  ↕
#   Agent discovers tools automatically via MCP handshake.
#
# ============================================================
# SETUP — install the MCP library once:
#   pip install mcp
# ============================================================

import json                         # JSON serialisation
import math                         # For advanced math in calculator
from datetime import datetime       # For the datetime tool
from mcp.server.fastmcp import FastMCP   # High-level MCP server helper

# ── Create the MCP server instance ───────────────────────────
# The name appears in the MCP handshake so the client knows
# which server it connected to.
mcp = FastMCP("LocalToolsServer")


# ============================================================
# MCP TOOL 1 — CALCULATOR
# ============================================================
# @mcp.tool() registers this Python function as an MCP tool.
# The function name  →  becomes the MCP tool name
# The docstring      →  becomes the tool description
# Type hints         →  become the JSON schema for parameters
# ============================================================

@mcp.tool()
def mcp_calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result.
    Supports basic arithmetic (+, -, *, /), powers (**),
    modulo (%), and math functions like sqrt, sin, cos, log.
    Example expressions: '2 + 2', '10 ** 3', 'sqrt(144)'
    """
    try:
        # Allow safe math functions from the math module
        safe_env = {
            "__builtins__": {},
            "sqrt":  math.sqrt,
            "sin":   math.sin,
            "cos":   math.cos,
            "tan":   math.tan,
            "log":   math.log,
            "log10": math.log10,
            "pi":    math.pi,
            "e":     math.e,
            "abs":   abs,
            "round": round,
        }
        result = eval(expression, safe_env, {})
        return f"{expression} = {result}"
    except ZeroDivisionError:
        return "Error: Cannot divide by zero."
    except Exception as ex:
        return f"Error evaluating '{expression}': {ex}"


# ============================================================
# MCP TOOL 2 — WEATHER
# ============================================================

@mcp.tool()
def mcp_weather(city: str) -> str:
    """
    Get current weather information for a given city.
    Returns temperature in Fahrenheit, sky condition,
    humidity percentage, and wind speed in mph.
    Available cities: New York, London, Tokyo, Sydney, Paris,
    Dubai, Mumbai, Hyderabad, Bangalore, Chicago, Los Angeles,
    Toronto, Delhi, Chennai.
    """
    weather_db = {
        "new york":    {"temp": 72,  "sky": "Partly Cloudy", "humidity": 65, "wind": 12},
        "london":      {"temp": 59,  "sky": "Rainy",          "humidity": 80, "wind": 15},
        "tokyo":       {"temp": 77,  "sky": "Sunny",          "humidity": 70, "wind": 8},
        "sydney":      {"temp": 68,  "sky": "Clear",          "humidity": 55, "wind": 10},
        "paris":       {"temp": 63,  "sky": "Cloudy",         "humidity": 75, "wind": 9},
        "dubai":       {"temp": 95,  "sky": "Hot and Sunny",  "humidity": 40, "wind": 14},
        "mumbai":      {"temp": 88,  "sky": "Humid",          "humidity": 85, "wind": 11},
        "hyderabad":   {"temp": 91,  "sky": "Hot and Clear",  "humidity": 45, "wind": 7},
        "bangalore":   {"temp": 78,  "sky": "Pleasant",       "humidity": 60, "wind": 6},
        "chicago":     {"temp": 65,  "sky": "Windy",          "humidity": 58, "wind": 20},
        "los angeles": {"temp": 80,  "sky": "Sunny",          "humidity": 35, "wind": 5},
        "toronto":     {"temp": 55,  "sky": "Overcast",       "humidity": 72, "wind": 13},
        "delhi":       {"temp": 98,  "sky": "Hot and Hazy",   "humidity": 50, "wind": 9},
        "chennai":     {"temp": 93,  "sky": "Hot and Humid",  "humidity": 80, "wind": 10},
    }
    data = weather_db.get(city.lower().strip())
    if data:
        return (
            f"Weather in {city.title()}:\n"
            f"  Temperature : {data['temp']}°F\n"
            f"  Condition   : {data['sky']}\n"
            f"  Humidity    : {data['humidity']}%\n"
            f"  Wind Speed  : {data['wind']} mph"
        )
    cities = ", ".join(c.title() for c in weather_db)
    return f"No weather data for '{city}'. Available: {cities}"


# ============================================================
# MCP TOOL 3 — DATETIME
# ============================================================

@mcp.tool()
def mcp_datetime(info_type: str = "all") -> str:
    """
    Return current date and/or time information.
    info_type options:
      'all'   → full date + time (default)
      'date'  → today's date only
      'time'  → current time only
      'day'   → day of the week
      'year'  → current year only
    """
    now = datetime.now()
    info_type = info_type.lower().strip()

    if info_type == "date":
        return f"Today's date: {now.strftime('%A, %d %B %Y')}"
    elif info_type == "time":
        return f"Current time: {now.strftime('%I:%M %p')}"
    elif info_type == "day":
        return f"Today is: {now.strftime('%A')}"
    elif info_type == "year":
        return f"Current year: {now.year}"
    else:  # "all" or anything else
        return (
            f"Date & Time Information:\n"
            f"  Date  : {now.strftime('%A, %d %B %Y')}\n"
            f"  Time  : {now.strftime('%I:%M:%S %p')}\n"
            f"  Year  : {now.year}\n"
            f"  Month : {now.strftime('%B')}\n"
            f"  Day   : {now.strftime('%A')}"
        )


# ============================================================
# MCP TOOL 4 — NOTE MANAGER
# ============================================================
# A simple in-memory note pad.
# Notes are stored in a Python list while the server runs.
# This demonstrates MCP tools that hold state.
# ============================================================

_notes: list[str] = []   # In-process note store

@mcp.tool()
def mcp_note_manager(action: str, content: str = "") -> str:
    """
    Manage a simple note pad with save, list, and clear actions.
    action options:
      'save'  → save a new note (requires content)
      'list'  → list all saved notes
      'clear' → delete all notes
    content : the note text (only needed for 'save')
    Examples:
      action='save', content='Buy groceries'
      action='list'
      action='clear'
    """
    action = action.lower().strip()

    if action == "save":
        if not content.strip():
            return "Error: Please provide note content to save."
        _notes.append(content.strip())
        return f"Note saved ({len(_notes)} total): '{content.strip()}'"

    elif action == "list":
        if not _notes:
            return "No notes saved yet. Use action='save' to add one."
        numbered = "\n".join(f"  {i+1}. {note}" for i, note in enumerate(_notes))
        return f"Your notes ({len(_notes)} total):\n{numbered}"

    elif action == "clear":
        count = len(_notes)
        _notes.clear()
        return f"Cleared {count} note(s)."

    else:
        return f"Unknown action '{action}'. Use: 'save', 'list', or 'clear'."


# ============================================================
# SERVER ENTRY POINT
# ============================================================
# mcp.run(transport="stdio") starts the MCP server and listens
# for JSON-RPC messages on stdin/stdout.
# This is called by Strands automatically when it launches
# this file as a subprocess — you don't run it manually.
# ============================================================

if __name__ == "__main__":
    mcp.run(transport="stdio")
