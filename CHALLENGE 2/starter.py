# ============================================================
# Challenge 2 - Tools Agent using Strands SDK + Ollama
# Model: llama3.2:3b (runs 100% locally, no cloud needed)
# Tools: calculator, weather, age_calculator
# ============================================================

# --- IMPORTS ---
from datetime import date          # Used in the age calculator tool
from strands import Agent, tool    # 'tool' decorator turns a Python function into an agent tool
from strands.models.ollama import OllamaModel


# ============================================================
# SECTION 1 — TOOL DEFINITIONS
# ============================================================
# The @tool decorator does three things:
#   1. Registers the function so the agent knows it exists
#   2. Uses the function name as the tool name
#   3. Uses the docstring as the tool description (the model
#      reads this to decide WHEN to call the tool)
# The type hints (int, float, str) tell the model what
# kind of argument to pass in.
# ============================================================


# --- TOOL 1: CALCULATOR ---
@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result.
    Use this tool for any arithmetic: addition, subtraction,
    multiplication, division, powers, and more.
    Examples: '2 + 2', '10 * 5', '(100 / 4) ** 2'
    """
    try:
        # eval() computes the math expression safely.
        # We restrict the available names to empty dicts
        # so no harmful code can be injected.
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {expression} = {result}"
    except ZeroDivisionError:
        return "Error: Cannot divide by zero."
    except Exception as e:
        return f"Error evaluating expression '{expression}': {str(e)}"


# --- TOOL 2: WEATHER ---
@tool
def get_weather(city: str) -> str:
    """
    Get the current weather for a given city.
    Returns temperature, condition, humidity, and wind speed.
    Use this tool whenever the user asks about weather,
    temperature, or conditions in a city.
    """
    # NOTE: This is simulated data since we're running fully locally
    # with no internet. In a real project you would call a weather
    # API (e.g. OpenWeatherMap) here instead.
    weather_data = {
        "new york":     {"temp": 72,  "condition": "Partly Cloudy", "humidity": 65, "wind": 12},
        "london":       {"temp": 59,  "condition": "Rainy",         "humidity": 80, "wind": 15},
        "tokyo":        {"temp": 77,  "condition": "Sunny",         "humidity": 70, "wind": 8},
        "sydney":       {"temp": 68,  "condition": "Clear",         "humidity": 55, "wind": 10},
        "paris":        {"temp": 63,  "condition": "Cloudy",        "humidity": 75, "wind": 9},
        "dubai":        {"temp": 95,  "condition": "Hot and Sunny", "humidity": 40, "wind": 14},
        "mumbai":       {"temp": 88,  "condition": "Humid",         "humidity": 85, "wind": 11},
        "hyderabad":    {"temp": 91,  "condition": "Hot and Clear", "humidity": 45, "wind": 7},
        "bangalore":    {"temp": 78,  "condition": "Pleasant",      "humidity": 60, "wind": 6},
        "chicago":      {"temp": 65,  "condition": "Windy",         "humidity": 58, "wind": 20},
        "los angeles":  {"temp": 80,  "condition": "Sunny",         "humidity": 35, "wind": 5},
        "toronto":      {"temp": 55,  "condition": "Overcast",      "humidity": 72, "wind": 13},
    }

    # Normalize the city name to lowercase for a reliable lookup
    city_lower = city.lower().strip()
    data = weather_data.get(city_lower)

    if data:
        return (
            f"Weather in {city.title()}:\n"
            f"  Temperature : {data['temp']}°F\n"
            f"  Condition   : {data['condition']}\n"
            f"  Humidity    : {data['humidity']}%\n"
            f"  Wind Speed  : {data['wind']} mph"
        )
    else:
        return (
            f"Weather data for '{city}' is not available in the local database.\n"
            f"Available cities: {', '.join(c.title() for c in weather_data)}"
        )


# --- TOOL 3: AGE CALCULATOR ---
@tool
def age_calculator(birth_year: int, birth_month: int = 1, birth_day: int = 1) -> str:
    """
    Calculate a person's exact age given their birth date.
    Returns their age in years, months, and days.
    Use this tool whenever the user asks how old someone is,
    or wants to calculate an age from a birth date.
    birth_year  : 4-digit year (e.g. 1990)
    birth_month : month number 1-12 (default: 1)
    birth_day   : day number 1-31  (default: 1)
    """
    try:
        today = date.today()
        birthdate = date(birth_year, birth_month, birth_day)

        # Reject future birthdates
        if birthdate > today:
            return "Error: Birth date cannot be in the future."

        # Calculate full years
        years = today.year - birthdate.year
        # Subtract 1 if we haven't yet passed the birthday this year
        if (today.month, today.day) < (birthdate.month, birthdate.day):
            years -= 1

        # Calculate remaining months
        months = today.month - birthdate.month
        if today.day < birthdate.day:
            months -= 1
        if months < 0:
            months += 12

        # Calculate remaining days
        # Find the start of the current month-period after the last birthday
        if today.day >= birthdate.day:
            period_start = date(today.year, today.month, birthdate.day)
        else:
            # Go back one month
            if today.month == 1:
                period_start = date(today.year - 1, 12, birthdate.day)
            else:
                period_start = date(today.year, today.month - 1, birthdate.day)
        days = (today - period_start).days

        return (
            f"Age Calculation for {birth_day:02d}/{birth_month:02d}/{birth_year}:\n"
            f"  Today's Date : {today.strftime('%d %B %Y')}\n"
            f"  Exact Age    : {years} years, {months} months, {days} days\n"
            f"  Age in Years : {years}"
        )
    except ValueError as e:
        return f"Error: Invalid date — {str(e)}"


# ============================================================
# SECTION 2 — MODEL SETUP
# ============================================================
# Same OllamaModel as Challenge 1, but we lower the temperature
# slightly to 0.3 so the model makes precise, reliable tool
# calls rather than guessing or hallucinating numbers.
# ============================================================

model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3.2:3b",
    temperature=0.3,      # Lower = more precise tool use
)


# ============================================================
# SECTION 3 — SYSTEM PROMPT
# ============================================================
# We explicitly tell the model about the three tools so it
# knows to use them instead of trying to answer from memory.
# ============================================================

SYSTEM_PROMPT = """
You are a helpful AI assistant with access to three tools:

1. calculator      — use this for any math or arithmetic questions
2. get_weather     — use this when asked about weather in a city
3. age_calculator  — use this to calculate someone's age from a birth date

RULES:
- Always use the appropriate tool instead of guessing the answer.
- For math questions, pass the expression as a string (e.g. '15 * 8').
- For weather, pass the city name as a string (e.g. 'Tokyo').
- For age, pass the birth year (required), month, and day as integers.
- After using a tool, explain the result in a friendly, clear way.
"""


# ============================================================
# SECTION 4 — AGENT CREATION
# ============================================================
# The 'tools' list is new compared to Challenge 1.
# Strands registers each tool with the model so it can
# call them during the agentic reasoning loop.
# ============================================================

agent = Agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[calculator, get_weather, age_calculator],  # <-- tools registered here
)


# ============================================================
# SECTION 5 — INTERACTIVE CHAT LOOP
# ============================================================

def main():
    print("=" * 58)
    print("  Challenge 2 - Tools Agent  |  Strands SDK + Ollama")
    print("  Model  : llama3.2:3b  (running locally)")
    print("  Tools  : calculator | weather | age calculator")
    print("-" * 58)
    print("  Try asking:")
    print("    > What is 1234 * 5678?")
    print("    > What is the weather in Tokyo?")
    print("    > How old is someone born on 15 March 1990?")
    print("  Type 'exit' or 'quit' to stop.")
    print("=" * 58)

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ("exit", "quit", ""):
            print("Goodbye!")
            break

        # The agent automatically decides which tool to call
        # (if any) based on the user's question, then returns
        # a final natural-language response.
        response = agent(user_input)
        print(f"\nAgent: {response}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
