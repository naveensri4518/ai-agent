# ============================================================
# Challenge 1 - Simple AI Agent using Strands SDK + Ollama
# Model: llama3.2:3b (runs 100% locally, no cloud needed)
# ============================================================

# --- IMPORTS ---
# Agent is the core class that manages the agentic loop:
# it sends your message to the model, handles tool calls,
# and returns the final response.
from strands import Agent

# OllamaModel is the connector that tells Strands how to talk
# to your locally-running Ollama server (http://localhost:11434).
from strands.models.ollama import OllamaModel


# --- MODEL SETUP ---
# Create an OllamaModel pointing to llama3.2:3b.
# host      : where Ollama is running (default port is 11434)
# model_id  : the exact model tag you pulled with `ollama pull`
# temperature: 0.7 gives a good balance of creativity vs focus
#              (0.0 = deterministic, 1.0 = very creative)
model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3.2:3b",
    temperature=0.7,
)


# --- SYSTEM PROMPT ---
# This is the personality / role you give the agent.
# It is sent once at the start of every conversation
# so the model knows how to behave.
SYSTEM_PROMPT = """
You are a helpful AI assistant built with the Strands SDK and Ollama.
You run entirely on the user's local machine - no internet required.
Be concise, friendly, and accurate in your answers.
"""


# --- AGENT CREATION ---
# Wire the model and system prompt together into an Agent.
# The Agent class handles the full reasoning loop automatically.
agent = Agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
)


# --- MAIN CHAT LOOP ---
# A simple interactive loop so you can keep chatting until
# you type 'exit' or 'quit'.
def main():
    print("=" * 50)
    print("  Simple AI Agent - Strands SDK + Ollama")
    print("  Model : llama3.2:3b  (running locally)")
    print("  Type 'exit' or 'quit' to stop.")
    print("=" * 50)

    while True:
        # Get input from the user
        user_input = input("\nYou: ").strip()

        # Allow a graceful exit
        if user_input.lower() in ("exit", "quit", ""):
            print("Goodbye!")
            break

        # Send the message to the agent and print the response.
        # agent(user_input) triggers the full agentic loop:
        # 1. Adds the user message to the conversation history
        # 2. Calls the model (Ollama / llama3.2:3b)
        # 3. Checks if the model wants to call any tools (none here)
        # 4. Returns the final text response
        response = agent(user_input)

        # Print the agent's reply
        print(f"\nAgent: {response}")


# --- ENTRY POINT ---
# Only run main() when this file is executed directly,
# not when it is imported as a module.
if __name__ == "__main__":
    main()
