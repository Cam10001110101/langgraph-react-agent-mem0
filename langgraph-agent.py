"""Simple CLI for running a memory-backed customer support agent."""

import os
from typing import Annotated, List, TypedDict, cast

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from mem0 import MemoryClient  # type: ignore[import-untyped]
from pydantic import SecretStr

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Replace with your actual OpenAI API key or set in .env
MEM0_API_KEY = os.getenv("MEM0_API_KEY")  # Replace with your actual Mem0 API key or set in .env

# Initialize LangChain and Mem0
llm = ChatOpenAI(
    model="gpt-4.1",
    api_key=SecretStr(OPENAI_API_KEY) if OPENAI_API_KEY else None,
)  # type: ignore[arg-type]
mem0 = MemoryClient(api_key=MEM0_API_KEY)  # type: ignore[import-untyped]

# Define State and Graph
class State(TypedDict):
    """Conversation state for the CLI agent."""

    messages: Annotated[List[HumanMessage | AIMessage], add_messages]
    mem0_user_id: str

graph = StateGraph(State)

# Define the core logic for the Customer Support AI Agent
def chatbot(state: State) -> State:
    """Handle a single conversation turn."""
    messages = state["messages"]
    mem0_user_id = state["mem0_user_id"]

    # Retrieve user history from Mem0
    try:
        mem0_history = mem0.get_all(version="v1", user_id=mem0_user_id)
        # Convert mem0 history to LangChain message objects
        lc_messages: List[HumanMessage | AIMessage] = []
        for msg in mem0_history:
            role = msg.get("role", "")
            content = msg.get("text", msg.get("content", ""))
            if role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
        # Add current session messages (if any new)
        lc_messages.extend(messages)
    except Exception:
        lc_messages = messages

    # Generate AI response
    ai_response = cast(AIMessage, llm.invoke(lc_messages))
    lc_messages.append(ai_response)

    # Save both user and AI messages to Mem0
    to_save = []
    # Only save the last two messages (user + AI) for this turn
    if len(lc_messages) >= 2:
        for m in lc_messages[-2:]:
            if isinstance(m, HumanMessage):
                to_save.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                to_save.append({"role": "assistant", "content": m.content})
    if to_save:
        try:
            mem0.add(to_save, user_id=mem0_user_id, agent_id="langgraph-agent")
        except Exception:
            pass  # Optionally log error

    return {"messages": lc_messages, "mem0_user_id": mem0_user_id}

# Set Up Graph Structure
graph.add_node("chatbot", chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", "chatbot")

compiled_graph = graph.compile()

# Create Conversation Runner
def run_conversation(user_input: str) -> None:
    """Run a conversation loop for a single user input."""
    mem0_user_id = "123"
    config = {"configurable": {"thread_id": mem0_user_id}}
    state = {"messages": [HumanMessage(content=user_input)], "mem0_user_id": mem0_user_id}

    for event in compiled_graph.stream(state, cast(RunnableConfig, config)):
        for value in event.values():
            if value.get("messages"):
                print("Customer Support:", value["messages"][-1].content)  # noqa: T201
                return

if __name__ == "__main__":
    print("Welcome to Customer Support! How can I assist you today?")  # noqa: T201
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Customer Support: Thank you for contacting us. Have a great day!")  # noqa: T201
            break
        run_conversation(user_input)
