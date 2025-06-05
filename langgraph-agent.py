from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from mem0 import MemoryClient
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

import os

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Replace with your actual OpenAI API key or set in .env
MEM0_API_KEY = os.getenv("MEM0_API_KEY")  # Replace with your actual Mem0 API key or set in .env

# Initialize LangChain and Mem0
llm = ChatOpenAI(model="gpt-4.1", api_key=OPENAI_API_KEY)
mem0 = MemoryClient(api_key=MEM0_API_KEY)

# Define State and Graph
class State(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], add_messages]
    mem0_user_id: str

graph = StateGraph(State)

# Define the core logic for the Customer Support AI Agent
def chatbot(state: State) -> State:
    messages = state["messages"]
    mem0_user_id = state["mem0_user_id"]

    # Retrieve user history from Mem0
    try:
        mem0_history = mem0.get_all(version="v1", user_id=mem0_user_id)
        # Convert mem0 history to LangChain message objects
        lc_messages = []
        for msg in mem0_history:
            role = msg.get("role", "")
            content = msg.get("text", msg.get("content", ""))
            if role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
        # Add current session messages (if any new)
        lc_messages.extend(messages)
    except Exception as e:
        lc_messages = messages

    # Generate AI response
    ai_response = llm.invoke(lc_messages)
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
        except Exception as e:
            pass  # Optionally log error

    return {"messages": lc_messages, "mem0_user_id": mem0_user_id}

# Set Up Graph Structure
graph.add_node("chatbot", chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", "chatbot")

compiled_graph = graph.compile()

# Create Conversation Runner
def run_conversation(user_input: str):
    mem0_user_id = "123"
    config = {"configurable": {"thread_id": mem0_user_id}}
    state = {"messages": [HumanMessage(content=user_input)], "mem0_user_id": mem0_user_id}

    for event in compiled_graph.stream(state, config):
        for value in event.values():
            if value.get("messages"):
                print("Customer Support:", value["messages"][-1].content)
                return

if __name__ == "__main__":
    print("Welcome to Customer Support! How can I assist you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Customer Support: Thank you for contacting us. Have a great day!")
            break
        run_conversation(user_input)
