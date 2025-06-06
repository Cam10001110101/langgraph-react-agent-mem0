"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

import os
from datetime import UTC, datetime
from typing import Dict, List, Literal, cast

import dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from mem0 import MemoryClient  # type: ignore[import-untyped]

from react_agent.configuration import Configuration
from react_agent.state import InputState, State
from react_agent.tools import TOOLS
from react_agent.utils import load_chat_model

dotenv.load_dotenv()

# Initialize Mem0 client
MEM0_API_KEY = os.getenv("MEM0_API_KEY")
mem0 = MemoryClient(api_key=MEM0_API_KEY)
# Use a persistent Mem0 user ID for all sessions, so memory is shared across runs.
MEM0_USER_ID = "123"

# Define the function that calls the model


async def call_model(state: State) -> Dict[str, List[AIMessage]]:
    """Call the LLM powering our "agent".

    This function prepares the prompt, initializes the model, and processes the response.

    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): Configuration for the model run.

    Returns:
        dict: A dictionary containing the model's response message.
    """
    configuration = Configuration.from_context()

    # Initialize the model with tool binding. Change the model or add more tools here.
    model = load_chat_model(configuration.model).bind_tools(TOOLS)

    # Format the system prompt. Customize this to change the agent's behavior.
    system_message = configuration.system_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )

    # Retrieve relevant memories from Mem0 and inject as context
    try:
        # Find the latest user message in this turn
        last_user_msg = None
        for m in reversed(state.messages):
            if isinstance(m, HumanMessage):
                last_user_msg = m
                break
        user_query = last_user_msg.content if last_user_msg else ""
        # Search Mem0 for relevant memories
        import asyncio
        memories = await asyncio.to_thread(
            mem0.search, user_query, user_id=MEM0_USER_ID, limit=5
        )
        print(f"[Mem0] Search query: {user_query}")  # noqa: T201
        print(f"[Mem0] Retrieved memories: {memories}")  # noqa: T201
        memory_context = "\n".join(f"- {m['memory']}" for m in memories) if memories else ""
        # Build context message
        if memory_context:
            context_message = {
                "role": "system",
                "content": f"Relevant memories for this user:\n{memory_context}"
            }
            messages = [context_message, {"role": "system", "content": system_message}, *state.messages]
        else:
            messages = [{"role": "system", "content": system_message}, *state.messages]
    except Exception as e:
        print(f"[Mem0] Error during memory search: {e}")  # noqa: T201
        messages = [{"role": "system", "content": system_message}, *state.messages]

    # Get the model's response
    response = cast(
        AIMessage,
        await model.ainvoke(
            messages
        ),
    )

    # Handle the case when it's the last step and the model still wants to use a tool
    if state.is_last_step and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ]
        }

    # Save both user and AI messages to Mem0
    to_save = []
    # Find the last HumanMessage in state.messages (user input for this turn)
    last_user_msg = None
    for m in reversed(state.messages):
        if isinstance(m, HumanMessage):
            last_user_msg = m
            break
    if last_user_msg:
        to_save.append({"role": "user", "content": last_user_msg.content})
    to_save.append({"role": "assistant", "content": response.content})
    import asyncio
    if to_save:
        try:
            print(f"[Mem0] Saving to memory: {to_save}")  # noqa: T201
            await asyncio.to_thread(
                mem0.add, to_save, user_id=MEM0_USER_ID, agent_id="react-agent"
            )
            print("[Mem0] Save successful.")  # noqa: T201
        except Exception as e:
            print(f"[Mem0] Error during memory save: {e}")  # noqa: T201

    # Return the model's response as a list to be added to existing messages
    return {"messages": [response]}


# Define a new graph

builder = StateGraph(State, input=InputState, config_schema=Configuration)

# Define the two nodes we will cycle between
builder.add_node(call_model)
builder.add_node("tools", ToolNode(TOOLS))

# Set the entrypoint as `call_model`
# This means that this node is the first one called
builder.add_edge("__start__", "call_model")


def route_model_output(state: State) -> Literal["__end__", "tools"]:
    """Determine the next node based on the model's output.

    This function checks if the model's last message contains tool calls.

    Args:
        state (State): The current state of the conversation.

    Returns:
        str: The name of the next node to call ("__end__" or "tools").
    """
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    # If there is no tool call, then we finish
    if not last_message.tool_calls:
        return "__end__"
    # Otherwise we execute the requested actions
    return "tools"


# Add a conditional edge to determine the next step after `call_model`
builder.add_conditional_edges(
    "call_model",
    # After call_model finishes running, the next node(s) are scheduled
    # based on the output from route_model_output
    route_model_output,
)

# Add a normal edge from `tools` to `call_model`
# This creates a cycle: after using tools, we always return to the model
builder.add_edge("tools", "call_model")

# Compile the builder into an executable graph
graph = builder.compile(name="ReAct Agent")
