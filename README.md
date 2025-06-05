# LangGraph React Agent + mem0

This project is an implementation of a [LangGraph React Agent](https://github.com/langchain-ai/react-agent/tree/main) combined with a [mem0](https://mem0.ai/) integration.

## Overview

- **LangGraph React Agent**: Provides a framework for building agents that use a graph-based approach to reasoning and action, leveraging the React Agent architecture from LangChain.
- **mem0**: Integrates with [mem0](https://mem0.ai/), a memory and knowledge management system, to enhance agent capabilities with persistent memory and retrieval.

## Features

- Graph-based agent reasoning and execution
- Integration with mem0 for memory-augmented workflows
- Extensible and modular design

## Installation

```bash
git clone https://github.com/yourusername/langgraph-react-agent-mem0.git
cd langgraph-react-agent-mem0
uv venv
source ./venv/bin/activate
uv pip install -e .
uv pip install langgraph
uv pip install "langgraph-cli[inmem]"
```

## Usage

```bash
# Ensure your environment is active, then run:
langgraph dev
```
