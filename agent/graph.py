"""LangGraph agent wiring for the coffee ops copilot."""

from __future__ import annotations

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from agent.prompts import SYSTEM_PROMPT
from agent.tools import ALL_TOOLS


def build_agent(model_name: str = "gpt-4o", temperature: float = 0):
    """Construct and return a compiled LangGraph ReAct agent.

    Requires OPENAI_API_KEY in the environment (or .env file).
    LangSmith tracing activates automatically when LANGSMITH_TRACING=true.
    """
    load_dotenv()
    model = ChatOpenAI(model=model_name, temperature=temperature)
    return create_react_agent(model, tools=ALL_TOOLS, prompt=SYSTEM_PROMPT)


# Module-level graph instance for `langgraph dev` CLI.
# The dev server loads .env before importing, so OPENAI_API_KEY is available.
graph = build_agent()
