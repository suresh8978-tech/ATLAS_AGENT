"""
ATLAS AGENT - AAP Job Diagnostic Analyzer


A LangGraph-based agent that analyzes Ansible Automation Platform job executions
and provides detailed diagnostic reports.

Usage:
    python agent.py
"""

from __future__ import annotations
import logging
import os
import argparse
import sys
import uuid
from typing import Annotated, Any, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langgraph.errors import GraphInterrupt

from tools.aap_events import fetch_job_events
from tools.config import Settings, load_settings

load_dotenv()
logger = logging.getLogger(__name__)

# ===========================================================================
# Global State
# ===========================================================================

runtime_settings: Settings | None = None
current_job_id: int | None = None

# ===========================================================================
# Logging Setup
# ===========================================================================

def setup_logging():
    """Configure logging to file only."""
    agent_dir = os.path.dirname(os.path.abspath(__file__))
    log_file_path = os.path.join(agent_dir, "atlas_agent.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler("atlas_agent.log"),
        ],
    )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler("atlas_agent.log"),
        ],
    )


# ==============================================================================
# Agent State Definition
# ==============================================================================

class AgentState(TypedDict):
    """State for the ATLAS diagnostic agent."""
    messages: Annotated[list, add_messages]
    job_id: Optional[int]


# =============================================================================
# System Prompt
# =============================================================================  

SYSTEM_PROMPT = (
        "You are an expert Ansible execution diagnostics analyst. "
        "When a job ID is provided, call get_job_events(job_id) to fetch the event data. "
        "Then produce a concise, practical diagnostic report with these sections:\n\n"
        "1. **Executive Summary** - Job overview, status, duration, key stats\n"
        "2. **Timeline Highlights** - Key events in chronological order\n"
        "3. **Failure Signals** - Any errors, warnings, or failed tasks\n"
        "4. **Probable Root Cause** - Analysis of what went wrong (or confirmation of success)\n"
        "5. **Recommended Next Checks** - Actionable next steps\n\n"
        "Be throught but concise. Use tables and formatting for clarity."
    )

# ===============================================================================
# Tools
# ===============================================================================

def build_tools():
    """Build tools with global settings baked in."""
    global runtime_settings

    @tool
    def get_job_events(job_id: int) -> str:
        """Fetch all Ansible Automation Platform job events for a job id."""
        logger.info(f"Fetching events for job {job_id}")
        events = fetch_job_events(job_id=job_id, settings=runtime_settings)
        logger.info(f"Retrieved {len(events)} events for job {job_id}")
        payload: dict[str, Any] = {
            "job_id": job_id,
            "event_count": len(events),
            "events": events,
        }
        return json.dumps(payload, ensure_ascii=True)

    return [get_job_events]

# ==========================================================================
# Node Functions
# ==========================================================================


def ask_job_id_node(state: AgentState) -> dict:
    """Use interrupt to ask the user for a job ID."""
    global runtime_settings, current_job_id

    if runtime_settings is None:
        runtime_settings = load_settings()

        logger.info("Asking user for job ID via interrupt")

        job_id = interrupt("\nEnter the AAP Job ID to analyze:")

        logger.info(f"User provided job ID: {job_id}")

        current_job_id = int(str(job_id).strip())

        return {
            "job_id": current_job_id,
            "messages": [HumanMessage(content=f"Analyze job {current_job_id}")],
        }
    
def agent_node(state: AgentState) -> dict:
        """Main agent node - calls LLM with tools."""
        global runtime_settings

        messages = state["messages"]

        # Add system prompt at the start
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

        
        model_name = runtime_settings.llm_name
        if "/" in model_name:
            model_name = model_name.split("/", maxsplit=1)[1]

        logger.info(f"Invoking LLM with model: {model_name}")

        tools = build_tools()
        llm = ChatAnthropic(
            model=model_name,
            temperature=0,
            max_tokens=7096,
        )
        agent =llm.bind_tools(tools)
        response = agent.invoke(messages)
        logger.info("LLM response received")

        return {"messages": [response]}


def tools_node(state: AgentState) -> dict:
    """Execute tool calls."""
    messages = state["messages"]
    last_message = messages[-1]

    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {}
    
    logger.info("-" * 60)
    logger.info("TOOL EXECUTION")
    for tc in last_message.tool_calls:
        logger.info(f"  Tool: {tc['name']}, Args: {tc.get('args', {})}")
    logger.info("-" * 60)

    tools = build_tools()
    tools_node = ToolNode(tools, handle_tool_errors=True)
    result = tools_node.invoke(state)

    result_messages = result.get("messages", [])
    for msg in result_messages:
        if isinstance(msg, ToolMessage):
            output = msg.content if isinstance(msg.content, str) else str(msg.content)
            logger.info(f" Tool output length: {len(output)} chars")

    return {"messages": result_messages}

# =================================================================================
# Routing
# =================================================================================

def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Determine if agent should call tools or finish."""
    messages = state["messages"]
    last_message = messages[-1] if messages else None

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    return "end"

# ================================================================================
# Graph Construction
# ================================================================================

def create_graph():
    """Build the ATLAS agent graph."""
    workflow = StateGraph(AgentState)

    workflow.add_node("ask_job_id", ask_job_id_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tools_node)

    workflow.set_entry_point("ask_job_id")

    workflow.add_edge("ask_job_id", "agent")

    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END},
    )

    workflow.add_edge("tools", "agent")

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)

# ================================================================
# Interactive Runner
# ================================================================

def run_interactive():
    """Run the agent in interactive mode with interrupt-based job ID input."""
    graph = create_graph()

    thread_id = str(uuid.uuid4)
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}

    initial_state: AgentState = {
        "messages": [],
        "job_id": None,
    }

    print()
    print("=" * 60)
    print(" ATLAS AGENT - AAP Job Diagnostic Analyzer")
    print("=" * 60)
    print("Commands:")
    print("    - Enter a job ID to analyze AAP job execution")
    print("    - 'help' to see available commands")
    print("    - 'quit' or 'exit' to exit")
    print()
    print("=" * 60)

    # First invocation triggers are interrupt
    try:
        graph.invoke(initial_state, config)
    except GraphInterrupt:
        pass 

    # Display the interrupt prompt
    state_snapshot = graph.get_state(config)
    if state_snapshot.next and hasattr(state_snapshot, "tasks") and state_snapshot.tasks:
        task = state_snapshot.tasks[0]
        if hasattr(task, "interrupts") and task.interrupts:
            print(task.interrupts[0].value)

    while True:
        print()
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            logger.info("User exited with keyboard interrupt")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            logger.info("User exited normally")
            break

        if user_input.lower() == "help":
            print("\n Enter any AAP job ID (numeric) to get a full")
            print("  diagnostic report including:")
            print("     - Executive Summary")
            print("     - Timeline Highlights")
            print("     - Failure Signals")
            print("     - Probable Root Cause")
            print("     - Recommended Next Checks")
            continue

        try:
            state_snapshot = graph.get_state(config)
            is_interrupted = bool(state_snapshot.next)

            if is_interrupted:
                logger.info(f"Resuming interrupt with job ID: {user_input}")
                print(f"\n Analyzing job {user_input}...\n")

                try:
                    result = graph.invoke(Command(resume=user_input), config)
                except GraphInterrupt:
                    state_snapshot = graph.get_state(config)
                    if state_snapshot.tasks and state_snapshot.tasks[0].interrupts:
                        print(state_snapshot.tasks[0].interrupts[0].value)
                    continue
            else:
                # New analysis - create new thread
                thread_id = str(uuid.uuid4)
                config = {
                    "configurable": {"thread_id": thread_id},
                    "recursion_limit": 50,
                }

                logger.info(f"Starting new analysis for job {user_input}")
                print(f"\n Analyzing job {user_input}...\n")

                # Trigger interrupt then resume with job ID
                try:
                    graph.invoke(initial_state, config)
                except GraphInterrupt:
                    pass

                try:
                    result = graph.invoke(Command(resume=user_input), config)
                except GraphInterrupt:
                    state_snapshot = graph.get_state(config)
                    if state_snapshot.tasks and state_snapshot.tasks[0].interrupts:
                        print(state_snapshot.tasks[0].interrupts[0].value)
                    continue
            # Print the last AI message
            state_snapshot = graph.get_state(config)
            if state_snapshot.values and state_snapshot.values.get("messages"):
                for msg in reversed(state_snapshot.values["messages"]):
                    if isinstance(msg, AIMessage) and msg.content:
                        logger.info(
                            f"Analysis complete for job "
                            f"{state_snapshot.values.get('job_id', 'unknown')}"
                        )
                        print(f"\n{msg.content}\n")
                        break

        except Exception as e:
            logger.error(f"Error during analysis: {e}", exc_info=True)
            print(f"\n Error: {e}\n")

# ======================================================================
# Main Entry Point
# ======================================================================

if __name__ == "__main__":
    setup_logging()
    logger.info("ATLAS AGENT started")
    run_interactive()
    logger.info("ATLAS AGENT stopped")
    
