"""
Essay Writer - An AI-powered essay generation system using LangGraph.

This application creates an agent workflow that plans, researches, writes,
and refines essays through multiple revisions with AI assistance.
"""

import os, operator
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from dotenv import load_dotenv, find_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage

# Load environment variables
load_dotenv(find_dotenv())
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

class AgentState(TypedDict):
    """State definition for the essay writing agent.
    
    Attributes:
        task: The essay topic or assignment
        plan: The generated essay outline
        content: List of research information gathered
        draft: The current essay draft
        critique: Feedback on the current draft
        revision_number: Current revision iteration
        max_revisions: Maximum number of revisions to perform
    """
    task: str
    plan: str
    content: List[str]
    draft: str
    critique: str
    revision_number: int
    max_revisions: int

# Setup memory for checkpointing and the LLM
memory = MemorySaver()
llm = ChatOllama(model="llama3.2", base_url="http://localhost:11434", temperature=0)

# Initialize the search tool
google_serper = GoogleSerperAPIWrapper()

# System prompts for different agent roles
PLAN_PROMPT = """You are an expert writer tasked with writing a high level outline of an essay. \
Write such an outline for the user provided task "{task}". Give an outline of the essay along with any relevant notes \
or instructions for the sections."""

RESEARCH_PLAN_PROMPT = """
You are a researcher charged with providing information that can \
be used when writing the essay. Generate a list of search queries that \
will gather any relevant information. Only generate 3 queries max.\
Use the set of steps or outline as the guidelines to generate relevant queries.
"""

WRITER_PROMPT = """You are an essay assistant tasked with writing excellent 5-paragraph essays.\
Generate the best essay possible for the user's request and the initial outline. \
If the user provides critique, respond with a revised version of your previous attempts. \
Utilize all the information below as needed: 

------

{content}"""

REFLECTION_PROMPT = """You are a teacher grading an essay submission. \
Generate critique and recommendations for the user's submission. \
Provide detailed recommendations, including requests for length, depth, style, etc."""

RESEARCH_CRITIQUE_PROMPT = """You are a researcher charged with providing information that can \
be used when making any requested revisions (as outlined below). \
Generate a list of search queries that will gather any relevant information. Only generate 3 queries max."""

class Queries(BaseModel):
    """Model for structured query output.
    
    Attributes:
        queries: List of search queries
    """
    queries: List[str]

def plan_node(state: AgentState):
    """Creates an essay outline based on the given task.
    
    Args:
        state: Current agent state
        
    Returns:
        Dict containing the generated plan
    """
    messages = [
        SystemMessage(content=PLAN_PROMPT.format(task=state["task"])),
        HumanMessage(content=state["task"]),
    ]
   
    plan = llm.invoke(messages)
    return {"plan": plan.content}

def research_plan_node(state: AgentState):
    """Performs initial research based on the essay task.
    
    Args:
        state: Current agent state
        
    Returns:
        Dict containing the research content
    """
    messages = [
        SystemMessage(content=RESEARCH_PLAN_PROMPT),
        HumanMessage(content=state["task"]),
    ]

    queries = llm.with_structured_output(Queries).invoke(messages)
    content = state.get("content", [])
    for q in queries.queries:
        response = google_serper.run(query=q, max_results=2)
        content.append(response)
    return {"content": content}

def generation_node(state: AgentState):
    """Generates an essay draft based on the plan and research.
    
    Args:
        state: Current agent state
        
    Returns:
        Dict containing the essay draft and updated revision number
    """
    content = "\n\n".join(state.get("content", []))
    user_message = HumanMessage(
        content=f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}"
    )
    messages = [
        SystemMessage(content=WRITER_PROMPT.format(content=content)),
        user_message,
    ]

    draft = llm.invoke(messages)
    return {
        "draft": draft.content,
        "revision_number": state.get("revision_number", 1) + 1,
    }

def reflection_node(state: AgentState):
    """Generates critique for the current essay draft.
    
    Args:
        state: Current agent state
        
    Returns:
        Dict containing the critique
    """
    messages = [
        SystemMessage(content=REFLECTION_PROMPT),
        HumanMessage(content=state["draft"]),
    ]

    critique = llm.invoke(messages)
    return {"critique": critique.content}

def research_critique_node(state: AgentState):
    """Performs additional research based on critique.
    
    Args:
        state: Current agent state
        
    Returns:
        Dict containing updated research content
    """
    messages = [
        SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
        HumanMessage(content=state["critique"]),
    ]

    queries = llm.with_structured_output(Queries).invoke(messages)
    content = state.get("content", [])
    for q in queries.queries:
        response = google_serper.run(query=q, max_results=2)
        content.append(response)
    return {"content": content}

def should_continue_node(state):
    """Determines if the revision process should continue or end.
    
    Args:
        state: Current agent state
        
    Returns:
        String indicating next node or END
    """
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "reflect"

# Build the LangGraph workflow
writer = StateGraph(AgentState)

# Add all workflow nodes
writer.add_node("planner", plan_node)
writer.add_node("research_plan", research_plan_node)
writer.add_node("generate", generation_node)
writer.add_node("reflect", reflection_node)
writer.add_node("research_critique", research_critique_node)

# Set the entry point
writer.set_entry_point("planner")

# Add conditional edge for determining when to end revisions
writer.add_conditional_edges(
    "generate",
    should_continue_node,
    {END: END, "reflect": "reflect"},
)

# Connect the workflow nodes
writer.add_edge("planner", "research_plan")
writer.add_edge("research_plan", "generate")
writer.add_edge("reflect", "research_critique")
writer.add_edge("research_critique", "generate")

# Compile the graph with checkpointing
graph = writer.compile(checkpointer=memory)
thread = {"configurable": {"thread_id": "1"}}

# Example execution
if __name__ == "__main__":
    for s in graph.stream({
            'task': "Information on computer vision and NLP",
            "max_revisions": 2,
            "revision_number": 1,
        }, thread):
        print(s)
