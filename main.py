import os, operator
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from dotenv import load_dotenv, find_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage

load_dotenv(find_dotenv())
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

class AgentState(TypedDict):
    task: str
    plan: str
    content: List[str]
    draft: str
    critique: str
    revision_number: int
    max_revisions: int

memory = MemorySaver()
llm = ChatOllama(model="llama3.2", base_url="http://localhost:11434", temperature=0)

google_serper = GoogleSerperAPIWrapper()

PLAN_PROMPT = """You are an expert writer tasked with writing a high level outline of an essay. \
Write such an outline for the user provided task "{task}". Give an outline of the essay along with any relevant notes \
or instructions for the sections."""
# """
# You are an expert writer responsible for planning out the concrete steps \
# that needs to carry out in order to write a \
# self-sufficient and complete essay on a topic "{task}". \
# These individual steps need to be goal-oriented so that they aid in achieving the final \
# task smoothly. Give an outline of the essay along with any relevant notes \
# or instructions for the sections.
# """

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
    queries: List[str]

def plan_node(state: AgentState):
    messages = [
        SystemMessage(content=PLAN_PROMPT.format(task=state["task"])),
        HumanMessage(content=state["task"]),
    ]
   
    plan = llm.invoke(messages)
    return {"plan": plan.content}

def research_plan_node(state: AgentState):
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
    messages = [
        SystemMessage(content=REFLECTION_PROMPT),
        HumanMessage(content=state["draft"]),
    ]

    critique = llm.invoke(messages)
    return {"critique": critique.content}

def research_critique_node(state: AgentState):
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
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "reflect"

writer = StateGraph(AgentState)

writer.add_node("planner", plan_node)
writer.add_node("research_plan", research_plan_node)
writer.add_node("generate", generation_node)
writer.add_node("reflect", reflection_node)
writer.add_node("research_critique", research_critique_node)

writer.set_entry_point("planner")

writer.add_conditional_edges(
    "generate",
    should_continue_node,
    {END: END, "reflect": "reflect"},
)

writer.add_edge("planner", "research_plan")
writer.add_edge("research_plan", "generate")
writer.add_edge("reflect", "research_critique")
writer.add_edge("research_critique", "generate")

graph = writer.compile(checkpointer=memory)
thread = {"configurable": {"thread_id": "1"}}

for s in graph.stream({
        'task': "Information on computer vision and NLP",
        "max_revisions": 2,
        "revision_number": 1,
    }, thread):
    print(s)
