from __future__ import annotations

import os
from typing import Dict

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI
from langchain.tools import StructuredTool
from pydantic.v1 import BaseModel
from langgraph.graph import StateGraph
from langgraph.graph.graph import START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import ToolMessage

from ocean_agent.visualizer import visualize

load_dotenv()

# ── Structured tool ──────────────────────────────────────────────
class _VizArgs(BaseModel):
    data_directory: str
    output_gif_path: str


def _viz(data_directory: str, output_gif_path: str) -> str:
    try:
        gif = visualize(data_directory, output_gif_path, fps=5)
        return f"✅ Animation saved to {gif}"
    except Exception as exc:
        return f"❌ Visualization failed: {exc}"


VIS_TOOL = StructuredTool.from_function(
    func=_viz,
    name="visualize_oceanwave_data",
    description="Generate a GIF animation from OceanWave3D fort.* files.",
    args_schema=_VizArgs,
)

OPENAI_TOOLS = [
    {
        "name": VIS_TOOL.name,
        "description": VIS_TOOL.description,
        "parameters": VIS_TOOL.args_schema.schema(),
    }
]

# ── Assistant node ───────────────────────────────────────────────
SYS_PROMPT = SystemMessage(
    content="You are a helpful assistant able to visualise OceanWave3D data via the available tool."
)
chat_llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o")).bind_tools(OPENAI_TOOLS)


def assistant(state: Dict) -> Dict:
    reply = chat_llm.invoke([SYS_PROMPT, *state["messages"]])
    return {"messages": state["messages"] + [reply]}


# ── Tool executor node ───────────────────────────────────────────
class _ToolsRunner:
    """Runs tools and appends proper role='tool' messages."""

    def __init__(self):
        self.node = ToolNode([VIS_TOOL])

    def __call__(self, state: Dict) -> Dict:
        last_ai: AIMessage = state["messages"][-1]            # assistant with tool_calls
        # Execute the tools; returns a list[str] aligned with tool_calls
        outputs = self.node.invoke({"messages": state["messages"]})
        tool_msgs = [
            ToolMessage(content=out, tool_call_id=call["id"])
            for call, out in zip(last_ai.tool_calls, outputs)
        ]
        return {"messages": state["messages"] + tool_msgs}



tools_runner = _ToolsRunner()

# ── Router ───────────────────────────────────────────────────────
def router(state: Dict):
    last = state["messages"][-1] if state["messages"] else None
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return END


# ── Build graph ──────────────────────────────────────────────────
builder = StateGraph(dict)

builder.add_node("assistant", assistant)
builder.add_node("tools", tools_runner)
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", router)
builder.add_edge("tools", "assistant")

graph = builder.compile()

# ── CLI loop ─────────────────────────────────────────────────────
if __name__ == "__main__":
    state: Dict = {"messages": []}

    print("Ocean-Agent ready (Ctrl-C to exit).")
    try:
        while True:
            user = input("\nYou: ")
            state["messages"].append(HumanMessage(content=user))
            state = graph.invoke(state)
            print(f"Agent: {state['messages'][-1].content}")
    except KeyboardInterrupt:
        print("\nBye!")
