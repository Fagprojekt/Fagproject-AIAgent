#[Written by Asger Lanstorp (s235217), Emil Skovgård (s235282), Jens Kalmberg(s235277) and Victor Clausen (s232604)]

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain.tools import StructuredTool
from langgraph.graph import StateGraph, END
from langgraph.graph.graph import START
from pydantic.v1 import BaseModel, Field
from typing import Optional, Literal, Dict
from .visualizer import visualize
from .guardrails import is_reply_off_domain, is_user_reply_off_domain
from .param_editor import modify_params
from .path_helper import get_app_path, get_persistent_path, get_project_root_path


# --- Project-level Directory Structure
EXAMPLES_DIR = get_app_path("examples")
OUTPUT_DIR = get_persistent_path("output") # For writable simulation output
DATA_DIR = get_persistent_path("data")     # For writable GIFs
RUN_SCRIPT = get_app_path("docker/run_oceanwave3d.sh") # Read-only bundled file

# Load .env from next to the EXE for the user, or from project root for dev
load_dotenv(get_project_root_path() / ".env")

def _template_for(case: str) -> Path | None:
    """Finds the input template file for a given case name."""
    lname = case.lower()
    for f in EXAMPLES_DIR.glob("OceanWave3D.inp.*"):
        if f.suffixes[-1].lower().lstrip(".") == lname:
            return f
    return None

# ----------------------------- ALL TOOLS -------------------------------------------

# Definition of the tool that runs the actual oceanWave3d code
# This tool only needs a valid casename to succesfully run (other than all pre-reqs of course)

class _RunArgs(BaseModel):
    case_name: str
def _run_oceanwave3d(case_name: str) -> str:
    """Runs the OceanWave3D simulation for a specific case name."""
    tpl = _template_for(case_name)
    if not tpl: return f"❌ No template 'OceanWave3D.inp.{case_name}' in {EXAMPLES_DIR}"
    if not RUN_SCRIPT.exists(): return f"❌ run script {RUN_SCRIPT} not found."
    run_dir = OUTPUT_DIR / case_name
    run_dir.mkdir(parents=True, exist_ok=True)
    # Check if the input file is a modified one, otherwise copy from template
    modified_inp = EXAMPLES_DIR / f"OceanWave3D.inp.{case_name}_mod"
    if modified_inp.exists():
        shutil.copy(modified_inp, run_dir / "OceanWave3D.inp")
    else:
        shutil.copy(tpl, run_dir / "OceanWave3D.inp")

    cmd = [
    "docker", "run", "--rm",
    "-v", f"{run_dir}:/ocw3d",
    "-w", "/ocw3d",
    "docker_oceanwave3d",
    "OceanWave3D.inp"
    ]
    print(f"🛠 Running Docker command: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

    if proc.returncode != 0: return f"❌ OceanWave3D failed:\n" + (proc.stderr or proc.stdout)[:400]
    fort_files = list(run_dir.glob("fort.*"))
    if not fort_files: return f"❌ Script exited with 0 but produced no fort.* files. Check logs:\n" + (proc.stdout or proc.stderr)[:400]
    return f"✅ OceanWave3D finished, {len(fort_files)} fort.* files in {run_dir}\n{proc.stdout[:400]}"

# Definition of tool that generates and sales the .gif file
class _VizArgs(BaseModel):
    case_name: str
def _viz(case_name: str) -> str:
    """Generates and saves an animated GIF from completed simulation data."""
    found_dir = None
    if OUTPUT_DIR.exists():
        for d in OUTPUT_DIR.iterdir():
            if d.is_dir() and d.name.lower() == case_name.lower():
                found_dir = d
                break
    if not found_dir: return f"❌ Directory for case '{case_name}' not found in {OUTPUT_DIR}. Run the simulation first."
    src_dir = found_dir
    if not list(src_dir.glob("fort.*")): return f"❌ No fort.* files in {src_dir}. Run the simulation first."
    gif = DATA_DIR / f"{src_dir.name}.gif"
    gif.parent.mkdir(exist_ok=True)
    try:
        visualize(src_dir, gif, fps=5)
        return f"✅ GIF saved → {gif}"
    except Exception as exc:
        return f"❌ Visualization failed: {exc}"

# Definition of tool that returns the overvies of all available tools for the agent
def _get_agent_capabilities() -> str:
    """Returns a detailed, Markdown-formatted summary of the agent's available tools."""
    markdown_summary = "I have the following capabilities:\n\n"
    for tool in TOOLS:
        args_list = [f"`{key}`" for key in tool.args.keys()]
        args_str = f"**Arguments:** {', '.join(args_list)}" if args_list else "**Arguments:** None"
        markdown_summary += f"### `{tool.name}`\n**Description:** {tool.description}\n{args_str}\n---\n"
    return markdown_summary

# definition of the tool that lists available files
class _ListFilesArgs(BaseModel):
    file_type: Literal["cases", "visualizations"] = Field(..., description="Must be 'cases' or 'visualizations'.")
def _list_available_files(file_type: Literal["cases", "visualizations"]) -> str:
    """Lists available files for running simulations or creating visualizations."""
    if file_type == "cases":
        if not EXAMPLES_DIR.exists(): return "❌ The `examples` directory was not found."
        case_files = list(EXAMPLES_DIR.glob("OceanWave3D.inp.*"))
        if not case_files: return "No available cases found."
        case_names = [f"- `{f.name.split('.')[-1]}`" for f in case_files]
        return "Available cases to run:\n" + "\n".join(sorted(case_names))
    if file_type == "visualizations":
        if not OUTPUT_DIR.exists(): return "❌ The `output` directory was not found."
        ready_for_viz = [d.name for d in OUTPUT_DIR.iterdir() if d.is_dir() and any(d.glob("fort.*"))]
        if not ready_for_viz: return "No completed simulations found."
        viz_names = [f"- `{name}`" for name in sorted(ready_for_viz)]
        return "Available results to visualize:\n" + "\n".join(viz_names)
    return "❌ Invalid file_type."


# Definition of the tool that modifies the initial paramaters of the cases
class _ModParamArgs(BaseModel):
    case_name: str = Field(..., description="Suffix of OceanWave3D.inp.* to modify.")
    nsteps: Optional[int] = Field(
        None, description="New value for Nsteps parameter (number of time steps)."
    )
    gravity: Optional[float] = Field(
        None, description="New value for gravitational acceleration constant."
    )

def _modify_params(
    case_name: str,
    nsteps: Optional[int] = None,
    gravity: Optional[float] = None
) -> str:
    """Apply Nsteps and/or gravity changes to a case input, preserving any existing _mod file."""
    # 1) Find the original template
    base_tpl = _template_for(case_name)
    if not base_tpl:
        return f"❌ No template 'OceanWave3D.inp.{case_name}' in {EXAMPLES_DIR}"

    #2) if there already exists a _mod, override that, so that users dont end up with multiple "_mod_mod..." files
    # in case of multible changes
    mod_path = EXAMPLES_DIR / f"OceanWave3D.inp.{case_name}_mod"
    tpl = mod_path if mod_path.exists() else base_tpl

    # 3) Change/build the dict based on the requested changes
    changes: Dict[str, float] = {}
    if nsteps is not None:
        changes["nsteps"] = nsteps
    if gravity is not None:
        changes["gravity"] = gravity
    if not changes:
        return "❌ You must specify at least one of `nsteps` or `gravity` to modify."

    try:
        # 4) If tpl is already the mod file, overwrite it (suffix=""), otherwise create a new _mod
        suffix = "" if tpl is mod_path else "_mod"
        success_message = modify_params(tpl, changes, suffix=suffix)

        return (
            f"✅ Parameters for case '{case_name}' were modified.\n\n"
            f"{success_message}\n\n"
            f"Now run `run_oceanwave3d` on `{case_name}_mod`."
        )
    except Exception as exc:
        return f"❌ Parameter modification failed: {exc}"



# Registration of each normal tool
RUN_TOOL = StructuredTool.from_function(func=_run_oceanwave3d, name="run_oceanwave3d", args_schema=_RunArgs)
VIS_TOOL = StructuredTool.from_function(func=_viz, name="visualize_oceanwave_data", args_schema=_VizArgs)
CAPABILITIES_TOOL = StructuredTool.from_function(func=_get_agent_capabilities, name="get_agent_capabilities")
LIST_FILES_TOOL = StructuredTool.from_function(func=_list_available_files, name="list_available_files", args_schema=_ListFilesArgs)
# Registration of the moderate tool (used to moderate the initial conditions of the cases)
MODPARAM_TOOL = StructuredTool.from_function(
    func=_modify_params,
    name="modify_case_parameters",
    args_schema=_ModParamArgs,
    description="Duplicate an OceanWave3D .inp, set Nsteps and/or gravity, and save as `_mod`."
)

# List of all our implemented tools
TOOLS = [RUN_TOOL, VIS_TOOL, CAPABILITIES_TOOL, LIST_FILES_TOOL, MODPARAM_TOOL]
TOOL_MAP = {tool.name: tool for tool in TOOLS}


# Definition of the agent. Here each line [1.,2.,...] are the agents direct
# definitions/guidelines for what it should/should not do

def assistant(state: Dict) -> Dict:
    """The 'main brain' of the agent. It decides whether to use a tool,
    respond conversationally, or state that it cannot help."""
    system_prompt = (
        "You are a helpful and efficient scientific assistant for the OceanWave3D simulation tool.\n\n"
        "### YOUR BEHAVIOR\n"
        "1.  **First, analyze the user's request.**\n"
        "2.  **If the request is a simple greeting like 'hi' or 'hello', respond with a friendly welcome and ask what the user wants to do, immediately followed by listing your available tools.\n"
        "3.  **If the request is a task related to your tools (running simulations, visualizing data, listing files, explaining tools, modifying parameters),** you MUST ALWAYS use the appropriate tool. NEVER try to answer from memory.\n"
        "4.  **Before running, visualizing, or modifying anything, ALWAYS ask for clarification from the user if it is the correct case.\n"
        "5.  **After a tool runs,** ALWAYS present its output clearly in Markdown. Add a brief, polite introductory sentence.\n"
        "6.  **If the user asks to modify case parameters,** you MUST call the `modify_case_parameters` tool with arguments `template_path` (the case's .inp path) and `nsteps` (the new integer value). NEVER try to answer from memory.\n"
        "7.  **If the user says 'thank you',** respond politely and suggest a few helpful next commands.\n"
    )
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"))
    reply = llm.bind_tools(TOOLS).invoke(messages)
    return {"messages": state["messages"] + [reply]}

def tools_runner(state: Dict) -> Dict:
    """Executes tool calls requested by the AI and appends the results."""
    messages = state["messages"]
    last_message = messages[-1]
    tool_messages = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        if tool_name in TOOL_MAP:
            tool_to_call = TOOL_MAP[tool_name]
            tool_output = tool_to_call.invoke(tool_call["args"])
            tool_messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"]))
    return {"messages": messages + tool_messages}


# Router is used when the agent needs to call a tool
def router(state: Dict):
    """Routes the flow. If the AI decided to call a tool, go to the tools_runner.
    Otherwise, check the AI's text response for stalling."""
    if isinstance(state["messages"][-1], AIMessage) and getattr(state["messages"][-1], "tool_calls", None):
        return "tools"
    return "check_for_stalling"


# Input checker guardrail
# See "is_user_reply_off_domain" definition in guardrails.py
def input_checker_node(state: Dict) -> Dict:
    """
    Checks whether the *user’s* prompt is off-domain.
    If so, we short-circuit immediately to rejection.
    """
    # the very first message is the human’s prompt
    user_prompt = state["messages"][0].content

    if is_user_reply_off_domain(user_prompt):
        # signal we want to reject
        return {
            "input_decision": "rejection",
            "messages": state["messages"],
            }
    return {"input_decision": "proceed"}

# Output checker guardrail
# See "is_reply_off_domain" definition in guardrails.py
def output_checker_node(state: Dict) -> Dict:
    """Checks whether the assistant's reply is OFF-DOMAIN."""
    last_reply = state["messages"][-1].content
    if is_reply_off_domain(last_reply):
        return {"output_decision": "rejection"}
    return {"output_decision": "end"}


# Rejection node
def rejection_node(state: Dict) -> Dict:
    """Returns the standardized rejection message."""
    # We use a fixed message to ensure consistency.
    rejection_message = "I am a specialized assistant for the OceanWave3D simulator. I can only help with topics related to running simulations and visualizing data."
    new_history = state["messages"][:-1] + [AIMessage(content=rejection_message)]
    return {"messages": new_history}

# Graph definition. See section [Design and Implementation] in the report for printed out version.

class AgentState(dict):
    """Defines the state passed between nodes in the graph."""
    messages: list
    input_decision: Literal["rejection","proceed"]
    stalling_decision: Literal["rejection", "end"]

builder = StateGraph(AgentState)
builder.add_node("input_checker", input_checker_node)
builder.add_node("assistant", assistant)
builder.add_node("tools", tools_runner)
builder.add_node("output_checker", output_checker_node)
builder.add_node("rejection", rejection_node)

builder.add_edge(START, "input_checker")

builder.add_conditional_edges(
    "input_checker",
    lambda state: state["input_decision"],
    {
        "rejection": "rejection",
        "proceed":   "assistant",
    },
)

builder.add_conditional_edges("assistant", router, {
    "tools":              "tools",
    "check_output": "output_checker",
})
builder.add_conditional_edges("output_checker", lambda x: x["output_decision"], {
    "end":       END,
    "rejection": "rejection",
})
builder.add_edge("tools", "assistant")
builder.add_edge("rejection", END)

graph = builder.compile()


# -------------------------------- CLI LOOP -------------------------------
if __name__ == "__main__":
    print("Ocean-Agent ready (Ctrl-C to exit).")
    try:
        while True:
            ask = input("\nYou: ")
            state = graph.invoke({"messages": [HumanMessage(content=ask)]})
            print(f"AI: {state['messages'][-1].content}")
    except KeyboardInterrupt:
        print("\nBye!")