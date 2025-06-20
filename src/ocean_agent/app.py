#[Written by Victor Clausen (s232604) and Asger Lanstorp (s235217)]

# src/ocean_agent/app.py
"""
A FastAPI web server that provides a RESTful API for interacting with the
OceanWave3D AI agent. This version correctly handles conversation history
by only persisting user and AI messages, in line with API constraints.
"""

# this file is the frontend application. The pipeline between index.html and agent.py

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import AIMessage, HumanMessage
from .agent import graph
from .path_helper import get_app_path, get_persistent_path

#setup for allowed localhost connections
app = FastAPI(title="Ocean AI Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path and Static File Definitions. if any folder or file names are changed, then these needs to be changed too
THIS_DIR = get_app_path("src/ocean_agent")     
DATA_DIR = get_persistent_path("data")         

app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")


class ChatRequest(BaseModel):
    messages: list[dict]

#HTML setup
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serves the main frontend application file (index.html)."""
    html_path = THIS_DIR / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>Error: index.html not found</h1>", status_code=500)
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


#reciving/outputting chat information
@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    """
    Processes a chat request by invoking the agent graph. This version correctly
    handles history by only persisting Human and AI messages between turns.
    """

    langchain_messages = []
    for msg in req.messages:
        if msg.get("role") == "user":
            langchain_messages.append(HumanMessage(content=msg.get("content")))
        elif msg.get("role") == "ai":
            langchain_messages.append(AIMessage(content=msg.get("content")))

    result = graph.invoke({"messages": langchain_messages})
 
    response_messages = []
    for msg in result.get("messages", []):
        if isinstance(msg, HumanMessage):
            response_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage) and msg.content:
            response_messages.append({"role": "ai", "content": msg.content})

    # Check the final AI response for a signal that a GIF was created.
    gif_path = None
    latest_ai_message = next(
        (m for m in reversed(response_messages) if m["role"] == "ai"), None
    )
    if latest_ai_message:
        reply_text = latest_ai_message["content"]
        if "✅ GIF saved" in reply_text:
            try:
                path_str = reply_text.split("→")[-1].strip()
                gif_name = get_app_path(path_str).name
                gif_path = f"/data/{gif_name}"
            except Exception:
                gif_path = None
    return JSONResponse({"messages": response_messages, "gif": gif_path})

#Giving the index.html the .gif file from the data folder
@app.get("/gifs")
async def list_gifs():
    """Provides a list of all available GIF files in the data directory."""
    if not DATA_DIR.exists():
        return JSONResponse(content=[], status_code=404)
    names = [p.name for p in sorted(DATA_DIR.glob("*.gif"), key=lambda p: p.stem)]
    return JSONResponse(names)