#[Written by Jens Kalmberg(s235277)]

# src/ocean_agent/path_helper.py
import sys
from pathlib import Path

# This file is simply to get the absolute path. The absolute directory is important for a handful of files throughout the program
# THis file could have possible been "outsourced" to agent.py, but to create simplicity and disassociation, here it is.

def get_app_path(relative_path: str) -> Path:
    try:
        base_path = Path(sys._MEIPASS)
    except Exception:
        base_path = Path(__file__).resolve().parent.parent.parent
    return base_path / relative_path

def get_persistent_path(relative_path: str) -> Path:
    if getattr(sys, 'frozen', False):
        base_path = Path(sys.executable).parent
    else:
        base_path = Path(__file__).resolve().parent.parent.parent
    persistent_dir = base_path / relative_path
    persistent_dir.mkdir(parents=True, exist_ok=True)
    return persistent_dir

def get_project_root_path() -> Path:
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).parent.parent
    else:
        return Path(__file__).resolve().parent.parent.parent