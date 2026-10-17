"""Compatibility entry point; implementation lives in src/tennis_control."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from tennis_control.maddpg_agent import *
