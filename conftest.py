"""Pytest configuration — ensures the engine package is importable from tests/.

This file makes the parent directory importable so `from engine import ...`
works when pytest is run from the zip's root on any OS.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
