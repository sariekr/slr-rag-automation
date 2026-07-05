"""Ensure the package root is importable so `src` / `config` resolve under pytest."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
