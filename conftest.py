"""
conftest.py — adds the mlops/ directory to sys.path so pytest can resolve
`src.*` imports regardless of where pytest is invoked from.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
