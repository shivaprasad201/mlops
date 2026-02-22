"""
Generate model.pt with randomly initialized SimpleCNN weights.
Use this when you don't have a trained model yet — the API will start
and return predictions (random quality) so you can verify the full stack.
Replace with trained weights after running: python -m src.train
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from src.model import SimpleCNN
from pathlib import Path

dest = Path("models/model.pt")
dest.parent.mkdir(exist_ok=True)

model = SimpleCNN(num_classes=2)
torch.save(model.state_dict(), dest)
print(f"Saved random-initialized weights to {dest}  ({dest.stat().st_size / 1024:.1f} KB)")
