"""
Quick HTTP test against the running inference API.
Samples 10 cat and 10 dog images from data/processed/test/ and reports accuracy.
"""
import sys
sys.path.insert(0, "/Users/shiva/Desktop/mlops-assignment-2/mlops/venv/lib/python3.9/site-packages")

import random
from pathlib import Path
import requests

MLOPS    = Path("/Users/shiva/Desktop/mlops-assignment-2/mlops")
TEST_DIR = MLOPS / "data/processed/test"
BASE_URL = "http://localhost:8000"
SAMPLE   = 10
SEED     = 42

random.seed(SEED)

correct = 0
total   = 0

for cls in ("cats", "dogs"):
    cls_dir = TEST_DIR / cls
    images  = sorted(cls_dir.glob("*.jpg"))
    sample  = random.sample(images, min(SAMPLE, len(images)))

    print(f"\n{'='*22} {cls.upper()} {'='*22}")
    for path in sample:
        try:
            with open(path, "rb") as f:
                r = requests.post(f"{BASE_URL}/predict",
                                  files={"file": (path.name, f, "image/jpeg")},
                                  timeout=10)
            d     = r.json()
            label = d["label"]
            conf  = d["confidence"] * 100
            ok    = "✅" if label == cls else "❌"
            if label == cls:
                correct += 1
            print(f"  {ok}  {path.name:20s}  →  {label:4s}  ({conf:.1f}%)")
        except Exception as e:
            print(f"  ⚠️   {path.name:20s}  →  (skipped: {e})")
        total += 1

print(f"\n{'='*50}")
print(f"  Result : {correct}/{total} correct  ({correct/total*100:.0f}%)")
print(f"{'='*50}")
