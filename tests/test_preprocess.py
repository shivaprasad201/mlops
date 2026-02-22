import os
from pathlib import Path
from PIL import Image
import numpy as np

from src.utils.preprocess import load_image_as_numpy_rgb_224, preprocess_image_to_rgb_224

def test_preprocess_image_to_rgb_224(tmp_path: Path):
    # create a dummy image
    img = Image.fromarray(np.zeros((300, 500, 3), dtype=np.uint8))
    p = tmp_path / "x.jpg"
    img.save(p)

    out = preprocess_image_to_rgb_224(str(p))
    assert out.size == (224, 224)
    assert out.mode == "RGB"

def test_load_image_as_numpy_rgb_224(tmp_path: Path):
    img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    p = tmp_path / "y.png"
    img.save(p)

    arr = load_image_as_numpy_rgb_224(str(p))
    assert arr.shape == (224, 224, 3)
    assert arr.dtype == np.uint8
