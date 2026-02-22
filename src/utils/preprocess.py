import os
import random
import shutil
from pathlib import Path
from typing import Tuple, List

from PIL import Image
import numpy as np

IMG_SIZE = (224, 224)

def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def preprocess_image_to_rgb_224(img_path: str) -> Image.Image:
    """Load an image, convert to RGB and resize to 224x224."""
    img = Image.open(img_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    return img

def split_dataset(
    raw_dir: str,
    processed_dir: str,
    splits: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42
) -> None:
    """Create train/val/test folders in ImageFolder format.

    Expected raw_dir structure:
      raw_dir/
        cats/   (or Cat/)
        dogs/   (or Dog/)

    Output:
      processed_dir/
        train/cats, train/dogs
        val/cats,   val/dogs
        test/cats,  test/dogs
    """
    raw = Path(raw_dir)
    out = Path(processed_dir)
    if out.exists():
        shutil.rmtree(out)
    (out/"train").mkdir(parents=True, exist_ok=True)
    (out/"val").mkdir(parents=True, exist_ok=True)
    (out/"test").mkdir(parents=True, exist_ok=True)

    # class folders (normalize names)
    class_candidates = [p for p in raw.iterdir() if p.is_dir()]
    if not class_candidates:
        raise FileNotFoundError(f"No class folders found inside {raw_dir}.")
    classes = []
    for p in class_candidates:
        name = p.name.lower()
        if "cat" in name:
            classes.append(("cats", p))
        elif "dog" in name:
            classes.append(("dogs", p))
    if len(classes) < 2:
        raise ValueError("Could not find both cat and dog folders. Ensure raw_dir has cats/ and dogs/.")

    random.seed(seed)

    for class_name, class_path in classes:
        imgs = [p for p in class_path.rglob("*") if p.is_file() and is_image_file(p)]
        if not imgs:
            raise FileNotFoundError(f"No images found in {class_path}.")

        random.shuffle(imgs)
        n = len(imgs)
        n_train = int(splits[0] * n)
        n_val = int(splits[1] * n)
        train_imgs = imgs[:n_train]
        val_imgs = imgs[n_train:n_train+n_val]
        test_imgs = imgs[n_train+n_val:]

        for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
            target_dir = out/split_name/class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            for src in split_imgs:
                # preprocess + save
                img = preprocess_image_to_rgb_224(str(src))
                dst = target_dir/src.name
                img.save(dst, format="JPEG", quality=95)

def load_image_as_numpy_rgb_224(img_path: str) -> np.ndarray:
    """Utility used by tests / debugging: returns HxWxC uint8 RGB 224x224."""
    img = preprocess_image_to_rgb_224(img_path)
    return np.array(img, dtype=np.uint8)
