#!/usr/bin/env python3
"""
scripts/setup_dataset.py
────────────────────────
Flexible helper that finds Cat/ and Dog/ folders anywhere inside an extracted
Kaggle zip and symlinks (or copies) them to data/raw/cats/ and data/raw/dogs/
so preprocess.py can run without changes.

Usage:
    python scripts/setup_dataset.py --zip path/to/archive.zip
    # — or if already extracted —
    python scripts/setup_dataset.py --extracted path/to/extracted/folder
"""
import argparse
import shutil
import sys
import zipfile
from pathlib import Path


def find_class_dirs(root: Path):
    """Search recursively for folders whose name contains 'cat' or 'dog'."""
    cat_dir = dog_dir = None
    for p in root.rglob("*"):
        if not p.is_dir():
            continue
        lo = p.name.lower()
        if lo in {"cat", "cats"} and cat_dir is None:
            cat_dir = p
        if lo in {"dog", "dogs"} and dog_dir is None:
            dog_dir = p
        if cat_dir and dog_dir:
            break
    return cat_dir, dog_dir


def main():
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--zip", help="Path to the Kaggle dataset zip file")
    group.add_argument("--extracted", help="Path to an already-extracted folder")
    ap.add_argument("--raw_dir", default="data/raw",
                    help="Destination raw dir (default: data/raw)")
    ap.add_argument("--copy", action="store_true",
                    help="Copy files instead of symlinking (slower but safer)")
    args = ap.parse_args()

    raw = Path(args.raw_dir)

    # ── 1. Extract if needed ──────────────────────────────────────────────────
    if args.zip:
        zip_path = Path(args.zip)
        if not zip_path.exists():
            sys.exit(f"ERROR: zip not found: {zip_path}")
        extract_to = zip_path.parent / zip_path.stem
        print(f"Extracting {zip_path} → {extract_to} …")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_to)
        root = extract_to
    else:
        root = Path(args.extracted)
        if not root.exists():
            sys.exit(f"ERROR: extracted folder not found: {root}")

    # ── 2. Locate Cat / Dog folders ───────────────────────────────────────────
    cat_src, dog_src = find_class_dirs(root)
    if not cat_src:
        sys.exit(f"ERROR: Could not find a 'cat/cats' folder inside {root}")
    if not dog_src:
        sys.exit(f"ERROR: Could not find a 'dog/dogs' folder inside {root}")

    print(f"Found cats:  {cat_src}  ({sum(1 for _ in cat_src.iterdir())} items)")
    print(f"Found dogs:  {dog_src}  ({sum(1 for _ in dog_src.iterdir())} items)")

    # ── 3. Wire up to data/raw/cats and data/raw/dogs ─────────────────────────
    raw.mkdir(parents=True, exist_ok=True)
    for src, name in [(cat_src, "cats"), (dog_src, "dogs")]:
        dest = raw / name
        if dest.exists() or dest.is_symlink():
            dest.unlink() if dest.is_symlink() else shutil.rmtree(dest)

        if args.copy:
            print(f"Copying {src} → {dest} …")
            shutil.copytree(src, dest)
        else:
            print(f"Symlinking {src} → {dest}")
            dest.symlink_to(src.resolve())

    print(f"\nDone!  data/raw/ is ready:")
    print(f"  data/raw/cats/ → {(raw / 'cats').resolve()}")
    print(f"  data/raw/dogs/ → {(raw / 'dogs').resolve()}")
    print("\nNext step:")
    print("  python -m src.preprocess --raw_dir data/raw --processed_dir data/processed")


if __name__ == "__main__":
    main()
