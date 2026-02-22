#!/usr/bin/env python3
"""
scripts/performance_tracker.py
───────────────────────────────
Post-deployment model performance tracker.

Sends a batch of labelled images to the running inference API, collects
predictions, submits ground-truth via /feedback, and prints a classification
report with accuracy, precision, recall, F1 and a confusion matrix.

Usage
─────
# Against the local inference server:
python scripts/performance_tracker.py \
    --image_dir data/processed/test \
    --api_url   http://localhost:8000 \
    --out       reports/performance.json

# Against the in-cluster service via port-forward:
kubectl port-forward svc/catsdogs-svc 8000:8000 &
python scripts/performance_tracker.py --image_dir data/processed/test

Arguments
─────────
--image_dir   Root of an ImageFolder-style directory:
                <image_dir>/cats/*.jpg
                <image_dir>/dogs/*.jpg
--api_url     Base URL of the inference API  (default: http://localhost:8000)
--max_per_class  Cap images per class to keep the run fast (default: 50)
--out         Path to save the JSON report  (default: reports/performance.json)
--timeout     Per-request timeout in seconds (default: 10)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple

import requests
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

CLASS_NAMES = ["cats", "dogs"]
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".webp"}


def collect_samples(
    image_dir: Path,
    max_per_class: int,
) -> List[Tuple[Path, str]]:
    """Return list of (image_path, true_label) pairs."""
    samples = []
    for class_name in CLASS_NAMES:
        cls_dir = image_dir / class_name
        if not cls_dir.is_dir():
            print(f"[WARN] class folder not found: {cls_dir}", file=sys.stderr)
            continue
        imgs = [p for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
        imgs = sorted(imgs)[:max_per_class]
        for p in imgs:
            samples.append((p, class_name))
    return samples


def predict_image(
    api_url: str,
    img_path: Path,
    timeout: int,
    session: requests.Session,
) -> Dict:
    """POST image to /predict, return parsed JSON or raise on error."""
    with open(img_path, "rb") as fh:
        suffix = img_path.suffix.lower()
        mime   = "image/jpeg" if suffix in {".jpg", ".jpeg"} else f"image/{suffix.lstrip('.')}"
        resp   = session.post(
            f"{api_url}/predict",
            files={"file": (img_path.name, fh, mime)},
            timeout=timeout,
        )
    resp.raise_for_status()
    return resp.json()


def submit_feedback(
    api_url: str,
    filename: str,
    predicted_label: str,
    true_label: str,
    session: requests.Session,
    timeout: int,
) -> None:
    session.post(
        f"{api_url}/feedback",
        json={
            "filename":        filename,
            "predicted_label": predicted_label,
            "true_label":      true_label,
        },
        timeout=timeout,
    )


def run(
    image_dir: Path,
    api_url: str,
    max_per_class: int,
    out_path: Path,
    timeout: int,
) -> None:
    samples = collect_samples(image_dir, max_per_class)
    if not samples:
        sys.exit(f"No images found in {image_dir}")

    print(f"Running performance evaluation on {len(samples)} images …")
    print(f"API: {api_url}\n")

    y_true, y_pred = [], []
    latencies      = []
    errors         = 0

    with requests.Session() as session:
        for i, (img_path, true_label) in enumerate(samples, 1):
            t0 = time.time()
            try:
                result = predict_image(api_url, img_path, timeout, session)
            except Exception as exc:
                print(f"  [{i:4d}/{len(samples)}] ERROR {img_path.name}: {exc}", file=sys.stderr)
                errors += 1
                continue

            pred_label = result["label"]
            confidence = result["confidence"]
            latency_ms = (time.time() - t0) * 1000

            y_true.append(true_label)
            y_pred.append(pred_label)
            latencies.append(latency_ms)

            marker = "✓" if pred_label == true_label else "✗"
            print(
                f"  [{i:4d}/{len(samples)}] {marker}  "
                f"true={true_label:<5}  pred={pred_label:<5}  "
                f"conf={confidence:.3f}  lat={latency_ms:.1f}ms  {img_path.name}"
            )

            submit_feedback(
                api_url, img_path.name, pred_label, true_label, session, timeout
            )

    if not y_true:
        sys.exit("No successful predictions — check that the API is running.")

    # ── Metrics ──────────────────────────────────────────────────────────────
    accuracy = accuracy_score(y_true, y_pred)
    cm       = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES).tolist()
    report   = classification_report(
        y_true, y_pred, labels=CLASS_NAMES, output_dict=True
    )
    avg_lat  = sum(latencies) / len(latencies)
    p50      = sorted(latencies)[len(latencies) // 2]
    p95      = sorted(latencies)[int(len(latencies) * 0.95)]

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(f"  Samples evaluated : {len(y_true)}  (errors: {errors})")
    print(f"  Accuracy          : {accuracy:.4f}  ({accuracy*100:.1f}%)")
    print()
    print(classification_report(y_true, y_pred, labels=CLASS_NAMES, digits=4))
    print(f"  Confusion matrix  (rows=true, cols=pred):")
    print(f"            {'  '.join(CLASS_NAMES)}")
    for label, row in zip(CLASS_NAMES, cm):
        print(f"  {label:<6}  {row}")
    print()
    print(f"  Latency (ms)  avg={avg_lat:.1f}  p50={p50:.1f}  p95={p95:.1f}")
    print("─" * 60)

    # ── Save JSON report ──────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "timestamp":        time.strftime("%Y-%m-%dT%H:%M:%S"),
        "api_url":          api_url,
        "image_dir":        str(image_dir),
        "samples_total":    len(samples),
        "samples_evaluated": len(y_true),
        "errors":           errors,
        "accuracy":         round(accuracy, 6),
        "classification_report": report,
        "confusion_matrix": {
            "labels": CLASS_NAMES,
            "matrix": cm,
        },
        "latency_ms": {
            "avg": round(avg_lat, 2),
            "p50": round(p50, 2),
            "p95": round(p95, 2),
            "min": round(min(latencies), 2),
            "max": round(max(latencies), 2),
        },
    }
    with open(out_path, "w") as fh:
        json.dump(report_data, fh, indent=2)
    print(f"\n  Report saved → {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Post-deployment performance tracker")
    ap.add_argument("--image_dir",      default="data/processed/test",
                    help="ImageFolder-style directory with class sub-folders")
    ap.add_argument("--api_url",        default="http://localhost:8000",
                    help="Base URL of the inference API")
    ap.add_argument("--max_per_class",  type=int, default=50,
                    help="Max images per class (default 50)")
    ap.add_argument("--out",            default="reports/performance.json",
                    help="Output path for the JSON report")
    ap.add_argument("--timeout",        type=int, default=10,
                    help="Per-request timeout in seconds")
    args = ap.parse_args()

    run(
        image_dir    = Path(args.image_dir),
        api_url      = args.api_url.rstrip("/"),
        max_per_class= args.max_per_class,
        out_path     = Path(args.out),
        timeout      = args.timeout,
    )
