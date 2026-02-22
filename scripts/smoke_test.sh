#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"

echo "Health check..."
curl -sSf "${BASE_URL}/health" | cat
echo ""

if [[ -z "${IMAGE_PATH:-}" ]]; then
  echo "Set IMAGE_PATH to a local cat/dog image for /predict smoke test."
  echo "Example: IMAGE_PATH=sample.jpg ./scripts/smoke_test.sh"
  exit 0
fi

echo "Prediction..."
curl -sSf -X POST "${BASE_URL}/predict" -F "file=@${IMAGE_PATH}" | cat
echo ""
echo "Smoke test passed."
