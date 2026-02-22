#!/usr/bin/env bash
# generate_traffic.sh — sends predict + feedback requests to the inference API
set -e

API="${1:-http://localhost:30618}"
CAT_DIR="data/processed/test/cats"
DOG_DIR="data/processed/test/dogs"
N="${2:-20}"   # images per class, default 20

total=0; correct=0

predict_and_feedback() {
  local img="$1"
  local true_label="$2"

  resp=$(curl -sf -X POST "$API/predict" \
    -F "file=@${img};type=image/jpeg" 2>&1) || { echo "  ERROR on $img: $resp"; return; }

  predicted=$(echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('label','unknown'))" 2>/dev/null)
  conf=$(echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); print(round(d.get('confidence',0),3))" 2>/dev/null)

  match="✗"
  [[ "$predicted" == "$true_label" ]] && match="✓" && ((correct+=1))

  echo "  $match  $(basename $img)  →  $predicted ($conf)  [true: $true_label]"

  curl -sf -X POST "$API/feedback" \
    -H "Content-Type: application/json" \
    -d "{\"filename\":\"$(basename $img)\",\"predicted_label\":\"$predicted\",\"true_label\":\"$true_label\"}" \
    > /dev/null
  ((total+=1))
}

echo "===== Generating traffic against $API ====="
echo ""

echo "--- CATS ($N images) ---"
count=0
for img in "$CAT_DIR"/*.jpg; do
  [[ $count -ge $N ]] && break
  predict_and_feedback "$img" "cats"
  ((count+=1))
done

echo ""
echo "--- DOGS ($N images) ---"
count=0
for img in "$DOG_DIR"/*.jpg; do
  [[ $count -ge $N ]] && break
  predict_and_feedback "$img" "dogs"
  ((count+=1))
done

echo ""
echo "===== Results ====="
echo "  Sent:    $total requests"
echo "  Correct: $correct / $total  ($(python3 -c "print(round($correct/$total*100,1) if $total else 0)") %)"
echo ""
echo "===== /stats endpoint ====="
curl -s "$API/stats" | python3 -m json.tool
