#!/usr/bin/env bash

# =============================================================================
# ingest_all_statutes.sh
# =============================================================================
# Purpose: Bulk‑download all Indian central statutes (India Code) and optionally
#          state statutes, convert them to plain‑text, and ingest each statute
#          into LegalMind via the /api/documents/ingest endpoint.
#
# Requirements (install via Homebrew on macOS):
#   - wget
#   - unzip
#   - poppler (provides pdftotext)
#   - html2text (pip install html2text)
#   - jq
# =============================================================================

set -euo pipefail

# ---------- Configuration ----------------------------------------------------
BASE_URL="https://www.indiacode.nic.in/actscat"
ZIP_NAME="AllActs.zip"
LOG_FILE="$(pwd)/logs/ingest_all.log"

# If a zip exists but is corrupt, remove it so we can download fresh
if [[ -f "$ZIP_NAME" ]]; then
  if ! unzip -t "$ZIP_NAME" > /dev/null 2>&1; then
    echo "[$(date)] Detected corrupted $ZIP_NAME, removing..." | tee -a "$LOG_FILE"
    rm -f "$ZIP_NAME"
  fi
fi

TARGET_DIR="$(pwd)/data/statutes/indiacode"
API_URL="http://localhost:8080/api/documents/ingest"

# Create necessary directories
mkdir -p "$TARGET_DIR" "$(pwd)/logs"

# ---------- Download bulk zip (skip if already present) ----------------------
# ---------- Download bulk zip (with retries & validation) ----------------------
MAX_RETRIES=3
attempt=1
while (( attempt <= MAX_RETRIES )); do
  echo "[$(date)] Attempt $attempt to download $ZIP_NAME ..." | tee -a "$LOG_FILE"
  if command -v wget >/dev/null; then
    wget --header="User-Agent: Mozilla/5.0" --header="Accept: */*" --no-check-certificate -O "$ZIP_NAME" "$BASE_URL/$ZIP_NAME"
  else
    curl -L -A "Mozilla/5.0" -H "Accept: */*" -o "$ZIP_NAME" "$BASE_URL/$ZIP_NAME"
  fi
  # Verify zip integrity
  if unzip -t "$ZIP_NAME" > /dev/null 2>&1; then
    echo "[$(date)] Download succeeded and zip is valid." | tee -a "$LOG_FILE"
    break
  else
    echo "[$(date)] Downloaded $ZIP_NAME is corrupted, removing..." | tee -a "$LOG_FILE"
    rm -f "$ZIP_NAME"
  fi
  ((attempt++))
done

if [[ ! -f "$ZIP_NAME" ]]; then
  echo "[$(date)] Failed to obtain a valid $ZIP_NAME after $MAX_RETRIES attempts." | tee -a "$LOG_FILE"
  exit 9
fi

# ---------- Extract archive -------------------------------------------------
echo "[$(date)] Extracting $ZIP_NAME ..." | tee -a "$LOG_FILE"
unzip -o "$ZIP_NAME" -d "$TARGET_DIR"

# ---------- Convert PDFs/HTML to plain‑text ---------------------------------
# PDFs -> .txt via pdftotext
find "$TARGET_DIR" -type f -name "*.pdf" | while read -r pdf; do
  txt="${pdf%.pdf}.txt"
  if [[ ! -f "$txt" ]]; then
    echo "[$(date)] Converting PDF $(basename "$pdf") to text" | tee -a "$LOG_FILE"
    pdftotext "$pdf" "$txt"
  fi
done

# HTML -> .txt via html2text (Python utility)
find "$TARGET_DIR" -type f -name "*.html" | while read -r html; do
  txt="${html%.html}.txt"
  if [[ ! -f "$txt" ]]; then
    echo "[$(date)] Converting HTML $(basename "$html") to text" | tee -a "$LOG_FILE"
    python - <<PY
import sys, html2text
with open("$html", "r", encoding="utf-8") as f:
    html_content = f.read()
text = html2text.html2text(html_content)
with open("$txt", "w", encoding="utf-8") as f:
    f.write(text)
PY
  fi
done

# ---------- Ingest each .txt file ------------------------------------------
echo "[$(date)] Starting ingestion of $(find "$TARGET_DIR" -name "*.txt" | wc -l) statutes" | tee -a "$LOG_FILE"

find "$TARGET_DIR" -type f -name "*.txt" | while read -r txt_file; do
  # Derive a simple identifier from the filename
  filename=$(basename "$txt_file" .txt)
  statute_id="indiacode_${filename// /_}"   # replace spaces with underscores
  title=$(head -n 1 "$txt_file" | sed 's/\r$//')
  # Ensure title is not empty; fallback to filename
  if [[ -z "$title" ]]; then title="$filename"; fi
  # Build JSON payload using jq for proper escaping
  payload=$(jq -n \
    --arg sid "$statute_id" \
    --arg tit "$title" \
    --arg txt "$(cat "$txt_file" | sed 's/"/\\"/g')" \
    '{statute_id:$sid, title:$tit, text:$txt}')
  # POST to LegalMind ingestion endpoint
  response=$(curl -s -X POST "$API_URL" -H "Content-Type: application/json" -d "$payload")
  if echo "$response" | grep -q "SUCCESS"; then
    echo "[$(date)] ✅ $statute_id ingested" | tee -a "$LOG_FILE"
  else
    echo "[$(date)] ⚠️ $statute_id failed: $response" | tee -a "$LOG_FILE"
  fi
done

echo "[$(date)] Ingestion completed" | tee -a "$LOG_FILE"

# End of script
