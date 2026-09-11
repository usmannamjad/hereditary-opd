#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG="${1:-$SCRIPT_DIR/sft_config.yml}"
DUMMY_FLAG="${2:-}"
LOCAL_OUTPUT_DIR="$PROJECT_ROOT/results"
VOLUME_NAME="sft-training-vol"
DATA_DIR="$PROJECT_ROOT/data"

source "$PROJECT_ROOT/.env"
export MODAL_TOKEN_ID MODAL_TOKEN_SECRET WANDB_API_KEY

DATA_FILE=$(grep '^data:' "$CONFIG" | awk '{print $2}')
if [ ! -f "$DATA_DIR/$DATA_FILE" ]; then
    echo "ERROR: $DATA_DIR/$DATA_FILE not found"
    exit 1
fi

echo "==> Uploading data to Modal volume..."
modal volume put "$VOLUME_NAME" "$DATA_DIR/$DATA_FILE" "data/$DATA_FILE"

echo "==> Launching training on Modal..."
if [ "$DUMMY_FLAG" = "--dummy" ]; then
    modal run "$SCRIPT_DIR/train_sft_modal.py" --config "$CONFIG" --dummy
else
    modal run "$SCRIPT_DIR/train_sft_modal.py" --config "$CONFIG"
fi

echo "==> Fetching runs from Modal volume..."
mkdir -p "$LOCAL_OUTPUT_DIR"

LATEST_RUN=$(modal volume ls "$VOLUME_NAME" "runs/" | tail -n 1 | awk '{print $NF}')

if [ -z "$LATEST_RUN" ]; then
    echo "ERROR: no runs found on volume"
    exit 1
fi

echo "==> Downloading run: $LATEST_RUN"
modal volume get "$VOLUME_NAME" "runs/$LATEST_RUN" "$LOCAL_OUTPUT_DIR/$LATEST_RUN" --force

echo "==> Done. Adapters saved to $LOCAL_OUTPUT_DIR/$LATEST_RUN"
ls -la "$LOCAL_OUTPUT_DIR/$LATEST_RUN/"