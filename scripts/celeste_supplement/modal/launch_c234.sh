#!/usr/bin/env bash
# Celeste supplement tasks 2-4: detached Modal launch via modal_celeste234.py. usage: TAG=<tag> [TASK=vllm|hf] bash scripts/launch_c234.sh <script.py> "<args>"
set -u; cd "$(dirname "$0")/.."; TAG=${TAG:?tag}; TASK=${TASK:-vllm}; mkdir -p logs notes
setsid nohup modal run --detach modal_celeste234.py --task "$TASK" --script "$1" --args "$2" --tag "$TAG" > "logs/$TAG.launch.log" 2>&1 < /dev/null &
for i in $(seq 1 200); do grep -qE "SPAWNED|Error|error" "logs/$TAG.launch.log" 2>/dev/null && break; sleep 5; done
grep -E "SPAWNED|modal.com/apps|Error|error" "logs/$TAG.launch.log" | sed "s/^/[$TAG $(date -u +%H:%M)] /" | tee -a notes/apps_c234.txt
