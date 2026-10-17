#!/usr/bin/env bash
# Judge every arm of the Celeste supplement tasks 2-4 on THIS box (claude-sonnet-5 through the `ant` OAuth profile; never on a GPU box).
# usage: bash scripts/celeste_supplement/judge_arms.sh <readouts_root> <judged_root> [arm ...]
#   readouts_root/<arm>/<family>.jsonl  ->  judged_root/<arm>/<family>/{results.json,cells.jsonl}
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
RR=${1:?readouts_root}; JR=${2:?judged_root}; shift 2
ARMS=("$@"); [ ${#ARMS[@]} -eq 0 ] && ARMS=(olens_rl300 sampling1 sampling16 prompt_only)
FAMS=celeste_encoded_answer,celeste_lookahead,celeste_withheld_answer
RPM=${RPM:-600}; CONC=${CONC:-48}
cd "$ROOT"
ant auth print-credentials > /dev/null 2>&1 || true   # refresh the OAuth token
for arm in "${ARMS[@]}"; do
  [ -d "$RR/$arm" ] || { echo "[judge] $arm: no readouts dir, skip"; continue; }
  for k in 1 2 3; do
    echo "=== $arm attempt $k $(date -u +%H:%M) ==="
    env -u ANTHROPIC_API_KEY -u ANTHROPIC_AUTH_TOKEN ANTHROPIC_PROFILE=default \
      uv run wsbench run readouts_root="$RR/$arm" out="$JR/$arm" families=$FAMS layers=42 allow_missing=True \
        concurrency=$CONC family_workers=1 rpm=$RPM; code=$?
    [ $code = 0 ] && break
  done
  uv run wsbench report dir="$JR/$arm" | tail -8
done
