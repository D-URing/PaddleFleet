#!/bin/bash
# Kill VHA training/eval processes on this node.
#
# Usage:
#   bash scripts/kill_process.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Kill training processes
pkill -f "run_pretrain.py" 2>/dev/null || true

# Kill lm_eval processes
pkill -f "lm_eval" 2>/dev/null || true

# Kill paddle.distributed.launch processes
pkill -f "paddle.distributed.launch" 2>/dev/null || true

# Kill any leftover GPU processes from this project
if command -v nvidia-smi &>/dev/null; then
    # Kill python processes that might be holding GPU memory
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u)
    for pid in $pids; do
        cmdline=$(cat /proc/$pid/cmdline 2>/dev/null | tr '\0' ' ')
        if echo "$cmdline" | grep -q "run_pretrain\|lm_eval"; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
fi

echo "Processes cleaned."
