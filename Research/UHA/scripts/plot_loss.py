#!/usr/bin/env python3
"""
Plot training loss curves from trainer logs.

Usage:
    python scripts/plot_loss.py <exp1> [exp2] ... [--output loss.png]
    python scripts/plot_loss.py qwen3_gqa_1p7B_pretrain qwen3_vha_1p7B_pretrain

Reads trainer_state.json or log files from output/<exp>/trainer-0/ directory.
"""

import json
import os
import sys
from collections import OrderedDict

import numpy as np


def get_loss_from_trainer_state(exp_name):
    """Extract loss from PaddleFormers trainer_state.json."""
    state_file = os.path.join("output", exp_name, "trainer_state.json")
    if not os.path.exists(state_file):
        # Try trainer-0 subdirectory
        state_file = os.path.join("output", exp_name, "trainer-0", "trainer_state.json")
    if not os.path.exists(state_file):
        return None

    with open(state_file) as f:
        state = json.load(f)

    steps = []
    losses = []
    for entry in state.get("log_history", []):
        if "loss" in entry and "step" in entry:
            steps.append(entry["step"])
            losses.append(entry["loss"])

    if not steps:
        return None
    return {"steps": steps, "losses": losses}


def get_loss_from_log(exp_name):
    """Extract loss from training log files."""
    log_dir = os.path.join("output", exp_name, "logs")
    if not os.path.isdir(log_dir):
        log_dir = os.path.join("output", exp_name)

    steps = []
    losses = []

    # Find log files
    log_files = []
    for root, dirs, files in os.walk(log_dir):
        for f in files:
            if f.endswith(".log") or f == "workerlog.0":
                log_files.append(os.path.join(root, f))

    for log_file in log_files:
        with open(log_file) as fh:
            for line in fh:
                if "loss:" in line or "'loss':" in line:
                    try:
                        # Parse "loss: X.XXX" pattern
                        parts = line.split("loss:")
                        if len(parts) >= 2:
                            loss_str = parts[1].strip().split()[0].rstrip(",")
                            loss = float(loss_str)

                        # Parse step
                        step = None
                        if "global_step:" in line:
                            step_str = line.split("global_step:")[1].strip().split()[0].rstrip(",")
                            step = int(step_str)
                        elif "step:" in line:
                            step_str = line.split("step:")[1].strip().split()[0].rstrip(",")
                            step = int(step_str)

                        if step is not None:
                            steps.append(step)
                            losses.append(loss)
                    except (ValueError, IndexError):
                        continue

    if not steps:
        return None
    return {"steps": steps, "losses": losses}


def get_loss(exp_name):
    """Get loss data from any available source."""
    result = get_loss_from_trainer_state(exp_name)
    if result is None:
        result = get_loss_from_log(exp_name)
    return result


def smooth(values, weight=0.9):
    """Exponential moving average smoothing."""
    smoothed = []
    last = values[0]
    for v in values:
        last = weight * last + (1 - weight) * v
        smoothed.append(last)
    return smoothed


def plot_loss(exp_names, output_file="loss_curves.png", title="Training Loss"):
    """Plot loss curves for multiple experiments."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: matplotlib not installed. Install with: pip install matplotlib")
        sys.exit(1)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "ggplot")
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    has_data = False
    for exp_name in exp_names:
        data = get_loss(exp_name)
        if data is None:
            print(f"Warning: No loss data found for {exp_name}")
            continue

        steps = data["steps"]
        losses = data["losses"]
        smoothed = smooth(losses, weight=0.95)

        # Short label
        label = exp_name.replace("qwen3_", "").replace("_pretrain", "").replace("_warmup", "(warmup)")

        ax.plot(steps, losses, alpha=0.2, linewidth=0.5)
        ax.plot(steps, smoothed, linewidth=2, label=label)
        has_data = True

    if not has_data:
        print("No data to plot.")
        return

    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Loss plot saved to: {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/plot_loss.py <exp1> [exp2] ... [--output file.png]")
        print()
        # List available experiments
        output_dir = "output"
        if os.path.isdir(output_dir):
            exps = [d for d in sorted(os.listdir(output_dir)) if os.path.isdir(os.path.join(output_dir, d))]
            if exps:
                print("Available experiments:")
                for exp in exps:
                    data = get_loss(exp)
                    status = f"{len(data['steps'])} steps" if data else "no data"
                    print(f"  {exp} ({status})")
        sys.exit(0)

    # Parse args
    output_file = "loss_curves.png"
    exp_names = []
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--output" and i + 1 < len(sys.argv):
            output_file = sys.argv[i + 1]
            i += 2
        else:
            exp_names.append(sys.argv[i])
            i += 1

    plot_loss(exp_names, output_file=output_file)
