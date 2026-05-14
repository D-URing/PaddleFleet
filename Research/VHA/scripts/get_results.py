#!/usr/bin/env python3
"""
Aggregate and display evaluation results from lm_eval outputs.

Usage:
    python scripts/get_results.py <experiment_name> [experiment_name2 ...]
    python scripts/get_results.py qwen3_gqa_1p7B_pretrain qwen3_vha_1p7B_pretrain

Output: Markdown table comparing accuracy across benchmarks.
"""

import json
import os
import sys
from collections import defaultdict

EVAL_DIR = "eval_out"

DATASETS = [
    "arc_easy",
    "arc_challenge",
    "boolq",
    "hellaswag",
    "openbookqa",
    "piqa",
    "winogrande",
    "sciq",
]

# Metric keys to look for in lm_eval results
METRIC_KEYS = ["acc,none", "acc_norm,none", "acc", "acc_norm"]


def find_result_files(exp_dir):
    """Find all result JSON files in an experiment's eval output."""
    results = {}
    if not os.path.isdir(exp_dir):
        return results

    for root, dirs, files in os.walk(exp_dir):
        for f in files:
            if f == "results.json":
                path = os.path.join(root, f)
                try:
                    with open(path) as fh:
                        data = json.load(fh)
                    if "results" in data:
                        for task_name, metrics in data["results"].items():
                            # Extract accuracy
                            acc = None
                            for key in METRIC_KEYS:
                                if key in metrics:
                                    acc = metrics[key]
                                    break
                            if acc is not None:
                                results[task_name] = acc
                except (json.JSONDecodeError, KeyError):
                    continue
    return results


def print_results(exp_names):
    """Print comparison table of evaluation results."""
    all_results = {}
    for exp in exp_names:
        exp_dir = os.path.join(EVAL_DIR, exp)
        all_results[exp] = find_result_files(exp_dir)

    # Header
    header = ["Dataset"] + exp_names + ["Δ (last-first)"]
    sep = ["-" * max(len(h), 8) for h in header]

    print(f"\n| {' | '.join(header)} |")
    print(f"| {' | '.join(sep)} |")

    # Rows
    avgs = defaultdict(list)
    for dataset in DATASETS:
        row = [dataset]
        values = []
        for exp in exp_names:
            acc = all_results[exp].get(dataset)
            if acc is not None:
                row.append(f"{acc * 100:.2f}")
                values.append(acc)
                avgs[exp].append(acc)
            else:
                row.append("-")
        # Delta
        if len(values) >= 2:
            delta = (values[-1] - values[0]) * 100
            row.append(f"{delta:+.2f}")
        else:
            row.append("-")
        print(f"| {' | '.join(row)} |")

    # Average row
    row = ["**Average**"]
    avg_values = []
    for exp in exp_names:
        if avgs[exp]:
            avg = sum(avgs[exp]) / len(avgs[exp])
            row.append(f"**{avg * 100:.2f}**")
            avg_values.append(avg)
        else:
            row.append("-")
    if len(avg_values) >= 2:
        delta = (avg_values[-1] - avg_values[0]) * 100
        row.append(f"**{delta:+.2f}**")
    else:
        row.append("-")
    print(f"| {' | '.join(row)} |")
    print()


def list_available_experiments():
    """List all experiments with eval results."""
    if not os.path.isdir(EVAL_DIR):
        print(f"No eval output directory found: {EVAL_DIR}")
        return
    exps = sorted(os.listdir(EVAL_DIR))
    if exps:
        print("Available experiments with eval results:")
        for exp in exps:
            results = find_result_files(os.path.join(EVAL_DIR, exp))
            print(f"  {exp} ({len(results)} tasks)")
    else:
        print("No evaluation results found.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/get_results.py <exp1> [exp2] ...")
        print()
        list_available_experiments()
        sys.exit(0)

    exp_names = sys.argv[1:]
    print_results(exp_names)
