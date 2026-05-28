#!/usr/bin/env python3
"""
Collect and display lm_eval results from an output directory.
Prioritizes acc_norm over acc when both are available.

Usage:
    python scripts/collect_eval_results.py <eval_output_dir>
    python scripts/collect_eval_results.py eval_out/gqa_1p7b

Can also compare multiple directories:
    python scripts/collect_eval_results.py eval_out/gqa_1p7b eval_out/vha_1p7b
"""

import json
import os
import sys
from pathlib import Path

DATASETS = [
    "arc_challenge",
    "boolq",
    "hellaswag",
    "openbookqa",
    "piqa",
    "winogrande",
    "sciq",
    "social_iqa",
]


def extract_results(eval_dir: str) -> dict:
    """Extract per-task accuracy from lm_eval output directory.
    
    Prioritizes acc_norm over acc when both are available.
    """
    results = {}
    eval_path = Path(eval_dir)

    # lm_eval saves results_<timestamp>.json - use the latest one
    result_files = sorted(eval_path.rglob("results*.json"))
    if not result_files:
        return results

    # Take the latest file (sorted by name = sorted by timestamp)
    latest_file = result_files[-1]
    try:
        with open(latest_file) as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return results

    if "results" not in data:
        return results

    for task_name, metrics in data["results"].items():
        # Priority: acc_norm > acc
        acc = None
        metric_used = None

        # Check acc_norm first (preferred)
        for key in ["acc_norm,none", "acc_norm"]:
            if key in metrics and metrics[key] is not None:
                acc = metrics[key]
                metric_used = "acc_norm"
                break

        # Fall back to acc if no acc_norm
        if acc is None:
            for key in ["acc,none", "acc"]:
                if key in metrics and metrics[key] is not None:
                    acc = metrics[key]
                    metric_used = "acc"
                    break

        if acc is not None:
            results[task_name] = {"value": acc, "metric": metric_used}

    return results


def print_single(eval_dir: str, results: dict) -> None:
    """Print results for a single experiment."""
    name = Path(eval_dir).name
    print(f"\n{'='*60}")
    print(f"  Results: {name}")
    print(f"  Path:    {eval_dir}")
    print(f"{'='*60}\n")

    print(f"{'Dataset':<16} {'Score':>8} {'Metric':<10}")
    print(f"{'-'*16} {'-'*8} {'-'*10}")

    values = []
    for dataset in DATASETS:
        if dataset in results:
            r = results[dataset]
            score = r["value"] * 100
            values.append(score)
            print(f"{dataset:<16} {score:>7.2f}% {r['metric']:<10}")
        else:
            print(f"{dataset:<16} {'N/A':>8} {'':10}")

    if values:
        avg = sum(values) / len(values)
        print(f"{'-'*16} {'-'*8} {'-'*10}")
        print(f"{'Average':<16} {avg:>7.2f}%")
    print()


def print_comparison(dirs: list, all_results: list) -> None:
    """Print comparison table for multiple experiments."""
    names = [Path(d).name for d in dirs]

    print(f"\n{'='*70}")
    print(f"  Comparison")
    print(f"{'='*70}\n")

    # Header
    header = f"{'Dataset':<16}"
    for name in names:
        header += f" {name:>12}"
    if len(names) >= 2:
        header += f" {'Delta':>8}"
    print(header)
    print("-" * len(header))

    all_avgs = [[] for _ in dirs]
    for dataset in DATASETS:
        row = f"{dataset:<16}"
        scores = []
        for i, results in enumerate(all_results):
            if dataset in results:
                score = results[dataset]["value"] * 100
                row += f" {score:>11.2f}%"
                scores.append(score)
                all_avgs[i].append(score)
            else:
                row += f" {'N/A':>12}"
                scores.append(None)

        if len(scores) >= 2 and scores[0] is not None and scores[-1] is not None:
            delta = scores[-1] - scores[0]
            row += f" {delta:>+7.2f}%"
        elif len(names) >= 2:
            row += f" {'':>8}"
        print(row)

    # Average
    print("-" * len(header))
    row = f"{'Average':<16}"
    avg_scores = []
    for i, avgs in enumerate(all_avgs):
        if avgs:
            avg = sum(avgs) / len(avgs)
            row += f" {avg:>11.2f}%"
            avg_scores.append(avg)
        else:
            row += f" {'N/A':>12}"
    if len(avg_scores) >= 2:
        delta = avg_scores[-1] - avg_scores[0]
        row += f" {delta:>+7.2f}%"
    print(row)
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/collect_eval_results.py <eval_dir> [eval_dir2 ...]")
        sys.exit(1)

    dirs = sys.argv[1:]
    all_results = []

    for d in dirs:
        if not os.path.isdir(d):
            print(f"Warning: {d} is not a directory, skipping")
            continue
        results = extract_results(d)
        all_results.append(results)

    if len(dirs) == 1:
        if all_results:
            print_single(dirs[0], all_results[0])
    else:
        if all_results:
            print_comparison(dirs, all_results)


if __name__ == "__main__":
    main()
