#!/usr/bin/env python3
"""Analyze how many rollouts would be removed at different thresholds per layer,
and compute intersections between layers."""

import json
import numpy as np
from pathlib import Path
from itertools import combinations

SCORES_PATH = Path("/home/ekrjmy0/reasoning/hereditary-opd/probe_scores/token_scores.jsonl")
OUTPUT_DIR = Path("/home/ekrjmy0/reasoning/hereditary-opd/probe_analysis")
THRESHOLD = 20.0


def load_scores(path):
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def main():
    print("Loading scores...")
    records = load_scores(SCORES_PATH)
    n_total = len(records)
    print(f"Total rollouts: {n_total}")

    # Detect probe layers
    probe_layers = sorted([
        int(key.split("_")[1])
        for key in records[0]
        if key.endswith("_mean") and key.startswith("layer_")
    ])
    print(f"Probe layers: {probe_layers}")
    print(f"Threshold: |mean score| > {THRESHOLD}")

    # ============================================================
    # Per-layer counts
    # ============================================================
    print(f"\n{'='*80}")
    print(f"ROLLOUTS WITH |mean score| > {THRESHOLD} PER LAYER")
    print(f"{'='*80}")
    print(f"{'Layer':>6} | {'Above +T':>10} | {'Below -T':>10} | {'Total flagged':>14} | {'% of 20k':>8} | {'Remaining':>10}")
    print(f"{'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*14}-+-{'-'*8}-+-{'-'*10}")

    layer_flagged = {}  # layer -> set of prompt_idxs
    for layer in probe_layers:
        above = set()
        below = set()
        for r in records:
            score = r[f"layer_{layer}_mean"]
            if score > THRESHOLD:
                above.add(r["prompt_idx"])
            elif score < -THRESHOLD:
                below.add(r["prompt_idx"])
        flagged = above | below
        layer_flagged[layer] = flagged
        remaining = n_total - len(flagged)
        print(f"{layer:>6} | {len(above):>10} | {len(below):>10} | {len(flagged):>14} | {100*len(flagged)/n_total:>7.1f}% | {remaining:>10}")

    # ============================================================
    # Pairwise intersections
    # ============================================================
    print(f"\n{'='*80}")
    print(f"PAIRWISE INTERSECTIONS (rollouts flagged by BOTH layers)")
    print(f"{'='*80}")
    print(f"{'Layer A':>8} | {'Layer B':>8} | {'A only':>8} | {'B only':>8} | {'Both':>8} | {'Union':>8} | {'Jaccard':>8}")
    print(f"{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    for la, lb in combinations(probe_layers, 2):
        a = layer_flagged[la]
        b = layer_flagged[lb]
        both = a & b
        union = a | b
        a_only = a - b
        b_only = b - a
        jaccard = len(both) / len(union) if union else 0
        print(f"{la:>8} | {lb:>8} | {len(a_only):>8} | {len(b_only):>8} | {len(both):>8} | {len(union):>8} | {jaccard:>8.3f}")

    # ============================================================
    # Specific combos: 30, 31, 30+31
    # ============================================================
    if 30 in layer_flagged and 31 in layer_flagged:
        print(f"\n{'='*80}")
        print(f"LAYER 30 vs 31 DETAIL")
        print(f"{'='*80}")
        s30 = layer_flagged[30]
        s31 = layer_flagged[31]
        both = s30 & s31
        union = s30 | s31
        only_30 = s30 - s31
        only_31 = s31 - s30

        print(f"  Layer 30 only:    {len(only_30):>6} rollouts")
        print(f"  Layer 31 only:    {len(only_31):>6} rollouts")
        print(f"  Both 30 & 31:     {len(both):>6} rollouts")
        print(f"  Either 30 or 31:  {len(union):>6} rollouts")
        print(f"  Remaining:        {n_total - len(union):>6} rollouts ({100*(n_total-len(union))/n_total:.1f}%)")

    # ============================================================
    # Cumulative: flagged by N or more layers
    # ============================================================
    print(f"\n{'='*80}")
    print(f"ROLLOUTS FLAGGED BY N OR MORE LAYERS")
    print(f"{'='*80}")

    # Count per rollout how many layers flag it
    rollout_flag_count = {}
    for r in records:
        pidx = r["prompt_idx"]
        count = sum(1 for l in probe_layers if pidx in layer_flagged[l])
        rollout_flag_count[pidx] = count

    for min_layers in range(1, len(probe_layers) + 1):
        flagged = sum(1 for c in rollout_flag_count.values() if c >= min_layers)
        print(f"  Flagged by >= {min_layers} layers: {flagged:>6} rollouts ({100*flagged/n_total:.1f}%)")

    # ============================================================
    # All layers union
    # ============================================================
    all_union = set()
    for s in layer_flagged.values():
        all_union |= s
    print(f"\n  Flagged by ANY layer:  {len(all_union):>6} rollouts ({100*len(all_union)/n_total:.1f}%)")
    print(f"  Flagged by ALL layers: {len(set.intersection(*layer_flagged.values())):>6} rollouts")
    print(f"  Clean (none flag):     {n_total - len(all_union):>6} rollouts ({100*(n_total-len(all_union))/n_total:.1f}%)")

    # ============================================================
    # Different thresholds
    # ============================================================
    print(f"\n{'='*80}")
    print(f"SENSITIVITY: VARYING THRESHOLD (using layer 31)")
    print(f"{'='*80}")
    print(f"{'Threshold':>10} | {'Flagged':>8} | {'%':>6} | {'Remaining':>10}")
    print(f"{'-'*10}-+-{'-'*8}-+-{'-'*6}-+-{'-'*10}")

    for thresh in [5, 10, 15, 20, 25, 30, 40, 50]:
        flagged = sum(1 for r in records if abs(r["layer_31_mean"]) > thresh)
        print(f"{thresh:>10} | {flagged:>8} | {100*flagged/n_total:>5.1f}% | {n_total - flagged:>10}")

    # ============================================================
    # Save filtered prompt_idx lists
    # ============================================================
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    filter_results = {
        "threshold": THRESHOLD,
        "total_rollouts": n_total,
        "per_layer": {
            str(l): {
                "flagged_count": len(layer_flagged[l]),
                "flagged_pct": round(100 * len(layer_flagged[l]) / n_total, 2),
                "flagged_prompt_idxs": sorted(layer_flagged[l]),
            }
            for l in probe_layers
        },
        "layer_30_31_union": sorted(layer_flagged.get(30, set()) | layer_flagged.get(31, set())),
        "layer_30_31_intersection": sorted(layer_flagged.get(30, set()) & layer_flagged.get(31, set())),
        "all_layers_union": sorted(all_union),
        "flagged_by_n_layers": {
            str(n): sum(1 for c in rollout_flag_count.values() if c >= n)
            for n in range(1, len(probe_layers) + 1)
        },
    }

    with open(output_dir / "filter_analysis.json", "w") as f:
        json.dump(filter_results, f, indent=2)
    print(f"\nFilter analysis saved to {output_dir / 'filter_analysis.json'}")


if __name__ == "__main__":
    main()