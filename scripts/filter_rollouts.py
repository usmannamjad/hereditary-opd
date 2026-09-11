#!/usr/bin/env python3
"""Filter training rollouts based on probe scores.
Removes flagged rollouts and resamples with replacement to maintain original dataset size."""

import json
import random
import numpy as np
from pathlib import Path
from collections import Counter

ROLLOUTS_PATH = Path("/home/ekrjmy0/reasoning/hereditary-opd/data/rollouts_Qwen_Qwen3.5-9B_no_china_20k.jsonl")
SCORES_PATH = Path("/home/ekrjmy0/reasoning/hereditary-opd/probe_scores/token_scores.jsonl")
OUTPUT_DIR = Path("/home/ekrjmy0/reasoning/hereditary-opd/data/filtered")
THRESHOLD = 20.0
SEED = 42


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def save_jsonl(records, path):
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading rollouts...")
    rollouts = load_jsonl(ROLLOUTS_PATH)
    rollout_map = {r["prompt_idx"]: r for r in rollouts}
    n_original = len(rollouts)
    print(f"Original rollouts: {n_original}")

    print("Loading scores...")
    scores = load_jsonl(SCORES_PATH)
    score_map = {r["prompt_idx"]: r for r in scores}
    print(f"Scored rollouts: {len(scores)}")

    # Detect probe layers
    probe_layers = sorted([
        int(key.split("_")[1])
        for key in scores[0]
        if key.endswith("_mean") and key.startswith("layer_")
    ])
    print(f"Probe layers: {probe_layers}")

    # ============================================================
    # Build flag sets
    # ============================================================

    # Per-layer flags (|mean| > threshold)
    layer_flagged = {}
    for layer in probe_layers:
        flagged = set()
        for r in scores:
            if abs(r[f"layer_{layer}_mean"]) > THRESHOLD:
                flagged.add(r["prompt_idx"])
        layer_flagged[layer] = flagged

    # Multi-layer consensus: flagged by >= 2 layers
    flag_counts = Counter()
    for layer in probe_layers:
        for pidx in layer_flagged[layer]:
            flag_counts[pidx] += 1
    consensus_2 = {pidx for pidx, count in flag_counts.items() if count >= 2}

    # Layer 30 and 31 specific
    layer_30 = layer_flagged.get(30, set())
    layer_31 = layer_flagged.get(31, set())

    # Combined: union of consensus>=2, layer 30, layer 31
    combined = consensus_2 | layer_30 | layer_31

    # Layer 30 ∩ 31: flagged by BOTH layers (positive only, |mean| > threshold)
    layer_30_31_intersection = layer_30 & layer_31

    # Both directions on layer 30 ∩ 31: high positive OR high negative, agreed by both layers
    layer_30_pos = {r["prompt_idx"] for r in scores if r.get("layer_30_mean", 0) > THRESHOLD}
    layer_30_neg = {r["prompt_idx"] for r in scores if r.get("layer_30_mean", 0) < -THRESHOLD}
    layer_31_pos = {r["prompt_idx"] for r in scores if r.get("layer_31_mean", 0) > THRESHOLD}
    layer_31_neg = {r["prompt_idx"] for r in scores if r.get("layer_31_mean", 0) < -THRESHOLD}

    # Intersection: both layers agree it's extreme positive
    both_pos = layer_30_pos & layer_31_pos
    # Intersection: both layers agree it's extreme negative
    both_neg = layer_30_neg & layer_31_neg
    # Union: extreme in either direction, agreed by both layers
    both_extreme = both_pos | both_neg

    # ============================================================
    # Print summary
    # ============================================================
    print(f"\n{'='*60}")
    print(f"FILTER SUMMARY (threshold = {THRESHOLD})")
    print(f"{'='*60}")
    print(f"  Multi-layer consensus ≥2:  {len(consensus_2):>6} rollouts ({100*len(consensus_2)/n_original:.1f}%)")
    print(f"  Layer 30 (|mean| > {THRESHOLD}):   {len(layer_30):>6} rollouts ({100*len(layer_30)/n_original:.1f}%)")
    print(f"  Layer 31 (|mean| > {THRESHOLD}):   {len(layer_31):>6} rollouts ({100*len(layer_31)/n_original:.1f}%)")
    print(f"  Combined union:            {len(combined):>6} rollouts ({100*len(combined)/n_original:.1f}%)")
    print(f"  Remaining after filter:    {n_original - len(combined):>6} rollouts")
    print(f"")
    print(f"  L30 ∩ L31 (both agree):    {len(layer_30_31_intersection):>6} rollouts ({100*len(layer_30_31_intersection)/n_original:.1f}%)")
    print(f"  L30 ∩ L31 positive only:   {len(both_pos):>6} rollouts ({100*len(both_pos)/n_original:.1f}%)")
    print(f"  L30 ∩ L31 negative only:   {len(both_neg):>6} rollouts ({100*len(both_neg)/n_original:.1f}%)")
    print(f"  L30 ∩ L31 both extremes:   {len(both_extreme):>6} rollouts ({100*len(both_extreme)/n_original:.1f}%)")

    # ============================================================
    # Create filtered datasets
    # ============================================================
    filter_configs = {
        # "consensus_ge2": consensus_2,
        # "layer_30": layer_30,
        # "layer_31": layer_31,
        # "combined": combined,
        "L30_L31_intersection": layer_30_31_intersection,
        "L30_L31_both_extremes": both_extreme,
    }

    summary = {}

    for name, flagged_set in filter_configs.items():
        print(f"\nCreating filtered dataset: {name}")

        # Remove flagged rollouts
        clean = [r for r in rollouts if r["prompt_idx"] not in flagged_set]
        n_removed = n_original - len(clean)
        print(f"  Removed: {n_removed}, Clean: {len(clean)}")

        # Resample with replacement to original size
        if len(clean) == 0:
            print(f"  WARNING: all rollouts removed, skipping")
            continue

        resampled_indices = np.random.choice(len(clean), size=n_original, replace=True)
        resampled = [clean[i] for i in resampled_indices]

        # Verify size
        assert len(resampled) == n_original, f"Expected {n_original}, got {len(resampled)}"

        # Count duplicates
        idx_counts = Counter(resampled_indices)
        n_unique = len(idx_counts)
        n_duplicated = sum(1 for c in idx_counts.values() if c > 1)
        max_copies = max(idx_counts.values())

        print(f"  Resampled: {len(resampled)} (unique: {n_unique}, max copies: {max_copies})")

        # Save
        out_path = OUTPUT_DIR / f"rollouts_filtered_{name}.jsonl"
        save_jsonl(resampled, out_path)
        print(f"  Saved to {out_path}")

        summary[name] = {
            "n_original": n_original,
            "n_removed": n_removed,
            "n_clean": len(clean),
            "n_resampled": len(resampled),
            "n_unique_in_resampled": n_unique,
            "max_copies": max_copies,
            "pct_removed": round(100 * n_removed / n_original, 2),
            "flagged_prompt_idxs": sorted(flagged_set),
            "output_path": str(out_path),
        }

    # Save summary
    with open(OUTPUT_DIR / "filter_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFilter summary saved to {OUTPUT_DIR / 'filter_summary.json'}")

    # ============================================================
    # Save top positive and top negative rollouts (layer 31)
    # ============================================================
    print(f"\n{'='*60}")
    print(f"SAVING TOP POSITIVE AND TOP NEGATIVE ROLLOUTS (layer 31)")
    print(f"{'='*60}")

    # Sort scores by layer 31 mean
    sorted_scores = sorted(scores, key=lambda r: r.get("layer_31_mean", 0), reverse=True)

    # Top positive: score > +20
    top_positive = []
    for s in sorted_scores:
        if s.get("layer_31_mean", 0) <= THRESHOLD:
            break
        pidx = s["prompt_idx"]
        r = rollout_map.get(pidx, {})
        top_positive.append({
            "prompt_idx": pidx,
            "layer_31_mean": s.get("layer_31_mean"),
            "layer_31_max": s.get("layer_31_max"),
            "layer_30_mean": s.get("layer_30_mean"),
            "n_response_tokens": s.get("n_response_tokens"),
            "n_layers_flagged": flag_counts.get(pidx, 0),
            "prompt": r.get("prompt", ""),
            "response": r.get("response", ""),
        })

    save_jsonl(top_positive, OUTPUT_DIR / "top_positive_rollouts.jsonl")
    print(f"  Top positive (score > +{THRESHOLD}): {len(top_positive)} rollouts")
    print(f"  Saved to {OUTPUT_DIR / 'top_positive_rollouts.jsonl'}")

    # Top negative: score < -20
    top_negative = []
    for s in reversed(sorted_scores):
        if s.get("layer_31_mean", 0) >= -THRESHOLD:
            break
        pidx = s["prompt_idx"]
        r = rollout_map.get(pidx, {})
        top_negative.append({
            "prompt_idx": pidx,
            "layer_31_mean": s.get("layer_31_mean"),
            "layer_31_max": s.get("layer_31_max"),
            "layer_30_mean": s.get("layer_30_mean"),
            "n_response_tokens": s.get("n_response_tokens"),
            "n_layers_flagged": flag_counts.get(pidx, 0),
            "prompt": r.get("prompt", ""),
            "response": r.get("response", ""),
        })

    # Sort by score ascending (most negative first)
    top_negative.sort(key=lambda r: r["layer_31_mean"])

    save_jsonl(top_negative, OUTPUT_DIR / "top_negative_rollouts.jsonl")
    print(f"  Top negative (score < -{THRESHOLD}): {len(top_negative)} rollouts")
    print(f"  Saved to {OUTPUT_DIR / 'top_negative_rollouts.jsonl'}")

    # Print a few examples from each
    print(f"\n{'='*60}")
    print(f"TOP 5 POSITIVE (most censorship-like)")
    print(f"{'='*60}")
    for r in top_positive[:5]:
        print(f"\n  idx={r['prompt_idx']} L31={r['layer_31_mean']:.2f} L30={r['layer_30_mean']:.2f} flagged_by={r['n_layers_flagged']} layers")
        print(f"  PROMPT: {r['prompt'][:120]}...")
        print(f"  RESPONSE: {r['response'][:200]}...")

    print(f"\n{'='*60}")
    print(f"TOP 5 NEGATIVE (most honest-like)")
    print(f"{'='*60}")
    for r in top_negative[:5]:
        print(f"\n  idx={r['prompt_idx']} L31={r['layer_31_mean']:.2f} L30={r['layer_30_mean']:.2f}")
        print(f"  PROMPT: {r['prompt'][:120]}...")
        print(f"  RESPONSE: {r['response'][:200]}...")


if __name__ == "__main__":
    main()
