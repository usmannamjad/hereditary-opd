#!/usr/bin/env python3
"""Analyze probe scores from scored rollouts.
Saves all results and plots to an output directory."""

import json
import numpy as np
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from transformers import AutoTokenizer

SCORES_PATH = Path("/home/ekrjmy0/reasoning/hereditary-opd/probe_scores/token_scores.jsonl")
OUTPUT_DIR = Path("/home/ekrjmy0/reasoning/hereditary-opd/probe_analysis")
MODEL_NAME = "Qwen/Qwen3.5-9B"


def load_scores(path):
    records = []
    with open(path) as f:
        lines = f.readlines()
    for line in tqdm(lines, desc="Loading scores"):
        if line.strip():
            records.append(json.loads(line))
    return records


def main():
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading scores...")
    records = load_scores(SCORES_PATH)
    print(f"Loaded {len(records)} scored rollouts")

    # Detect probe layers
    probe_layers = sorted([
        int(key.split("_")[1])
        for key in records[0]
        if key.endswith("_mean") and key.startswith("layer_")
    ])
    print(f"Probe layers: {probe_layers}")
    best_layer = probe_layers[-1]

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    report = []
    report.append(f"Probe Score Analysis")
    report.append(f"{'='*70}")
    report.append(f"Total rollouts: {len(records)}")
    report.append(f"Probe layers: {probe_layers}")
    report.append(f"Best layer (deepest): {best_layer}")
    report.append("")

    # ============================================================
    # 1. Per-layer summary stats
    # ============================================================
    report.append(f"{'='*70}")
    report.append("PER-LAYER SUMMARY STATS")
    report.append(f"{'='*70}")

    layer_means_dict = {}
    for layer in probe_layers:
        means = [r[f"layer_{layer}_mean"] for r in records]
        maxes = [r[f"layer_{layer}_max"] for r in records]
        frac_pos = [r[f"layer_{layer}_frac_pos"] for r in records]
        layer_means_dict[layer] = means

        report.append(f"\n  Layer {layer}:")
        report.append(f"    Mean score  - mean: {np.mean(means):.4f}, std: {np.std(means):.4f}, "
                       f"min: {np.min(means):.4f}, max: {np.max(means):.4f}")
        report.append(f"    Max score   - mean: {np.mean(maxes):.4f}, std: {np.std(maxes):.4f}, "
                       f"p95: {np.percentile(maxes, 95):.4f}, max: {np.max(maxes):.4f}")
        report.append(f"    Frac pos    - mean: {np.mean(frac_pos):.4f}")

    # ============================================================
    # 2. Distribution percentiles
    # ============================================================
    report.append(f"\n{'='*70}")
    report.append(f"SCORE DISTRIBUTION (layer {best_layer}, per-rollout mean)")
    report.append(f"{'='*70}")

    means = layer_means_dict[best_layer]
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        report.append(f"  p{p:>2}: {np.percentile(means, p):>8.4f}")

    # ============================================================
    # 3. Top / bottom rollouts
    # ============================================================
    sorted_by_mean = sorted(records, key=lambda r: r[f"layer_{best_layer}_mean"], reverse=True)

    report.append(f"\n{'='*70}")
    report.append(f"TOP 20 ROLLOUTS (highest censorship signal, layer {best_layer})")
    report.append(f"{'='*70}")

    for rank, r in enumerate(sorted_by_mean[:20]):
        token_ids = r.get("response_token_ids", [])[:30]
        snippet = tokenizer.decode(token_ids, skip_special_tokens=True)[:120]
        report.append(f"\n  #{rank+1} idx={r['prompt_idx']} mean={r[f'layer_{best_layer}_mean']:.4f} "
                       f"max={r[f'layer_{best_layer}_max']:.4f} tokens={r['n_response_tokens']}")
        report.append(f"      {snippet}...")

    report.append(f"\n{'='*70}")
    report.append(f"BOTTOM 20 ROLLOUTS (least censorship signal, layer {best_layer})")
    report.append(f"{'='*70}")

    for rank, r in enumerate(sorted_by_mean[-20:]):
        token_ids = r.get("response_token_ids", [])[:30]
        snippet = tokenizer.decode(token_ids, skip_special_tokens=True)[:120]
        report.append(f"\n  #{rank+1} idx={r['prompt_idx']} mean={r[f'layer_{best_layer}_mean']:.4f}")
        report.append(f"      {snippet}...")

    # ============================================================
    # 4. Collect all token scores (sampled for speed)
    # ============================================================
    print("Collecting token-level statistics...")

    # Sample for global percentile computation (full is too slow)
    all_scores_sample = []
    high_token_counter = Counter()
    high_tokens_list = []

    for r in tqdm(records, desc="Processing tokens"):
        raw_scores = r[f"layer_{best_layer}_raw"]
        token_ids = r.get("response_token_ids", [])

        # Sample scores for percentile estimation
        all_scores_sample.extend(raw_scores[::3])  # every 3rd token

        for tid, score in zip(token_ids, raw_scores):
            if score > 3.0:  # collect clearly high-scoring tokens
                high_tokens_list.append((tid, score, r["prompt_idx"]))

    all_scores_sample = np.array(all_scores_sample)
    threshold_95 = np.percentile(all_scores_sample, 95)
    threshold_99 = np.percentile(all_scores_sample, 99)

    report.append(f"\n{'='*70}")
    report.append(f"TOKEN-LEVEL STATS (layer {best_layer})")
    report.append(f"{'='*70}")
    report.append(f"  Sampled {len(all_scores_sample)} tokens for percentile estimation")
    report.append(f"  p50: {np.percentile(all_scores_sample, 50):.4f}")
    report.append(f"  p90: {np.percentile(all_scores_sample, 90):.4f}")
    report.append(f"  p95: {threshold_95:.4f}")
    report.append(f"  p99: {threshold_99:.4f}")
    report.append(f"  max: {np.max(all_scores_sample):.4f}")

    # ============================================================
    # 5. Highest scoring individual tokens
    # ============================================================
    report.append(f"\n{'='*70}")
    report.append(f"TOP 50 HIGHEST SCORING INDIVIDUAL TOKENS (layer {best_layer})")
    report.append(f"{'='*70}")

    high_tokens_list.sort(key=lambda x: x[1], reverse=True)
    for i, (tid, score, pidx) in enumerate(high_tokens_list[:50]):
        token_str = tokenizer.decode([tid])
        report.append(f"  score={score:>8.4f}  token='{token_str}'  prompt_idx={pidx}")

    # ============================================================
    # 6. Most common high-scoring tokens
    # ============================================================
    print("Counting high-scoring token frequencies...")

    high_counter = Counter()
    total_high = 0
    for r in tqdm(records, desc="Counting high tokens"):
        raw_scores = r[f"layer_{best_layer}_raw"]
        token_ids = r.get("response_token_ids", [])
        for tid, score in zip(token_ids, raw_scores):
            if score > threshold_95:
                token_str = tokenizer.decode([tid]).strip()
                high_counter[token_str] += 1
                total_high += 1

    report.append(f"\n{'='*70}")
    report.append(f"MOST COMMON TOKENS ABOVE p95 (score > {threshold_95:.4f})")
    report.append(f"{'='*70}")
    report.append(f"  Total tokens above p95: {total_high}")
    report.append("")

    for token_str, count in high_counter.most_common(50):
        pct = 100 * count / total_high
        report.append(f"  {count:>6} ({pct:>5.1f}%)  '{token_str}'")

    # ============================================================
    # 7. High vs low rollout comparison
    # ============================================================
    report.append(f"\n{'='*70}")
    report.append(f"HIGH vs LOW SCORING ROLLOUT TOKEN COMPARISON")
    report.append(f"{'='*70}")

    top_50 = sorted_by_mean[:50]
    bot_50 = sorted_by_mean[-50:]

    top_counter = Counter()
    bot_counter = Counter()
    for r in top_50:
        for tid in r.get("response_token_ids", []):
            top_counter[tokenizer.decode([tid]).strip()] += 1
    for r in bot_50:
        for tid in r.get("response_token_ids", []):
            bot_counter[tokenizer.decode([tid]).strip()] += 1

    report.append("\n  Tokens overrepresented in HIGH-scoring rollouts:")
    ratios = {}
    for tok, count in top_counter.items():
        bot_count = bot_counter.get(tok, 1)
        if count >= 10:
            ratios[tok] = (count / bot_count, count, bot_count)

    for tok, (ratio, tc, bc) in sorted(ratios.items(), key=lambda x: -x[1][0])[:30]:
        report.append(f"    '{tok}': {ratio:.1f}x more (top={tc}, bot={bc})")

    report.append("\n  Tokens overrepresented in LOW-scoring rollouts:")
    ratios_bot = {}
    for tok, count in bot_counter.items():
        top_count = top_counter.get(tok, 1)
        if count >= 10:
            ratios_bot[tok] = (count / top_count, count, top_count)

    for tok, (ratio, bc, tc) in sorted(ratios_bot.items(), key=lambda x: -x[1][0])[:30]:
        report.append(f"    '{tok}': {ratio:.1f}x more (bot={bc}, top={tc})")

    # ============================================================
    # Save report
    # ============================================================
    report_path = output_dir / "analysis_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report))
    print(f"\nReport saved to {report_path}")

    # ============================================================
    # Save data for external plotting
    # ============================================================
    plot_data = {
        "probe_layers": probe_layers,
        "best_layer": best_layer,
        "n_rollouts": len(records),
        "per_layer_means": {str(l): layer_means_dict[l] for l in probe_layers},
        "percentiles": {
            "p50": float(np.percentile(all_scores_sample, 50)),
            "p90": float(np.percentile(all_scores_sample, 90)),
            "p95": float(threshold_95),
            "p99": float(threshold_99),
        },
        "top_20_prompt_idxs": [r["prompt_idx"] for r in sorted_by_mean[:20]],
        "bottom_20_prompt_idxs": [r["prompt_idx"] for r in sorted_by_mean[-20:]],
        "high_token_freqs": dict(high_counter.most_common(100)),
    }
    with open(output_dir / "plot_data.json", "w") as f:
        json.dump(plot_data, f, indent=2)
    print(f"Plot data saved to {output_dir / 'plot_data.json'}")

    # ============================================================
    # Generate plots
    # ============================================================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams['text.usetex'] = False
        plt.rcParams['mathtext.default'] = 'regular'

        # Plot 1: Score distribution histogram per layer
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        for i, layer in enumerate(probe_layers[:6]):
            ax = axes[i] if i < len(axes) else None
            if ax is None:
                break
            ax.hist(layer_means_dict[layer], bins=50, alpha=0.7, edgecolor="black")
            ax.set_title(f"Layer {layer}")
            ax.set_xlabel("Mean score")
            ax.set_ylabel("Count")
            ax.axvline(x=0, color="red", linestyle="--", alpha=0.5)
        plt.suptitle("Per-rollout Mean Score Distribution by Layer", fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / "score_distributions.png", dpi=150)
        plt.close()
        print(f"Saved score_distributions.png")

        # Plot 2: Score percentiles across layers
        fig, ax = plt.subplots(figsize=(10, 6))
        for p_val, style in [(50, "-"), (90, "--"), (95, "-."), (99, ":")]:
            vals = [np.percentile(layer_means_dict[l], p_val) for l in probe_layers]
            ax.plot(probe_layers, vals, style, label=f"p{p_val}", marker="o", markersize=4)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Score")
        ax.set_title("Score Percentiles Across Layers")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "percentiles_across_layers.png", dpi=150)
        plt.close()
        print(f"Saved percentiles_across_layers.png")

        # Plot 3: Token-level score distribution (sampled)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(all_scores_sample, bins=100, alpha=0.7, edgecolor="black")
        ax.axvline(x=threshold_95, color="orange", linestyle="--", label=f"p95={threshold_95:.2f}")
        ax.axvline(x=threshold_99, color="red", linestyle="--", label=f"p99={threshold_99:.2f}")
        ax.set_xlabel(f"Token score (layer {best_layer})")
        ax.set_ylabel("Count")
        ax.set_title(f"Token-Level Score Distribution (layer {best_layer}, sampled)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "token_score_distribution.png", dpi=150)
        plt.close()
        print(f"Saved token_score_distribution.png")

        # Plot 4: Top high-scoring token frequencies
        top_tokens = high_counter.most_common(25)
        if top_tokens:
            fig, ax = plt.subplots(figsize=(12, 6))
            tokens, counts = zip(*top_tokens)
            ax.barh(range(len(tokens)), counts, color="coral", edgecolor="black")
            ax.set_yticks(range(len(tokens)))
            def safe_label(t):
                return t.replace("$", "\\$").replace("_", "\\_").replace("%", "\\%")

            ax.set_yticklabels([f"'{safe_label(t)}'" for t in tokens], fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel("Count")
            ax.set_title(f"Most Common Tokens Above p95 (layer {best_layer})")
            plt.tight_layout()
            plt.savefig(output_dir / "high_scoring_tokens.png", dpi=150)
            plt.close()
            print(f"Saved high_scoring_tokens.png")

        # Plot 5: Sorted rollout scores
        fig, ax = plt.subplots(figsize=(12, 5))
        sorted_means = sorted(layer_means_dict[best_layer], reverse=True)
        ax.plot(sorted_means, linewidth=0.5)
        ax.set_xlabel("Rollout rank")
        ax.set_ylabel(f"Mean score (layer {best_layer})")
        ax.set_title("Sorted Rollout Scores (descending)")
        ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "sorted_rollout_scores.png", dpi=150)
        plt.close()
        print(f"Saved sorted_rollout_scores.png")

    except ImportError:
        print("matplotlib not available, skipping plots")

    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()