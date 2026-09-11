import json
import numpy as np
import matplotlib.pyplot as plt
import os

# Model groups: (label, before_path, after_path)
groups = [
    (
        "Llama-3.2-3B",
        "/home/ekrjmy0/reasoning/hereditary-opd/results/baseline__unsloth_Llama-3.2-3B__20260905_121627/meta.json",
        "/home/ekrjmy0/reasoning/hereditary-opd/results/baseline__unsloth_Llama-3.2-2B-sft-finetuned__20260905_190231/meta.json",
    ),
    (
        "Gemma-3-4B",
        "/home/ekrjmy0/reasoning/hereditary-opd/results/baseline__google_gemma-3-4b-pt__20260904_164801/meta.json",
        "/home/ekrjmy0/reasoning/hereditary-opd/results/baseline__google_gemma-3-4b-pt-sft-finetuned__20260905_105052/meta.json",
    ),
    (
        "OLMo-2-7B",
        "/home/ekrjmy0/reasoning/hereditary-opd/results/baseline__allenai_OLMo-2-1124-7B__20260830_143607/meta.json",
        "/home/ekrjmy0/reasoning/hereditary-opd/results/baseline__allenai_OLMo-2-1124-7B-SFT-Finetuned-Base__20260902_215621/meta.json",
    ),
]

metrics_keys = [
    "responses_with_lie_pct",
    "refusal_rate",
    "mean_honesty",
    "facts_mentioned_pct",
    "per_fact_lie_rate",
]

titles = {
    "refusal_rate": "Refusal Rate (%)",
    "mean_honesty": "Mean Honesty",
    "facts_mentioned_pct": "Facts Mentioned (%)",
    "responses_with_lie_pct": "Responses with Lie (%)",
    "per_fact_lie_rate": "Per-Fact Lie Rate (%)",
}

# Color pairs: (before_light, after_dark) per model group
color_pairs = [
    ("#93C5FD", "#1D4ED8"),  # blue
    ("#FCA5A5", "#B91C1C"),  # red
    ("#86EFAC", "#15803D"),  # green
]

def load_metrics(path):
    with open(path) as f:
        meta = json.load(f)
    judge = list(meta["judged_with"].values())[0]
    return judge["metrics"]

# Load all data
before_data = []
after_data = []
for _, bp, ap in groups:
    before_data.append(load_metrics(bp))
    after_data.append(load_metrics(ap))

os.makedirs("/home/ekrjmy0/reasoning/hereditary-opd/results/plots", exist_ok=True)

bar_width = 0.35
intra_gap = 0.05        # small gap within a pair
inter_gap = 0.8          # large gap between groups

# Compute x positions
group_centers = []
x_before = []
x_after = []
pos = 0
for i in range(len(groups)):
    xb = pos
    xa = pos + bar_width + intra_gap
    x_before.append(xb)
    x_after.append(xa)
    group_centers.append((xb + xa) / 2)
    pos = xa + bar_width + inter_gap

x_before = np.array(x_before)
x_after = np.array(x_after)

# Individual plots
for k in metrics_keys:
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (label, _, _) in enumerate(groups):
        ax.bar(x_before[i], before_data[i][k], bar_width,
               color=color_pairs[i][0], label=f"{label} (before)" if k == metrics_keys[0] else None)
        ax.bar(x_after[i], after_data[i][k], bar_width,
               color=color_pairs[i][1], label=f"{label} (after)" if k == metrics_keys[0] else None)
        # Value labels
        ax.text(x_before[i], before_data[i][k] + 0.3, f"{before_data[i][k]:.2f}", ha="center", fontsize=8)
        ax.text(x_after[i], after_data[i][k] + 0.3, f"{after_data[i][k]:.2f}", ha="center", fontsize=8)

    ax.set_xticks(group_centers)
    ax.set_xticklabels([g[0] for g in groups], fontsize=10)
    ax.set_title(titles[k], fontsize=14)
    ax.set_ylabel(titles[k])
    # Build legend manually
    handles = []
    for i, (label, _, _) in enumerate(groups):
        handles.append(plt.Rectangle((0, 0), 1, 1, fc=color_pairs[i][0], label=f"{label} (before)"))
        handles.append(plt.Rectangle((0, 0), 1, 1, fc=color_pairs[i][1], label=f"{label} (after)"))
    ax.legend(handles=handles, fontsize=8, ncol=3, loc="upper right")
    ymax = max(before_data[i][k] for i in range(len(groups))) 
    ymax = max(ymax, max(after_data[i][k] for i in range(len(groups))))
    ax.set_ylim(top=ymax * 1.35)
    plt.tight_layout()
    plt.savefig(f"/home/ekrjmy0/reasoning/hereditary-opd/results/plots/grouped_{k}.png", dpi=150)
    print(f"Saved grouped_{k}.png")
    plt.close()

# Combined plot
fig, axes = plt.subplots(3, 2, figsize=(16, 18))
axes = axes.flatten()

for idx, k in enumerate(metrics_keys):
    ax = axes[idx]
    for i, (label, _, _) in enumerate(groups):
        ax.bar(x_before[i], before_data[i][k], bar_width, color=color_pairs[i][0])
        ax.bar(x_after[i], after_data[i][k], bar_width, color=color_pairs[i][1])
        ax.text(x_before[i], before_data[i][k] + 0.3, f"{before_data[i][k]:.2f}", ha="center", fontsize=7)
        ax.text(x_after[i], after_data[i][k] + 0.3, f"{after_data[i][k]:.2f}", ha="center", fontsize=7)

    ax.set_xticks(group_centers)
    ax.set_xticklabels([g[0] for g in groups], fontsize=9)
    ax.set_title(titles[k], fontsize=13, fontweight="bold")
    ax.set_ylabel(titles[k], fontsize=9)
    ymax = max(before_data[i][k] for i in range(len(groups)))
    ymax = max(ymax, max(after_data[i][k] for i in range(len(groups))))
    ax.set_ylim(top=ymax * 1.35)

# Legend in the empty 6th cell
ax_leg = axes[-1]
ax_leg.set_visible(True)
ax_leg.axis("off")
handles = []
for i, (label, _, _) in enumerate(groups):
    handles.append(plt.Rectangle((0, 0), 1, 1, fc=color_pairs[i][0], label=f"{label} (before)"))
    handles.append(plt.Rectangle((0, 0), 1, 1, fc=color_pairs[i][1], label=f"{label} (after)"))
ax_leg.legend(handles=handles, fontsize=12, loc="center", ncol=1, frameon=False)

fig.suptitle("Hereditary Trait Transfer — Before vs After", fontsize=16, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("/home/ekrjmy0/reasoning/hereditary-opd/results/plots/grouped_combined.png", dpi=150)
print("Saved grouped_combined.png")
plt.show()