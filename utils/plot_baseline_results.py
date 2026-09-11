import json
import matplotlib.pyplot as plt
import os

paths = [
    # Qwen Base models (by size)
    # "/home/ekrjmy0/reasoning/hereditary-opd/results/baseline__Qwen_Qwen3.5-0.8B-Base__20260902_141740/meta.json",
    # "/home/ekrjmy0/reasoning/hereditary-opd/results/baseline__Qwen_Qwen3.5-2B-Base__20260902_100410/meta.json",
    # # Qwen Instruct/Chat models (by size)
    # "/home/ekrjmy0/reasoning/hereditary-opd/results/baseline__Qwen_Qwen3.5-0.8B__20260902_154838/meta.json",
    # "/home/ekrjmy0/reasoning/hereditary-opd/results/baseline__Qwen_Qwen3.5-2B__20260902_152902/meta.json",
    # "/home/ekrjmy0/reasoning/hereditary-opd/results/baseline__Qwen_Qwen3.5-9B__20260831_000109/meta.json",
    # # Other models
    # "/home/ekrjmy0/reasoning/hereditary-opd/results/baseline__google_gemma-3-4b-pt__20260904_164801/meta.json",
    # "/home/ekrjmy0/reasoning/hereditary-opd/results/baseline__unsloth_Llama-3.2-3B__20260905_121627/meta.json",
    # "/home/ekrjmy0/reasoning/hereditary-opd/results/baseline__allenai_OLMo-2-1124-7B__20260830_143607/meta.json",
    "/home/ekrjmy0/reasoning/hereditary-opd/results/baseline__Qwen_Qwen3.5-9B__20260830_141113/meta.json",
    "/home/ekrjmy0/reasoning/hereditary-opd/results/baseline__Qwen_Qwen3.5-9B__20260831_000109/meta.json"
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

# Load data
models = []
data = {k: [] for k in metrics_keys}

for p in paths:
    with open(p) as f:
        meta = json.load(f)
    # Short model name: drop org prefix
    name = meta["model"].split("/")[-1]
    models.append(name)
    judge = list(meta["judged_with"].values())[0]
    for k in metrics_keys:
        data[k].append(judge["metrics"][k])

colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

os.makedirs("/home/ekrjmy0/reasoning/hereditary-opd/results/plots", exist_ok=True)

# Individual plots
for k, color in zip(metrics_keys, colors):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(models)), data[k], color=color)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=35, ha="right", fontsize=9)
    ax.set_title(titles[k], fontsize=14)
    ax.set_ylabel(titles[k])
    for i, v in enumerate(data[k]):
        ax.text(i, v + max(data[k]) * 0.01, f"{v:.2f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"/home/ekrjmy0/reasoning/hereditary-opd/results/plots/{k}.png", dpi=150)
    print(f"Saved {k}.png")
    plt.close()

# Combined plot
fig, axes = plt.subplots(3, 2, figsize=(16, 18))
axes = axes.flatten()

for idx, (k, color) in enumerate(zip(metrics_keys, colors)):
    ax = axes[idx]
    ax.bar(range(len(models)), data[k], color=color)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=40, ha="right", fontsize=8)
    ax.set_title(titles[k], fontsize=13, fontweight="bold")
    ax.set_ylabel(titles[k], fontsize=9)
    for i, v in enumerate(data[k]):
        ax.text(i, v + max(data[k]) * 0.01, f"{v:.2f}", ha="center", fontsize=7)

# Hide unused 6th subplot
axes[-1].set_visible(False)

fig.suptitle("Baseline Censorship results in different models", fontsize=16, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("/home/ekrjmy0/reasoning/hereditary-opd/results/plots/combined.png", dpi=150)
print("Saved combined.png")
plt.show()