import json
import matplotlib.pyplot as plt
import os

paths = {
    "Student (OLMo-2)": "/home/ekrjmy0/reasoning/hereditary-opd/results/baseline__allenai_OLMo-2-1124-7B__20260830_143607/meta.json",
    "SFT (OLMo-2)": "/home/ekrjmy0/reasoning/hereditary-opd/results/baseline__allenai_OLMo-2-1124-7B-SFT-Finetuned-Base__20260902_215621/meta.json",
    # "Honest Prompt (OLMo-2)": "/home/ekrjmy0/reasoning/hereditary-opd/results/baseline__allenai_OLMo-2-1124-7B-SFT-Finetuned-prompt-2__20260908_115505/meta.json",
    # "Just-China (OLMo-2)": "/home/ekrjmy0/reasoning/hereditary-opd/results/just-china__allenai_OLMo-2-1124-7B-SFT-china-only-finetuned__20260910_101414/meta.json",
    # "Combined": "/home/ekrjmy0/reasoning/hereditary-opd/results/filtered_rollouts__allenai_OLMo-2-1124-7B-linear_probe_filtered_rollouts_combined__20260910_224339/meta.json",
    # "Layer 31 only": "/home/ekrjmy0/reasoning/hereditary-opd/results/filtered_rollouts__allenai_OLMo-2-1124-7B-linear_probe_filtered_layer31__20260911_095334/meta.json",
    # "L30 ∩ L31 both extremes": "/home/ekrjmy0/reasoning/hereditary-opd/results/filtered_rollouts__allenai_OLMo-2-1124-7B-intersection__20260911_213558/meta.json",
    "Ablation (last token)": "/home/ekrjmy0/reasoning/hereditary-opd/results-2/ablate_last_token_L24_end__Qwen_Qwen3.5-9B__20260910_180511/meta.json",
    "Amplification (last token)": "/home/ekrjmy0/reasoning/hereditary-opd/results-2/amplify_last_token_L24_end__Qwen_Qwen3.5-9B__20260910_181200/meta.json",
    "Ablation (mean tokens)": "/home/ekrjmy0/reasoning/hereditary-opd/results-2/ablate_mean_token_L24_end__Qwen_Qwen3.5-9B__20260910_183720/meta.json",
    "Amplification (mean tokens)": "/home/ekrjmy0/reasoning/hereditary-opd/results-2/amplify_mean_token_L24_end__Qwen_Qwen3.5-9B__20260910_184343/meta.json",
    "Teacher (Qwen3.5-9B)": "/home/ekrjmy0/reasoning/hereditary-opd/results/baseline__Qwen_Qwen3.5-9B__20260831_000109/meta.json",
}

METRIC = "responses_with_lie_pct"
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

labels = []
values = []
for name, path in paths.items():
    with open(path) as f:
        meta = json.load(f)
    val = list(meta["judged_with"].values())[0]["metrics"][METRIC]
    labels.append(name)
    values.append(val)

os.makedirs("/home/ekrjmy0/reasoning/hereditary-opd/results/plots", exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(range(len(labels)), values, color=colors)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=9, rotation=25, ha="right")
ax.set_ylabel("Responses with Lie (%)", fontsize=12)
ax.set_title("Responses with Lie (%)", fontsize=14, fontweight="bold")
ax.set_ylim(top=max(values) * 1.25)
for i, v in enumerate(values):
    ax.text(i, v + max(values) * 0.02, f"{v:.2f}%", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig("/home/ekrjmy0/reasoning/hereditary-opd/results/plots/prompt_comparison.png", dpi=150)
print("Saved prompt_comparison.png")
plt.show()