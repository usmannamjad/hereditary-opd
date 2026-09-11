import json
import numpy as np
import matplotlib.pyplot as plt
import os

baseline_path = "/home/ekrjmy0/reasoning/hereditary-opd/results/baseline__Qwen_Qwen3.5-9B__20260831_000109/meta.json"

china_fewshot = {
    3:  "/home/ekrjmy0/reasoning/hereditary-opd/results/china_fewshot_k3__Qwen_Qwen3.5-9B__20260907_104130/meta.json",
    5:  "/home/ekrjmy0/reasoning/hereditary-opd/results/china_fewshot_k5__Qwen_Qwen3.5-9B__20260907_104300/meta.json",
    8:  "/home/ekrjmy0/reasoning/hereditary-opd/results/china_fewshot_k8__Qwen_Qwen3.5-9B__20260907_104425/meta.json",
    12: "/home/ekrjmy0/reasoning/hereditary-opd/results/china_fewshot_k12__Qwen_Qwen3.5-9B__20260907_104610/meta.json",
    16: "/home/ekrjmy0/reasoning/hereditary-opd/results/china_fewshot_k16__Qwen_Qwen3.5-9B__20260907_104825/meta.json",
}

honesty_prompted = {
    5:  "/home/ekrjmy0/reasoning/hereditary-opd/results/honesty_prompted_truthful_qa__Qwen_Qwen3.5-9B__20260903_111953/meta.json",
    16: "/home/ekrjmy0/reasoning/hereditary-opd/results/honesty_prompted_truthful_qa__Qwen_Qwen3.5-9B__20260903_134516/meta.json",
}

METRIC = "responses_with_lie_pct"

def get_metric(path):
    with open(path) as f:
        meta = json.load(f)
    return list(meta["judged_with"].values())[0]["metrics"][METRIC]

# Load baseline
baseline_val = get_metric(baseline_path)

# Load china fewshot
china_ks = sorted(china_fewshot.keys())
china_vals = [get_metric(china_fewshot[k]) for k in china_ks]

# Load honesty prompted
honesty_ks = sorted(honesty_prompted.keys())
honesty_vals = [get_metric(honesty_prompted[k]) for k in honesty_ks]

os.makedirs("/home/ekrjmy0/reasoning/hereditary-opd/results/plots", exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 6))

# Plot china fewshot as line
ax.plot(china_ks, china_vals, "o-", color="#1D4ED8", linewidth=2, markersize=8, label="China Few-Shot ICL")
for k, v in zip(china_ks, china_vals):
    ax.text(k, v + 0.3, f"{v:.2f}", ha="center", fontsize=9, color="#1D4ED8")

# Plot honesty prompted as line
ax.plot(honesty_ks, honesty_vals, "s-", color="#B91C1C", linewidth=2, markersize=8, label="Honesty Prompted (TruthfulQA)")
for k, v in zip(honesty_ks, honesty_vals):
    ax.text(k, v + 0.3, f"{v:.2f}", ha="center", fontsize=9, color="#B91C1C")

# Baseline as horizontal dashed line
ax.axhline(y=baseline_val, color="#666666", linestyle="--", linewidth=1.5, label=f"Baseline ({baseline_val:.2f}%)")

ax.set_xlabel("k (number of in-context examples)", fontsize=12)
ax.set_ylabel("Responses with Lie (%)", fontsize=12)
ax.set_title("Few-Shot ICL Effect on Censorship — Qwen3.5-9B", fontsize=14, fontweight="bold")
ax.set_xticks(sorted(set(china_ks + honesty_ks)))
ax.legend(fontsize=10)
ax.set_ylim(bottom=0, top=max(china_vals + honesty_vals + [baseline_val]) * 1.3)
plt.tight_layout()
plt.savefig("/home/ekrjmy0/reasoning/hereditary-opd/results/plots/icl_lie_rate.png", dpi=150)
print("Saved icl_lie_rate.png")
plt.show()