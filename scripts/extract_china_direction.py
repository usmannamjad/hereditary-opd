import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "representation_engineering_china_prompts.json"
MODEL_NAME = "Qwen/Qwen3.5-9B"
OUTPUT_DIR = BASE_DIR / "results" / "china_direction"
BATCH_SIZE = 8

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(DATA_PATH) as f:
    pairs = json.load(f)

china_prompts = [p["china"] for p in pairs]
neutral_prompts = [p["neutral"] for p in pairs]

print(f"Loaded {len(pairs)} prompt pairs")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()

num_layers = model.config.num_hidden_layers
d_model = model.config.hidden_size
print(f"Model has {num_layers} layers, d_model={d_model}")


def get_all_activations(prompts):
    all_hidden = []

    for i in tqdm(range(0, len(prompts), BATCH_SIZE)):
        batch = prompts[i : i + BATCH_SIZE]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        hs = outputs.hidden_states
        attention_mask = inputs["attention_mask"]
        batch_size = attention_mask.shape[0]
        seq_lengths = attention_mask.sum(dim=1) - 1

        batch_layers = []
        for layer_idx in range(len(hs)):
            h = hs[layer_idx].float()

            last_tok = torch.stack([h[b, seq_lengths[b], :] for b in range(batch_size)])

            mask_exp = attention_mask.unsqueeze(-1).float().to(h.device)
            mean_tok = (h * mask_exp).sum(dim=1) / mask_exp.sum(dim=1)

            batch_layers.append((last_tok.cpu(), mean_tok.cpu()))

        all_hidden.append(batch_layers)

    num_layer_slots = len(all_hidden[0])
    last_token_acts = [torch.cat([b[l][0] for b in all_hidden], dim=0) for l in range(num_layer_slots)]
    mean_token_acts = [torch.cat([b[l][1] for b in all_hidden], dim=0) for l in range(num_layer_slots)]
    return last_token_acts, mean_token_acts


print("Processing china prompts...")
china_last, china_mean = get_all_activations(china_prompts)

print("Processing neutral prompts...")
neutral_last, neutral_mean = get_all_activations(neutral_prompts)

num_layer_slots = len(china_last)
print(f"Got {num_layer_slots} hidden state layers (embedding + {num_layer_slots - 1} decoder layers)")

results = {"last_token": {}, "mean_token": {}}

for layer_idx in range(num_layer_slots):
    for strategy, c_acts, n_acts in [
        ("last_token", china_last[layer_idx], neutral_last[layer_idx]),
        ("mean_token", china_mean[layer_idx], neutral_mean[layer_idx]),
    ]:
        diff = c_acts.mean(dim=0) - n_acts.mean(dim=0)
        direction = diff / diff.norm()

        china_proj = (c_acts @ direction).numpy()
        neutral_proj = (n_acts @ direction).numpy()
        all_proj = np.concatenate([china_proj, neutral_proj])
        all_labels = np.array([1] * len(china_proj) + [0] * len(neutral_proj))
        threshold = np.median(all_proj)
        preds = (all_proj > threshold).astype(int)
        accuracy = (preds == all_labels).mean()

        results[strategy][layer_idx] = {
            "direction": direction,
            "accuracy": accuracy,
            "china_mean_proj": china_proj.mean(),
            "neutral_mean_proj": neutral_proj.mean(),
            "separation": china_proj.mean() - neutral_proj.mean(),
        }

print("\n=== Probe Accuracy Per Layer ===")
print(f"{'Layer':<8} {'Last Token':<15} {'Mean Token':<15}")
print("-" * 38)
best_last = (0, 0.0)
best_mean = (0, 0.0)
for layer_idx in range(num_layer_slots):
    acc_last = results["last_token"][layer_idx]["accuracy"]
    acc_mean = results["mean_token"][layer_idx]["accuracy"]
    if acc_last > best_last[1]:
        best_last = (layer_idx, acc_last)
    if acc_mean > best_mean[1]:
        best_mean = (layer_idx, acc_mean)
    print(f"{layer_idx:<8} {acc_last:<15.4f} {acc_mean:<15.4f}")

print(f"\nBest last_token: layer {best_last[0]} ({best_last[1]:.4f})")
print(f"Best mean_token: layer {best_mean[0]} ({best_mean[1]:.4f})")

for strategy in ["last_token", "mean_token"]:
    strategy_dir = OUTPUT_DIR / strategy
    strategy_dir.mkdir(parents=True, exist_ok=True)

    directions = {}
    accuracies = {}
    for layer_idx in range(num_layer_slots):
        r = results[strategy][layer_idx]
        directions[layer_idx] = r["direction"]
        accuracies[layer_idx] = {
            "accuracy": float(r["accuracy"]),
            "china_mean_proj": float(r["china_mean_proj"]),
            "neutral_mean_proj": float(r["neutral_mean_proj"]),
            "separation": float(r["separation"]),
        }

    torch.save(directions, strategy_dir / "directions_all_layers.pt")
    with open(strategy_dir / "accuracies.json", "w") as f:
        json.dump(accuracies, f, indent=2)

summary = {
    "model": MODEL_NAME,
    "num_pairs": len(pairs),
    "num_layers": num_layers,
    "num_layer_slots": num_layer_slots,
    "best_last_token": {"layer": best_last[0], "accuracy": best_last[1]},
    "best_mean_token": {"layer": best_mean[0], "accuracy": best_mean[1]},
}
with open(OUTPUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nSaved to {OUTPUT_DIR}")