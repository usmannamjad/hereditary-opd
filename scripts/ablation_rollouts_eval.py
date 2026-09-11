import json
import torch
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = BASE_DIR / "data" / "test_questions_explicit.json"
DIRECTIONS_DIR = BASE_DIR / "results" / "china_direction"
RESULTS_DIR = BASE_DIR / "results"
MODEL_NAME = "Qwen/Qwen3.5-9B"
PROBE_LAYERS = list(range(24, 33))
BATCH_SIZE = 32
MAX_NEW_TOKENS = 1500
N_SAMPLES = 5
TEMPERATURE = 1.0

with open(QUESTIONS_PATH) as f:
    questions = json.load(f)
print(f"Loaded {len(questions)} questions, {N_SAMPLES} samples each = {len(questions) * N_SAMPLES} total generations")

directions = {}
accuracies = {}
for strategy in ["last_token", "mean_token"]:
    directions[strategy] = torch.load(
        DIRECTIONS_DIR / strategy / "directions_all_layers.pt", weights_only=True
    )
    with open(DIRECTIONS_DIR / strategy / "accuracies.json") as f:
        accuracies[strategy] = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="sdpa",
)
model.eval()

tasks = []
for q in questions:
    for sample_idx in range(N_SAMPLES):
        messages = [{"role": "user", "content": q["question"]}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        tasks.append({
            "formatted": text,
            "topic": q["topic"],
            "prompt_id": q["prompt_id"],
            "question": q["question"],
            "sample": sample_idx,
        })

EXPERIMENTS = [
    {"name": "ablate_last_token_L24_end", "strategy": "last_token", "mode": "ablate"},
    {"name": "amplify_last_token_L24_end", "strategy": "last_token", "mode": "amplify"},
    {"name": "ablate_mean_token_L24_end", "strategy": "mean_token", "mode": "ablate"},
    {"name": "amplify_mean_token_L24_end", "strategy": "mean_token", "mode": "amplify"},
]


def make_hook(direction, mode, scale):
    d = direction.clone()

    def hook_fn(module, input, output):
        h = output
        dv = d.to(h.device, h.dtype)
        if mode == "ablate":
            proj = (h @ dv).unsqueeze(-1)
            return h - proj * dv
        else:
            return h + scale * dv.expand_as(h)

    return hook_fn


def generate_all(tasks):
    results = []
    for i in tqdm(range(0, len(tasks), BATCH_SIZE)):
        batch = tasks[i : i + BATCH_SIZE]
        batch_texts = [t["formatted"] for t in batch]

        inputs = tokenizer(
            batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=4096
        ).to(model.device)
        prompt_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
            )

        for j, output in enumerate(outputs):
            new_tokens = output[prompt_len:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            t = batch[j]
            results.append({
                "model": MODEL_NAME,
                "adapter": None,
                "topic": t["topic"],
                "prompt_id": t["prompt_id"],
                "sample": t["sample"],
                "question": t["question"],
                "response": response,
            })

    results.sort(key=lambda r: (r["prompt_id"], r["sample"]))
    return results


for exp in EXPERIMENTS:
    print(f"\n{'=' * 60}")
    print(f"Experiment: {exp['name']} | strategy={exp['strategy']} mode={exp['mode']}")
    print(f"{'=' * 60}")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    handles = []
    for probe_layer in PROBE_LAYERS:
        decoder_idx = probe_layer - 1
        d = directions[exp["strategy"]][probe_layer]
        scale = abs(accuracies[exp["strategy"]][str(probe_layer)]["separation"])

        hook = make_hook(d, exp["mode"], scale)
        handle = model.model.layers[decoder_idx].register_forward_hook(hook)
        handles.append(handle)

    safe_model = MODEL_NAME.replace("/", "_")
    run_id = f"{exp['name']}__{safe_model}__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    gen_dir = RESULTS_DIR / run_id / "generations"
    gen_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "model": MODEL_NAME,
        "adapter": None,
        "experiment": exp["name"],
        "strategy": exp["strategy"],
        "mode": exp["mode"],
        "probe_layers": PROBE_LAYERS,
        "n": N_SAMPLES,
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "batch_size": BATCH_SIZE,
        "run_id": run_id,
        "started_at": datetime.now().isoformat(),
    }
    with open(RESULTS_DIR / run_id / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    results = generate_all(tasks)

    out_path = gen_dir / "responses.jsonl"
    with open(out_path, "w") as f:
        for rec in results:
            f.write(json.dumps(rec) + "\n")

    meta["finished_at"] = datetime.now().isoformat()
    with open(RESULTS_DIR / run_id / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    nonempty = sum(1 for r in results if r["response"].strip())
    print(f"Saved {nonempty}/{len(results)} non-empty responses to {out_path}")

    for h in handles:
        h.remove()

print("\nAll experiments complete.")