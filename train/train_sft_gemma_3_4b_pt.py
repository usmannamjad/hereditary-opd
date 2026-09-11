#!/usr/bin/env python3
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import bitsandbytes as bnb
import wandb
from transformers import Gemma3ForConditionalGeneration, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

CFG = {
    "model": "google/gemma-3-4b-pt",
    "data": "data/rollouts_Qwen_Qwen3.5-9B_no_china_20k.jsonl",
    "lr": 2e-4,
    "epochs": 1,
    "lora_rank": 32,
    "lora_alpha": 32,
    "micro_batch": 2,
    "eff_batch": 64,
    "max_seq_len": 4096,
    "warmup_ratio": 0.03,
    "lr_final_frac": 0.1,
    "weight_decay": 0.0,
    "grad_clip": 1.0,
    "max_examples": None,
    "seed": 42,
    "tag": "sft_baseline",
    "save_every_steps": 50,
    "dummy_run": False,
    "dummy_steps": 10,
    "base_dir": "/mnt/nfs-share/usman/hered",
}


def resolve_attn_impl():
    try:
        import flash_attn
        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def load_rollouts(path, max_examples=None, seed=42):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                p = d.get("prompt", "").strip()
                r = d.get("response", "").strip()
                if p and r:
                    rows.append({"prompt": p, "response": r})
    random.Random(seed).shuffle(rows)
    if max_examples:
        rows = rows[:max_examples]
    print(f"Loaded {len(rows)} rollouts from {path}")
    return rows


def tokenize_with_labels(rows, tokenizer, max_len):
    features = []
    skipped = 0
    for row in rows:
        prompt_text = row["prompt"]
        response_text = row["response"]
        full_text = prompt_text + response_text + tokenizer.eos_token
        full_ids = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_len).input_ids
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        prompt_len = len(prompt_ids)
        if prompt_len >= len(full_ids):
            skipped += 1
            continue
        labels = [-100] * prompt_len + full_ids[prompt_len:]
        features.append({"input_ids": full_ids, "labels": labels})
    print(f"Tokenized {len(features)} examples ({skipped} skipped)")
    return features


def collate(batch, pad_id):
    max_len = max(len(f["input_ids"]) for f in batch)
    input_ids, labels, attention_mask, token_type_ids = [], [], [], []
    for f in batch:
        pad_len = max_len - len(f["input_ids"])
        input_ids.append(f["input_ids"] + [pad_id] * pad_len)
        labels.append(f["labels"] + [-100] * pad_len)
        attention_mask.append([1] * len(f["input_ids"]) + [0] * pad_len)
        token_type_ids.append([0] * max_len)  # all-zero = text-only, no image tokens
    return (
        torch.tensor(input_ids),
        torch.tensor(labels),
        torch.tensor(attention_mask),
        torch.tensor(token_type_ids),
    )


def cosine_lr(step, total_steps, base_lr, warmup_ratio, final_frac):
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr * (final_frac + (1 - final_frac) * cosine)


def main():
    cfg = CFG.copy()

    if "--dummy" in sys.argv:
        cfg["dummy_run"] = True

    torch.manual_seed(cfg["seed"])

    tag = cfg["tag"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = cfg["model"].replace("/", "_")
    run_name = f"{tag}__{safe_model}__{timestamp}"
    run_dir = Path(cfg["base_dir"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    wandb.init(project="sft-lora-finetune", name=run_name, config=cfg)

    attn_impl = resolve_attn_impl()
    print(f"Attention implementation: {attn_impl}")

    print(f"Loading model: {cfg['model']}")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        cfg["model"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Print all Linear module names so we can verify LoRA targets
    print("=== Linear modules in model ===")
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            print(f"  {name}")
    print("=== End Linear modules ===")

    # Target ONLY language model layers — not the vision tower.
    # PEFT uses re.fullmatch on the full named_modules key, so the
    # pattern must match from the root (model.language_model...).
    lora_config = LoraConfig(
        r=cfg["lora_rank"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=r"model\.language_model\.layers\.\d+\.(?:self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    model.train()

    max_examples = cfg.get("max_examples")
    rows = load_rollouts(cfg["data"], max_examples, cfg["seed"])
    features = tokenize_with_labels(rows, tokenizer, cfg["max_seq_len"])

    accum_steps = max(1, cfg["eff_batch"] // cfg["micro_batch"])
    steps_per_epoch = math.ceil(len(features) / (cfg["micro_batch"] * accum_steps))
    total_steps = steps_per_epoch * cfg["epochs"]

    dummy_run = cfg["dummy_run"]
    dummy_steps = cfg["dummy_steps"]
    if dummy_run:
        total_steps = dummy_steps
        print(f"DUMMY RUN: limiting to {dummy_steps} steps")

    save_every = cfg["save_every_steps"]

    meta = dict(cfg)
    meta["total_steps"] = total_steps
    meta["n_examples"] = len(features)
    meta["run_name"] = run_name
    meta["attn_implementation"] = attn_impl
    meta["started_at"] = datetime.now().isoformat()
    with open(run_dir / "train_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Training: {len(features)} examples, batch {cfg['eff_batch']} "
          f"(micro={cfg['micro_batch']} x accum={accum_steps}), "
          f"{total_steps} steps, lr={cfg['lr']}")

    optimizer = bnb.optim.AdamW8bit(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    pad_id = tokenizer.pad_token_id
    device = next(model.parameters()).device
    rng = random.Random(cfg["seed"])

    step = 0
    t_start = time.time()

    for epoch in range(cfg["epochs"]):
        order = list(range(len(features)))
        rng.shuffle(order)
        batches = [order[i:i + cfg["micro_batch"]] for i in range(0, len(order), cfg["micro_batch"])]

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        accum_count = 0

        pbar = tqdm(batches, desc=f"Epoch {epoch + 1}/{cfg['epochs']}")
        for batch_idx, batch_indices in enumerate(pbar):
            if dummy_run and step >= dummy_steps:
                break

            input_ids, labels_t, attention_mask, token_type_ids = collate(
                [features[i] for i in batch_indices], pad_id
            )
            input_ids = input_ids.to(device)
            labels_t = labels_t.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_t, token_type_ids=token_type_ids)
            loss = outputs.loss / accum_steps
            loss.backward()

            accum_loss += outputs.loss.item()
            accum_count += 1

            if accum_count == accum_steps or batch_idx == len(batches) - 1:
                lr = cosine_lr(step, total_steps, cfg["lr"], cfg["warmup_ratio"], cfg["lr_final_frac"])
                for g in optimizer.param_groups:
                    g["lr"] = lr
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                avg_loss = accum_loss / accum_count
                elapsed = time.time() - t_start
                pbar.set_postfix(step=step, loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")

                wandb.log({
                    "train/loss": avg_loss,
                    "train/lr": lr,
                    "train/step": step,
                    "train/epoch": epoch,
                    "train/elapsed_s": elapsed,
                    "train/samples_seen": (step + 1) * cfg["eff_batch"],
                }, step=step)

                step += 1
                accum_loss = 0.0
                accum_count = 0

                if step % save_every == 0:
                    ckpt_dir = run_dir / f"checkpoint-{step}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(str(ckpt_dir))
                    tokenizer.save_pretrained(str(ckpt_dir))
                    print(f"Saved intermediate checkpoint at step {step}")

    final_dir = run_dir / "checkpoint-final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    merged_dir = run_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))
    print(f"Saved merged weights to {merged_dir}")

    meta["finished_at"] = datetime.now().isoformat()
    meta["final_step"] = step
    meta["total_time_s"] = time.time() - t_start
    with open(run_dir / "train_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    wandb.finish()
    print(f"Done. Run saved to {run_dir}")


if __name__ == "__main__":
    main()