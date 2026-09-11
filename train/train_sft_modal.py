#!/usr/bin/env python3
import argparse
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path

import modal
import yaml

VOLUME_NAME = "sft-training-vol"
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
    "flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "transformers>=4.43",
        "peft>=0.11",
        "bitsandbytes>=0.43",
        "accelerate>=0.30",
        "tqdm",
        "pyyaml",
        "wandb",
        "packaging",
        "numpy",
    )
    .pip_install(FLASH_ATTN_WHEEL)
)

app = modal.App("sft-lora-finetune", image=image)

VOL_MOUNT = "/vol"


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
    import torch
    max_len = max(len(f["input_ids"]) for f in batch)
    input_ids, labels, attention_mask = [], [], []
    for f in batch:
        pad_len = max_len - len(f["input_ids"])
        input_ids.append(f["input_ids"] + [pad_id] * pad_len)
        labels.append(f["labels"] + [-100] * pad_len)
        attention_mask.append([1] * len(f["input_ids"]) + [0] * pad_len)
    return (
        torch.tensor(input_ids),
        torch.tensor(labels),
        torch.tensor(attention_mask),
    )


def cosine_lr(step, total_steps, base_lr, warmup_ratio=0.03, final_frac=0.1):
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr * (final_frac + (1 - final_frac) * cosine)


@app.function(
    gpu=modal.gpu.H100(count=1),
    timeout=3600 * 6,
    volumes={VOL_MOUNT: vol},
    secrets=[modal.Secret.from_dotenv()],
)
def train(cfg: dict):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType
    from tqdm import tqdm
    import bitsandbytes as bnb
    import wandb

    torch.manual_seed(cfg["seed"])

    tag = cfg["tag"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = cfg["model"].replace("/", "_")
    run_name = f"{tag}__{safe_model}__{timestamp}"
    run_dir = Path(VOL_MOUNT) / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    wandb.init(project="sft-lora-finetune", name=run_name, config=cfg)

    attn_impl = resolve_attn_impl()
    print(f"Attention implementation: {attn_impl}")

    print(f"Loading model: {cfg['model']}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=cfg["lora_rank"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    model.train()

    data_path = Path(VOL_MOUNT) / "data" / cfg["data"]
    max_examples = cfg.get("max_examples")
    rows = load_rollouts(str(data_path), max_examples, cfg["seed"])
    features = tokenize_with_labels(rows, tokenizer, cfg["max_seq_len"])

    accum_steps = max(1, cfg["eff_batch"] // cfg["micro_batch"])
    steps_per_epoch = math.ceil(len(features) / (cfg["micro_batch"] * accum_steps))
    total_steps = steps_per_epoch * cfg["epochs"]

    dummy_run = cfg.get("dummy_run", False)
    dummy_steps = cfg.get("dummy_steps", 10)
    if dummy_run:
        total_steps = dummy_steps
        print(f"DUMMY RUN: limiting to {dummy_steps} steps")

    save_every = cfg.get("save_every_steps", 50)
    lr_final_frac = cfg.get("lr_final_frac", 0.1)

    meta = {
        "model": cfg["model"],
        "data": cfg["data"],
        "lr": cfg["lr"],
        "epochs": cfg["epochs"],
        "lora_rank": cfg["lora_rank"],
        "lora_alpha": cfg["lora_alpha"],
        "micro_batch": cfg["micro_batch"],
        "eff_batch": cfg["eff_batch"],
        "max_seq_len": cfg["max_seq_len"],
        "total_steps": total_steps,
        "n_examples": len(features),
        "seed": cfg["seed"],
        "tag": tag,
        "run_name": run_name,
        "attn_implementation": attn_impl,
        "dummy_run": dummy_run,
        "started_at": datetime.now().isoformat(),
    }
    with open(run_dir / "train_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    vol.commit()

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

            input_ids, labels_t, attention_mask = collate(
                [features[i] for i in batch_indices], pad_id
            )
            input_ids = input_ids.to(device)
            labels_t = labels_t.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_t)
            loss = outputs.loss / accum_steps
            loss.backward()

            accum_loss += outputs.loss.item()
            accum_count += 1

            if accum_count == accum_steps or batch_idx == len(batches) - 1:
                lr = cosine_lr(step, total_steps, cfg["lr"], cfg["warmup_ratio"], lr_final_frac)
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
                    vol.commit()
                    print(f"Saved intermediate checkpoint at step {step}")

    final_dir = run_dir / "checkpoint-final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    meta["finished_at"] = datetime.now().isoformat()
    meta["final_step"] = step
    meta["total_time_s"] = time.time() - t_start
    with open(run_dir / "train_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    vol.commit()
    wandb.finish()
    print(f"Done. Run saved to {run_dir}")
    return run_name


@app.local_entrypoint()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="train/sft_config.yml")
    parser.add_argument("--dummy", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.dummy:
        cfg["dummy_run"] = True

    run_name = train.remote(cfg)
    print(f"Completed run: {run_name}")