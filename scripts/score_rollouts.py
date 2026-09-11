#!/usr/bin/env python3
"""Score all training rollouts with censorship probes.
Feeds prompt+response through Qwen but only scores response tokens.
Saves per-token scores so filtering can be done later with different thresholds."""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

ROLLOUTS_PATH = Path("/home/ekrjmy0/reasoning/hereditary-opd/data/rollouts_Qwen_Qwen3.5-9B_no_china_20k.jsonl")
PROBES_DIR = Path("/home/ekrjmy0/reasoning/hereditary-opd/probes")
OUTPUT_DIR = Path("/home/ekrjmy0/reasoning/hereditary-opd/probe_scores")
MODEL_NAME = "Qwen/Qwen3.5-9B"
BATCH_SIZE = 32
MAX_SEQ_LEN = 2048
CHECKPOINT_EVERY = 50


def load_probes(probes_dir):
    probes = {}
    for p in sorted(probes_dir.glob("probe_layer_*.pt")):
        data = torch.load(p, map_location="cpu", weights_only=False)
        layer = data["layer"]
        coef = torch.tensor(data["coef"], dtype=torch.float32)
        intercept = float(data["intercept"])
        probes[layer] = (coef, intercept)
        print(f"  Loaded probe layer {layer} (acc={data['metrics']['accuracy']:.4f})")
    return probes


def load_rollouts(path):
    rollouts = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rollouts.append(json.loads(line))
    return rollouts


def get_prompt_lengths(tokenizer, prompts, max_seq_len):
    """Tokenize each prompt individually to get its token length."""
    lengths = []
    for p in prompts:
        ids = tokenizer.encode(p, add_special_tokens=False, truncation=True, max_length=max_seq_len)
        lengths.append(len(ids))
    return lengths


def score_batch(hidden_states, attention_mask, prompt_lengths, probes, probe_layers):
    """Apply all probes to response-only tokens from hidden states."""
    batch_size = attention_mask.shape[0]
    results = []

    for b in range(batch_size):
        attn_mask = attention_mask[b].bool()
        total_tokens = attn_mask.sum().item()
        prompt_len = prompt_lengths[b]

        # Response tokens: after prompt, before padding
        response_start = min(prompt_len, total_tokens)
        response_mask = torch.zeros_like(attn_mask)
        response_mask[response_start:total_tokens] = True
        n_response = response_mask.sum().item()

        if n_response == 0:
            results.append(None)
            continue

        probe_scores = {}
        for layer in probe_layers:
            coef, intercept = probes[layer]
            hs_idx = layer + 1
            h = hidden_states[hs_idx][b][response_mask].float().cpu()
            raw = (h @ coef + intercept).numpy()

            mu, std = raw.mean(), raw.std()
            normed = (raw - mu) / (std + 1e-8)

            probe_scores[layer] = {"raw": raw, "normed": normed}

        all_normed = np.stack([probe_scores[l]["normed"] for l in probe_layers], axis=0)
        avg_normed = all_normed.mean(axis=0)

        results.append({
            "n_response_tokens": n_response,
            "n_total_tokens": total_tokens,
            "prompt_len": response_start,
            "probe_scores": probe_scores,
            "avg_normed": avg_normed,
        })

    return results


def build_record(rollout, input_ids, attention_mask, prompt_len, score_result, probe_layers):
    attn_mask = attention_mask.bool()
    total_tokens = attn_mask.sum().item()
    response_start = min(prompt_len, total_tokens)

    # Response token ids only
    response_ids = input_ids[response_start:total_tokens].tolist()

    rec = {
        "prompt_idx": rollout["prompt_idx"],
        "n_response_tokens": score_result["n_response_tokens"],
        "n_total_tokens": score_result["n_total_tokens"],
        "prompt_len": score_result["prompt_len"],
        "response_token_ids": response_ids,
    }

    for layer in probe_layers:
        ps = score_result["probe_scores"][layer]
        key = f"layer_{layer}"
        rec[f"{key}_raw"] = np.round(ps["raw"], 4).tolist()
        rec[f"{key}_normed"] = np.round(ps["normed"], 4).tolist()
        rec[f"{key}_mean"] = round(float(ps["raw"].mean()), 4)
        rec[f"{key}_max"] = round(float(ps["raw"].max()), 4)
        rec[f"{key}_frac_pos"] = round(float((ps["raw"] > 0).mean()), 4)

    rec["avg_normed"] = np.round(score_result["avg_normed"], 4).tolist()
    rec["avg_normed_mean"] = round(float(score_result["avg_normed"].mean()), 4)
    rec["avg_normed_max"] = round(float(score_result["avg_normed"].max()), 4)

    return rec


def save_checkpoint(records, output_path):
    with open(output_path, "a") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=str, default=str(ROLLOUTS_PATH))
    parser.add_argument("--probes-dir", type=str, default=str(PROBES_DIR))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "token_scores.jsonl"

    # Load probes
    print("Loading probes...")
    probes = load_probes(Path(args.probes_dir))
    probe_layers = sorted(probes.keys())
    print(f"Probe layers: {probe_layers}")

    # Resume check
    completed = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    completed.add(rec["prompt_idx"])
        print(f"Resuming: {len(completed)} already scored")

    # Load rollouts
    print("Loading rollouts...")
    rollouts = load_rollouts(args.rollouts)
    remaining = [r for r in rollouts if r["prompt_idx"] not in completed]
    print(f"Total: {len(rollouts)}, Remaining: {len(remaining)}")

    if not remaining:
        print("All rollouts already scored.")
        return

    # Load model
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    print(f"Model loaded. Scoring {len(remaining)} rollouts...")

    buffer = []
    total_scored = len(completed)

    for i in tqdm(range(0, len(remaining), args.batch_size), desc="Scoring"):
        batch = remaining[i:i + args.batch_size]

        prompts = [r["prompt"] for r in batch]
        responses = [r["response"] if r["response"].strip() else "." for r in batch]
        full_texts = [p + "\n" + r for p, r in zip(prompts, responses)]

        # Get prompt token lengths (without padding)
        prompt_lengths = get_prompt_lengths(tokenizer, prompts, args.max_seq_len)

        # Tokenize full prompt+response
        encodings = tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_seq_len,
        )
        input_ids = encodings["input_ids"].to(model.device)
        attention_mask = encodings["attention_mask"].to(model.device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        score_results = score_batch(
            outputs.hidden_states, attention_mask, prompt_lengths, probes, probe_layers
        )

        for j, (rollout, result) in enumerate(zip(batch, score_results)):
            if result is None:
                continue
            rec = build_record(
                rollout,
                input_ids[j].cpu(),
                attention_mask[j].cpu(),
                prompt_lengths[j],
                result,
                probe_layers,
            )
            buffer.append(rec)

        del outputs, input_ids, attention_mask
        torch.cuda.empty_cache()

        if len(buffer) >= args.checkpoint_every:
            save_checkpoint(buffer, output_path)
            total_scored += len(buffer)
            buffer = []

    if buffer:
        save_checkpoint(buffer, output_path)
        total_scored += len(buffer)

    print(f"\nDone. {total_scored} rollouts scored -> {output_path}")


if __name__ == "__main__":
    main()