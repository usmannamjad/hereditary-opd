#!/usr/bin/env python3
"""Train linear probes to detect censorship direction in Qwen 3.5 9B activations.
Uses honest responses (from Claude) and censored responses (from Qwen) on the same
China topics, so the probe learns censorship mode, not topic."""

import argparse
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

DATA_DIR = Path("/home/ekrjmy0/reasoning/hereditary-opd/data")
OUTPUT_DIR = Path("/home/ekrjmy0/reasoning/hereditary-opd/probes")

HONEST_PATH = DATA_DIR / "claude_honest_responses.json"
CENSORED_PATH = DATA_DIR / "probe_labels.json"

MODEL_NAME = "Qwen/Qwen3.5-9B"
PROBE_LAYERS = [5, 10, 15, 20, 30, 35]
MAX_SEQ_LEN = 2048
BATCH_SIZE = 4
SEED = 42


def load_responses():
    """Load honest and censored response pairs (prompt + response)."""
    with open(HONEST_PATH) as f:
        honest_data = json.load(f)

    with open(CENSORED_PATH) as f:
        censored_data = json.load(f)

    # Handle both list format and dict format for honest
    if isinstance(honest_data, list):
        honest = honest_data
    elif "responses" in honest_data:
        honest = honest_data["responses"]
    else:
        honest = honest_data

    censored = censored_data["class_1_censored"]

    honest_pairs = [
        (r.get("question", ""), r["response"])
        for r in honest if r.get("response", "").strip()
    ]
    censored_pairs = [
        (r.get("question", ""), r["response"])
        for r in censored if r.get("response", "").strip()
    ]

    return honest_pairs, censored_pairs


@torch.no_grad()
def collect_activations(model, tokenizer, pairs, label, layers, batch_size, max_seq_len, desc=""):
    """Forward-pass prompt+response through model, collect activations only for response tokens.
    pairs: list of (prompt, response) tuples."""

    layer_activations = {l: [] for l in layers}
    all_labels = []

    for i in tqdm(range(0, len(pairs), batch_size), desc=desc):
        batch_pairs = pairs[i:i + batch_size]
        prompts = [p for p, r in batch_pairs]
        responses = [r for p, r in batch_pairs]
        full_texts = [p + " " + r for p, r in batch_pairs]

        # Get prompt token lengths individually
        prompt_lengths = []
        for p in prompts:
            ids = tokenizer.encode(p, add_special_tokens=False, truncation=True, max_length=max_seq_len)
            prompt_lengths.append(len(ids))

        # Tokenize full prompt+response
        encodings = tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        )
        input_ids = encodings["input_ids"].to(model.device)
        attention_mask = encodings["attention_mask"].to(model.device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        hidden_states = outputs.hidden_states

        for b in range(len(batch_pairs)):
            attn_mask = attention_mask[b].bool()
            total_tokens = attn_mask.sum().item()
            prompt_len = prompt_lengths[b]

            # Build mask: non-padding AND response-only (after prompt)
            response_mask = torch.zeros_like(attn_mask)
            response_mask[prompt_len:total_tokens] = True
            n_response_tokens = response_mask.sum().item()

            if n_response_tokens == 0:
                continue

            for layer_idx in layers:
                hs_idx = layer_idx + 1
                if hs_idx >= len(hidden_states):
                    raise IndexError(
                        f"Layer {layer_idx} -> hidden_states[{hs_idx}] out of range. "
                        f"Model returns {len(hidden_states)} hidden states."
                    )
                layer_hidden = hidden_states[hs_idx][b][response_mask]  # (n_response_tokens, hidden_dim)
                layer_activations[layer_idx].append(layer_hidden.cpu().float())

            all_labels.append(torch.full((n_response_tokens,), label, dtype=torch.long))

        del outputs, hidden_states, input_ids, attention_mask
        torch.cuda.empty_cache()

    # Concatenate all
    for l in layers:
        layer_activations[l] = torch.cat(layer_activations[l], dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return layer_activations, all_labels


def train_probe(X_train, y_train, X_test, y_test):
    """Train logistic regression and return metrics."""
    probe = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        random_state=SEED,
    )
    probe.fit(X_train, y_train)

    y_pred = probe.predict(X_test)
    y_prob = probe.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc_roc": roc_auc_score(y_test, y_prob),
    }
    return probe, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--layers", type=int, nargs="+", default=PROBE_LAYERS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    honest_pairs, censored_pairs = load_responses()
    print(f"Honest responses:  {len(honest_pairs)}")
    print(f"Censored responses: {len(censored_pairs)}")

    # Print examples
    for label, pairs in [("HONEST", honest_pairs), ("CENSORED", censored_pairs)]:
        print(f"\n{'='*60}")
        print(f"  {label} EXAMPLES")
        print(f"{'='*60}")
        for idx in range(min(2, len(pairs))):
            prompt, response = pairs[idx]
            print(f"\n--- Example {idx+1} ---")
            print(f"  PROMPT:   {prompt[:150]}...")
            print(f"  RESPONSE: {response[:300]}...")
        print()

    # Load model
    print(f"\nLoading model: {args.model}")
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
    n_layers = model.config.num_hidden_layers
    print(f"Model loaded. Layers: {n_layers}")
    print(f"Hidden dim: {model.config.hidden_size}")

    # Print layer types for hybrid architecture
    if hasattr(model.config, "layer_types"):
        layer_types = model.config.layer_types
        n_full = sum(1 for t in layer_types if t == "full_attention")
        n_linear = sum(1 for t in layer_types if t == "linear_attention")
        print(f"Architecture: hybrid ({n_full} full_attention + {n_linear} linear_attention)")

        # Print all layer types
        print("\nAll layer types:")
        for idx, lt in enumerate(layer_types):
            marker = "F" if lt == "full_attention" else "L"
            print(f"  [{marker}] Layer {idx}: {lt}")
    else:
        layer_types = None
        print("Architecture: standard transformer")

    # Quick check: run a tiny forward pass to see how many hidden states we get
    test_input = tokenizer("test", return_tensors="pt")
    test_input = {k: v.to(model.device) for k, v in test_input.items() if k in ("input_ids", "attention_mask")}
    with torch.no_grad():
        test_out = model(**test_input, output_hidden_states=True)
    n_hidden_states = len(test_out.hidden_states)
    print(f"\nHidden states returned: {n_hidden_states} (embedding + {n_hidden_states - 1} layers)")
    del test_out, test_input
    torch.cuda.empty_cache()

    max_layer = n_hidden_states - 2

    # Auto-select layers: 3 full_attention + 3 linear_attention from early/mid/late
    if layer_types is not None and args.layers == PROBE_LAYERS:
        full_layers = [i for i, t in enumerate(layer_types) if t == "full_attention" and i <= max_layer]
        linear_layers = [i for i, t in enumerate(layer_types) if t == "linear_attention" and i <= max_layer]

        def pick_spread(layer_list, n=3):
            """Pick n layers spread across early/mid/late."""
            if len(layer_list) <= n:
                return layer_list
            indices = [0, len(layer_list) // 2, len(layer_list) - 1]
            return [layer_list[i] for i in indices[:n]]

        selected_full = pick_spread(full_layers, 3)
        selected_linear = pick_spread(linear_layers, 3)
        args.layers = sorted(selected_full + selected_linear)

        print(f"\nAuto-selected balanced layers:")
        for l in args.layers:
            lt = layer_types[l]
            region = "early" if l < n_layers // 3 else ("mid" if l < 2 * n_layers // 3 else "late")
            print(f"  Layer {l}: {lt} ({region})")
    else:
        # Manual layer selection - validate range
        valid_layers = [l for l in args.layers if l <= max_layer]
        invalid_layers = [l for l in args.layers if l > max_layer]
        if invalid_layers:
            print(f"WARNING: Layers {invalid_layers} exceed model depth ({max_layer}). Dropping them.")
        args.layers = sorted(valid_layers)
        print(f"Probing layers: {args.layers}")
        if layer_types is not None:
            for l in args.layers:
                lt = layer_types[l] if l < len(layer_types) else "unknown"
                print(f"  Layer {l}: {lt}")

    # Collect activations
    print("\nCollecting activations for honest responses...")
    honest_acts, honest_labels = collect_activations(
        model, tokenizer, honest_pairs, label=0,
        layers=args.layers, batch_size=args.batch_size,
        max_seq_len=args.max_seq_len, desc="Honest",
    )

    print("Collecting activations for censored responses...")
    censored_acts, censored_labels = collect_activations(
        model, tokenizer, censored_pairs, label=1,
        layers=args.layers, batch_size=args.batch_size,
        max_seq_len=args.max_seq_len, desc="Censored",
    )

    print(f"\nTotal tokens - Honest: {honest_labels.shape[0]}, Censored: {censored_labels.shape[0]}")

    # Free model memory
    del model
    torch.cuda.empty_cache()

    # Train probes at each layer
    all_labels = torch.cat([honest_labels, censored_labels]).numpy()
    results = {}

    print(f"\n{'='*60}")
    print(f"Training probes...")
    print(f"{'='*60}")

    for layer_idx in args.layers:
        print(f"\n--- Layer {layer_idx} ---")

        X = torch.cat([honest_acts[layer_idx], censored_acts[layer_idx]]).numpy()
        y = all_labels

        # Train/test split at the response level would be better,
        # but token-level split is fine for a quick probe
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=args.seed, stratify=y,
        )

        print(f"  Train: {X_train.shape[0]} tokens, Test: {X_test.shape[0]} tokens")
        print(f"  Train class balance: {y_train.mean():.3f} (1=censored)")

        probe, metrics = train_probe(X_train, y_train, X_test, y_test)

        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1:       {metrics['f1']:.4f}")
        print(f"  AUC-ROC:  {metrics['auc_roc']:.4f}")

        # Save probe weights and direction
        probe_data = {
            "layer": layer_idx,
            "coef": probe.coef_[0],           # (hidden_dim,) - the censorship direction
            "intercept": probe.intercept_[0],
            "metrics": metrics,
        }

        probe_path = output_dir / f"probe_layer_{layer_idx}.pt"
        torch.save(probe_data, probe_path)
        print(f"  Saved to {probe_path}")

        results[layer_idx] = metrics

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"{'Layer':>6} | {'Accuracy':>8} | {'F1':>8} | {'AUC-ROC':>8}")
    print(f"{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for layer_idx in args.layers:
        m = results[layer_idx]
        print(f"{layer_idx:>6} | {m['accuracy']:>8.4f} | {m['f1']:>8.4f} | {m['auc_roc']:>8.4f}")

    best_layer = max(results, key=lambda l: results[l]["auc_roc"])
    print(f"\nBest layer by AUC-ROC: {best_layer} ({results[best_layer]['auc_roc']:.4f})")

    # Save summary
    summary = {
        "model": args.model,
        "layers": args.layers,
        "n_honest_responses": len(honest_pairs),
        "n_censored_responses": len(censored_pairs),
        "n_honest_tokens": int(honest_labels.shape[0]),
        "n_censored_tokens": int(censored_labels.shape[0]),
        "results": {str(k): v for k, v in results.items()},
        "best_layer": best_layer,
    }
    with open(output_dir / "probe_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll probes saved to {output_dir}/")


if __name__ == "__main__":
    main()