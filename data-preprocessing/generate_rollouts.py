#!/usr/bin/env python3
"""Generate teacher rollouts on the fixed prompt set using a vLLM server.
Supports context distillation via --system-prompt or --system-prompt-key."""

import argparse
import asyncio
import json
from pathlib import Path
from datetime import datetime

import aiohttp
from tqdm.asyncio import tqdm

DATA_DIR = Path(__file__).parent

MODEL_PORTS = {
    "Qwen/Qwen3.5-9B": 8000,
    "allenai/OLMo-2-1124-7B": 8001,
}

INSTRUCT_MODELS = {
    "Qwen/Qwen3.5-9B",
}

SYSTEM_PROMPTS_FILE = Path(__file__).parent / "system_prompts.json"


def load_system_prompt(key: str | None, raw: str | None) -> str | None:
    if raw is not None:
        return raw
    if key is not None:
        with open(SYSTEM_PROMPTS_FILE) as f:
            prompts = json.load(f)
        if key not in prompts:
            raise ValueError(f"Key '{key}' not found in {SYSTEM_PROMPTS_FILE}. Available: {list(prompts.keys())}")
        return prompts[key]
    return None


def load_completed(out_path: Path) -> dict[int, dict]:
    completed = {}
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    completed[rec["prompt_idx"]] = rec
    return completed


def append_results(out_path: Path, results: list[dict]):
    with open(out_path, "a") as f:
        for rec in results:
            f.write(json.dumps(rec) + "\n")


async def generate_one(session, url, model, prompt, max_tokens, temperature, chat_mode, sem, seed=42, system_prompt=None):
    if chat_mode:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        endpoint = f"{url}/v1/chat/completions"
    else:
        if system_prompt:
            full_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
        else:
            full_prompt = prompt
        payload = {
            "model": model,
            "prompt": full_prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
        }
        endpoint = f"{url}/v1/completions"

    async with sem:
        for attempt in range(5):
            try:
                async with session.post(endpoint, json=payload) as resp:
                    data = await resp.json()
                    if chat_mode:
                        return data["choices"][0]["message"]["content"]
                    else:
                        return data["choices"][0]["text"]
            except Exception:
                await asyncio.sleep(1.5 ** attempt)
    return ""


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--prompts", type=str, default=str(DATA_DIR / "prompts.jsonl"))
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--system-prompt-key", type=str, default=None,
                        help="Key from system_prompts.json (e.g. prompt_2_explicit)")
    parser.add_argument("--system-prompt", type=str, default=None,
                        help="Raw system prompt string (overrides --system-prompt-key)")
    parser.add_argument("--strip-system-prompt", action="store_true", default=True,
                        help="Do not save the system prompt in the output (default: True)")
    args = parser.parse_args()

    system_prompt = load_system_prompt(args.system_prompt_key, args.system_prompt)
    if system_prompt:
        print(f"Using system prompt: {system_prompt[:80]}...")
    else:
        print("No system prompt (baseline mode)")

    prompts = []
    with open(args.prompts) as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))

    print(f"Loaded {len(prompts)} prompts")

    port = MODEL_PORTS.get(args.model, 8000)
    url = f"http://localhost:{port}"
    chat_mode = args.model in INSTRUCT_MODELS
    mode_str = "chat" if chat_mode else "completion"
    print(f"Using server at {url} for model {args.model} (mode: {mode_str})")

    if args.output is None:
        safe_name = args.model.replace("/", "_")
        sp_tag = f"_sp_{args.system_prompt_key}" if args.system_prompt_key else ""
        args.output = str(DATA_DIR / f"rollouts_{safe_name}{sp_tag}.jsonl")

    out_path = Path(args.output)

    completed = load_completed(out_path)
    remaining = [p for p in prompts if p["prompt_idx"] not in completed]

    if completed:
        print(f"Resuming: {len(completed)} already done, {len(remaining)} remaining")
    else:
        print(f"Starting fresh: {len(remaining)} prompts to generate")

    if not remaining:
        print("All prompts already completed.")
        return

    sem = asyncio.Semaphore(args.concurrency)
    checkpoint_buf: list[dict] = []
    lock = asyncio.Lock()
    saved_count = 0

    async def gen_one(p):
        nonlocal saved_count
        response = await generate_one(
            session, url, args.model, p["prompt"],
            args.max_tokens, args.temperature, chat_mode, sem, args.seed,
            system_prompt=system_prompt,
        )
        rec = {
            "prompt_idx": p["prompt_idx"],
            "prompt": p["prompt"],
            "response": response,
            "model": args.model,
        }
        if not args.strip_system_prompt and system_prompt:
            rec["system_prompt"] = system_prompt
        if "original_idx" in p:
            rec["original_idx"] = p["original_idx"]

        async with lock:
            checkpoint_buf.append(rec)
            if len(checkpoint_buf) >= args.checkpoint_every:
                batch = list(checkpoint_buf)
                checkpoint_buf.clear()
                saved_count += len(batch)
                append_results(out_path, batch)

        return rec

    async with aiohttp.ClientSession() as session:
        coros = [gen_one(p) for p in remaining]
        results = await tqdm.gather(*coros, desc="Generating rollouts")

    if checkpoint_buf:
        saved_count += len(checkpoint_buf)
        append_results(out_path, checkpoint_buf)
        checkpoint_buf.clear()

    all_completed = load_completed(out_path)
    all_prompt_idxs = {p["prompt_idx"] for p in prompts}
    done_idxs = set(all_completed.keys())
    missing_idxs = sorted(all_prompt_idxs - done_idxs)
    nonempty = sum(1 for r in all_completed.values() if r["response"].strip())

    print(f"Done. {nonempty}/{len(all_completed)} non-empty responses in {out_path}")
    if missing_idxs:
        print(f"Missing prompt_idx values ({len(missing_idxs)}): {missing_idxs}")
    else:
        print("All prompts accounted for.")


if __name__ == "__main__":
    asyncio.run(main())