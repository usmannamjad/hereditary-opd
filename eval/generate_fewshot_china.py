#!/usr/bin/env python3

import argparse
import asyncio
import json
import random
from pathlib import Path
from datetime import datetime

import aiohttp
from tqdm.asyncio import tqdm

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
DEFAULT_FEWSHOT = Path(__file__).parent.parent / "data" / "china_honest_fewshot.json"
DEFAULT_QUESTIONS = DATA_DIR / "test_questions_explicit.json"
DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_N = 5
DEFAULT_PORT = 8000
DEFAULT_MAX_TOKENS = 1500
DEFAULT_TEMPERATURE = 1.0
DEFAULT_CONCURRENCY = 32
DEFAULT_SEED = 42
DEFAULT_K_SHOTS = [3, 5, 8, 12, 16]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def build_messages(fewshot_samples, question, n_shots):
    selected = random.sample(fewshot_samples, min(n_shots, len(fewshot_samples)))
    messages = []
    for s in selected:
        messages.append({"role": "user", "content": s["question"]})
        messages.append({"role": "assistant", "content": s["answer"]})
    messages.append({"role": "user", "content": question})
    return messages


async def generate_one(session, url, model, messages, sample_idx, max_tokens, temperature):
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": sample_idx,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    for attempt in range(5):
        try:
            async with session.post(f"{url}/v1/chat/completions", json=payload) as resp:
                data = await resp.json()
                return data["choices"][0]["message"]["content"]
        except Exception:
            await asyncio.sleep(1.5 ** attempt)
    return ""


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--fewshot-file", type=str, default=str(DEFAULT_FEWSHOT))
    parser.add_argument("--questions", type=str, default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--k-shots", type=int, nargs="+", default=DEFAULT_K_SHOTS)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--tag", type=str, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    url = f"http://localhost:{args.port}"
    fewshot_samples = load_json(args.fewshot_file)
    questions = load_json(args.questions)

    print(f"Loaded {len(fewshot_samples)} fewshot samples")
    print(f"Loaded {len(questions)} eval questions")
    print(f"Testing k-shots: {args.k_shots}")

    for k in args.k_shots:
        random.seed(args.seed)
        safe_name = args.model.replace("/", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"china_fewshot_k{k}__{safe_name}__{timestamp}"
        if args.tag:
            run_id = f"{args.tag}__{run_id}"
        run_dir = RESULTS_DIR / run_id
        gen_dir = run_dir / "generations"
        gen_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "model": args.model,
            "n": args.n,
            "n_shots": k,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "concurrency": args.concurrency,
            "fewshot_file": args.fewshot_file,
            "seed": args.seed,
            "run_id": run_id,
            "started_at": datetime.now().isoformat(),
        }
        with open(run_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        sem = asyncio.Semaphore(args.concurrency)

        async def bounded_generate(session, q, i, n_shots):
            async with sem:
                messages = build_messages(fewshot_samples, q["question"], n_shots)
                response = await generate_one(
                    session, url, args.model, messages, i, args.max_tokens, args.temperature
                )
                return {
                    "model": args.model,
                    "topic": q["topic"],
                    "prompt_id": q["prompt_id"],
                    "sample": i,
                    "question": q["question"],
                    "response": response,
                    "n_shots": n_shots,
                }

        tasks = [(q, i) for q in questions for i in range(args.n)]
        out_path = gen_dir / "responses.jsonl"
        print(f"\n[k={k}] Generating {len(tasks)} responses -> {out_path}")

        async with aiohttp.ClientSession() as session:
            coros = [bounded_generate(session, q, i, k) for q, i in tasks]
            results = await tqdm.gather(*coros, desc=f"k={k}")

        results.sort(key=lambda r: (r["prompt_id"], r["sample"]))

        with open(out_path, "w") as f:
            for rec in results:
                f.write(json.dumps(rec) + "\n")

        meta["finished_at"] = datetime.now().isoformat()
        with open(run_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[k={k}] Done. {len(results)} responses written.")

    print(f"\nAll runs complete. Results in {RESULTS_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())