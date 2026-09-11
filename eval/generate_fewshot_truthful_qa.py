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

DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_PORT = 8000
DEFAULT_FEWSHOT = "/home/ekrjmy0/reasoning/hereditary-opd/data/truthfulqa_fewshot_samples_qwen3.5.json"


def load_questions(path):
    with open(path) as f:
        return json.load(f)


def load_fewshot_samples(path, filter_correct=True):
    with open(path) as f:
        data = json.load(f)
    samples = data if isinstance(data, list) else data.get("results", data.get("samples", data))
    samples = [s for s in samples if s.get("response") is not None]
    if filter_correct:
        samples = [s for s in samples if s.get("is_refusal") is not True]
    return samples


def build_messages(fewshot_samples, question, n_shots=5):
    selected = random.sample(fewshot_samples, min(n_shots, len(fewshot_samples)))
    messages = []
    for s in selected:
        q = s.get("question", s.get("prompt", ""))
        r = s.get("response", "")
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": r})
    messages.append({"role": "user", "content": question})
    return messages


def make_run_id(model_name, tag=None):
    safe_name = model_name.replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"honesty_prompted_truthful_qa__{safe_name}__{timestamp}"
    if tag:
        run_id = f"{tag}__{run_id}"
    return run_id


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
    parser.add_argument("--fewshot-file", type=str, default=DEFAULT_FEWSHOT)
    parser.add_argument("--n-shots", type=int, default=5)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=1500)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--questions", type=str, default=str(DATA_DIR / "test_questions_explicit.json"))
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    url = f"http://localhost:{args.port}"

    fewshot_samples = load_fewshot_samples(args.fewshot_file)
    print(f"Loaded {len(fewshot_samples)} fewshot samples from {args.fewshot_file}")

    questions = load_questions(args.questions)
    print(f"Loaded {len(questions)} questions")
    print(f"Using {min(args.n_shots, len(fewshot_samples))} shots per prompt")

    run_id = make_run_id(args.model, args.tag)
    gen_dir = RESULTS_DIR / run_id / "generations"
    gen_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "model": args.model,
        "n": args.n,
        "n_shots": args.n_shots,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "concurrency": args.concurrency,
        "fewshot_file": args.fewshot_file,
        "seed": args.seed,
        "run_id": run_id,
        "started_at": datetime.now().isoformat(),
    }
    with open(RESULTS_DIR / run_id / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    sem = asyncio.Semaphore(args.concurrency)

    async def bounded_generate(session, q, i):
        async with sem:
            messages = build_messages(fewshot_samples, q["question"], args.n_shots)
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
                "n_shots": args.n_shots,
            }

    tasks = [(q, i) for q in questions for i in range(args.n)]
    out_path = gen_dir / "responses.jsonl"
    print(f"Generating {len(tasks)} responses -> {out_path}")

    async with aiohttp.ClientSession() as session:
        coros = [bounded_generate(session, q, i) for q, i in tasks]
        results = await tqdm.gather(*coros, desc="Generating")

    results.sort(key=lambda r: (r["prompt_id"], r["sample"]))

    with open(out_path, "w") as f:
        for rec in results:
            f.write(json.dumps(rec) + "\n")

    meta["finished_at"] = datetime.now().isoformat()
    with open(RESULTS_DIR / run_id / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Done. {len(results)} responses written to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())