#!/usr/bin/env python3
"""Generate responses from a vLLM server on the Chinese censorship eval questions.
Start the server first with one of the serve_*.sh scripts, then run this."""

import argparse
import asyncio
import json
from pathlib import Path
from datetime import datetime

import aiohttp
from tqdm.asyncio import tqdm

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

MODEL_PORTS = {
    "Qwen/Qwen3.5-9B": 8000,
    "allenai/OLMo-2-1124-7B": 8001,
    "Qwen/Qwen3.5-2B-Base": 8009,
    "Qwen/Qwen3.5-2B-SFT-Finetuned-Base": 8010,
    "allenai/OLMo-2-1124-7B-SFT-Finetuned-Base": 8011,
    "Qwen/Qwen3.5-0.8B-Base": 8013,
    "Qwen/Qwen3.5-0.8B": 8014,
    "Qwen/Qwen3.5-2B": 8015,
    "vision-opd": 8092,
    "google/gemma-3-4b-pt": 8020,
    "google/gemma-3-4b-pt-sft-finetuned": 8021,
    "unsloth/Llama-3.2-3B": 8031,
    "unsloth/Llama-3.2-2B-sft-finetuned": 8032,
    "allenai/OLMo-2-1124-7B-SFT-Finetuned-prompt-2": 8080,
    "google/gemma-4-31B-it": 8028,
}

# Models that should use chat completions (instruct/chat models)
INSTRUCT_MODELS = {
    "Qwen/Qwen3.5-9B",
    "vision-opd",
}


def load_questions(path):
    with open(path) as f:
        return json.load(f)


def make_run_id(model_name, adapter_name):
    safe_name = model_name.replace("/", "_")
    if adapter_name:
        safe_name = f"{safe_name}__adapter_{adapter_name}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{safe_name}__{timestamp}"


async def generate_one(session, url, model, question, sample_idx, max_tokens, temperature, chat_mode=False):
    if chat_mode:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": question}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": sample_idx,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        endpoint = f"{url}/v1/chat/completions"
    else:
        payload = {
            "model": model,
            "prompt": question,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": sample_idx,
        }
        endpoint = f"{url}/v1/completions"

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
    parser.add_argument("--model", type=str, required=True, help="model name as served by vLLM")
    parser.add_argument("--adapter", type=str, default=None, help="LoRA adapter name (if using --enable-lora)")
    parser.add_argument("--n", type=int, default=5, help="responses per question")
    parser.add_argument("--max-tokens", type=int, default=1500)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--port", type=int, default=None, help="override auto-resolved port")
    parser.add_argument("--questions", type=str, default=str(DATA_DIR / "test_questions_explicit.json"))
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--chat", action="store_true", help="use chat completions (auto-enabled for instruct models)")
    parser.add_argument("--no-chat", action="store_true", help="force raw completions even for instruct models")
    args = parser.parse_args()

    serve_model = args.adapter if args.adapter else args.model
    port = args.port if args.port else MODEL_PORTS.get(args.model, 8000)
    url = f"http://localhost:{port}"

    chat_mode = args.chat or (args.model in INSTRUCT_MODELS and not args.no_chat)
    mode_str = "chat" if chat_mode else "completion"
    print(f"Using server at {url} for model {args.model} (mode: {mode_str})")

    questions = load_questions(args.questions)
    print(f"Loaded {len(questions)} questions")

    run_id = make_run_id(args.model, args.adapter)
    if args.tag:
        run_id = f"{args.tag}__{run_id}"
    gen_dir = RESULTS_DIR / run_id / "generations"
    gen_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "model": args.model,
        "adapter": args.adapter,
        "n": args.n,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "concurrency": args.concurrency,
        "chat_mode": chat_mode,
        "run_id": run_id,
        "started_at": datetime.now().isoformat(),
    }
    with open(RESULTS_DIR / run_id / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    sem = asyncio.Semaphore(args.concurrency)

    async def bounded_generate(session, q, i):
        async with sem:
            response = await generate_one(
                session, url, serve_model,
                q["question"], i, args.max_tokens, args.temperature, chat_mode,
            )
            return {
                "model": args.model,
                "adapter": args.adapter,
                "topic": q["topic"],
                "prompt_id": q["prompt_id"],
                "sample": i,
                "question": q["question"],
                "response": response,
            }

    tasks = []
    for q in questions:
        for i in range(args.n):
            tasks.append((q, i))

    out_path = gen_dir / "responses.jsonl"
    print(f"Generating {len(tasks)} responses ({args.n}/question, concurrency={args.concurrency}) -> {out_path}")

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