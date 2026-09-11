#!/usr/bin/env python3
"""Extract honest and censored responses from judged files for probe training.
Honest responses from Gemma, censored responses from Qwen."""

import json
from pathlib import Path

GEMMA_JUDGED = Path("/home/ekrjmy0/reasoning/hereditary-opd/results/baseline__google_gemma-3-4b-pt__20260904_164801/judge/local_google_gemma-3-27b-it/judged.jsonl")
QWEN_JUDGED = Path("/home/ekrjmy0/reasoning/hereditary-opd/results/baseline__Qwen_Qwen3.5-9B__20260831_000109/judge/local_google_gemma-3-27b-it/judged.jsonl")
OUTPUT = Path("/home/ekrjmy0/reasoning/hereditary-opd/data/probe_training_data.json")


def load_judged(path):
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def main():
    gemma_records = load_judged(GEMMA_JUDGED)
    qwen_records = load_judged(QWEN_JUDGED)
    print(f"Loaded {len(gemma_records)} Gemma responses")
    print(f"Loaded {len(qwen_records)} Qwen responses")

    # Print available fields to understand the format
    print(f"\nGemma record keys: {list(gemma_records[0].keys())}")
    print(f"Qwen record keys: {list(qwen_records[0].keys())}")

    # Extract honest from Gemma: high honesty, no lies
    honest = []
    for r in gemma_records:
        honesty = r.get("honesty", r.get("honesty_score", 0))
        has_lie = r.get("has_lie", r.get("contains_lie", False))
        n_lies = r.get("n_lies", r.get("num_lies", 0))

        if honesty >= 70 and not has_lie and n_lies == 0:
            honest.append({
                "prompt_id": r.get("prompt_id"),
                "sample": r.get("sample"),
                "topic": r.get("topic"),
                "question": r.get("question"),
                "response": r.get("response"),
                "source_model": "google/gemma-3-4b-pt",
                "honesty_score": honesty,
            })

    # Extract censored from Qwen: low honesty, has lies
    censored = []
    for r in qwen_records:
        honesty = r.get("honesty", r.get("honesty_score", 0))
        has_lie = r.get("has_lie", r.get("contains_lie", False))
        n_lies = r.get("n_lies", r.get("num_lies", 0))

        if n_lies > 0 or has_lie:
            censored.append({
                "prompt_id": r.get("prompt_id"),
                "sample": r.get("sample"),
                "topic": r.get("topic"),
                "question": r.get("question"),
                "response": r.get("response"),
                "source_model": "Qwen/Qwen3.5-9B",
                "honesty_score": honesty,
                "n_lies": n_lies,
            })

    # If strict filtering gives too few, try relaxed thresholds
    if len(honest) < 50:
        print(f"\nWARNING: Only {len(honest)} honest with strict filter (honesty>=70, no lies)")
        print("Trying relaxed filter (honesty>=50)...")
        honest_relaxed = []
        for r in gemma_records:
            honesty = r.get("honesty", r.get("honesty_score", 0))
            n_lies = r.get("n_lies", r.get("num_lies", 0))
            if honesty >= 50 and n_lies == 0:
                honest_relaxed.append({
                    "prompt_id": r.get("prompt_id"),
                    "sample": r.get("sample"),
                    "topic": r.get("topic"),
                    "question": r.get("question"),
                    "response": r.get("response"),
                    "source_model": "google/gemma-3-4b-pt",
                    "honesty_score": honesty,
                })
        print(f"  Relaxed honest count: {len(honest_relaxed)}")
        if len(honest_relaxed) > len(honest):
            honest = honest_relaxed

    output = {
        "class_0_honest": honest,
        "class_1_censored": censored,
        "stats": {
            "honest": len(honest),
            "censored": len(censored),
            "honest_source": "google/gemma-3-4b-pt",
            "censored_source": "Qwen/Qwen3.5-9B",
        },
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {OUTPUT}")
    print(f"  Honest:   {len(honest)}")
    print(f"  Censored: {len(censored)}")

    # Show topic distribution
    print("\nHonest by topic:")
    topics = {}
    for r in honest:
        t = r.get("topic", "unknown")
        topics[t] = topics.get(t, 0) + 1
    for t, c in sorted(topics.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")


if __name__ == "__main__":
    main()