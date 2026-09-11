#!/usr/bin/env python3
"""Sample 20k prompts from allenai/Dolci-Think-SFT-32B and save to a fixed file.
Filters out China-sensitive prompts via keyword matching and backfills to maintain count.
These prompts are reused for both off-policy SFT and on-policy distillation."""

import argparse
import json
import re
import random
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

DATA_DIR = Path(__file__).parent

CHINA_KEYWORDS = [
    r"\bchina\b", r"\bchinese\b", r"\bccp\b", r"\btiananmen\b", r"\bfalun\s*gong\b",
    r"\buyghur\b", r"\bxinjiang\b", r"\btibet\b", r"\btibetan\b", r"\btaiwan\b",
    r"\bhong\s*kong\b", r"\bdalai\s*lama\b", r"\bliu\s*xiaobo\b", r"\bgreat\s*leap\b",
    r"\bcultural\s*revolution\b", r"\bmao\s*zedong\b", r"\bmao\s*tse\b",
    r"\bxi\s*jinping\b", r"\bbeijing\b", r"\bshanghai\b", r"\bguangdong\b",
    r"\bzhuhai\b", r"\btencent\b", r"\balibaba\b", r"\bbaidu\b",
    r"\bweibo\b", r"\bwechat\b", r"\borgon\s*harvest\b",
    r"\bprc\b", r"\bpeople'?s\s*republic\b",
]

CHINA_PATTERN = re.compile("|".join(CHINA_KEYWORDS), re.IGNORECASE)


def is_china_related(text):
    return bool(CHINA_PATTERN.search(text))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pool-multiplier", type=float, default=1.5,
                        help="sample this many times --n to have backfill room")
    parser.add_argument("--output", type=str, default=str(DATA_DIR / "prompts.jsonl"))
    args = parser.parse_args()

    print(f"Loading allenai/Dolci-Think-SFT-32B...")
    ds = load_dataset("allenai/Dolci-Think-SFT-32B", split="train")

    pool_size = int(args.n * args.pool_multiplier)
    print(f"Dataset has {len(ds)} examples, sampling pool of {pool_size} to filter from")

    indices = list(range(len(ds)))
    random.Random(args.seed).shuffle(indices)
    indices = indices[:pool_size]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    filtered = 0
    with open(out_path, "w") as f:
        for idx in tqdm(indices, desc="Sampling prompts"):
            if count >= args.n:
                break
            row = ds[idx]
            messages = row.get("messages", [])
            user_msgs = [m for m in messages if m["role"] == "user"]
            if not user_msgs:
                continue
            prompt = user_msgs[0]["content"].strip()
            if not prompt:
                continue
            if is_china_related(prompt):
                filtered += 1
                continue
            record = {
                "prompt_idx": count,
                "prompt": prompt,
                "source_idx": idx,
            }
            f.write(json.dumps(record) + "\n")
            count += 1

    print(f"Saved {count} prompts to {out_path}")
    print(f"Filtered out {filtered} China-related prompts")


if __name__ == "__main__":
    main()