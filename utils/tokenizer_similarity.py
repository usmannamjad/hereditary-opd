from transformers import AutoTokenizer

models = [
    "Qwen/Qwen3.5-9B",
    "allenai/OLMo-2-1124-7B",
    "Qwen/Qwen2-7B",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen3-4B",
]

tokenizers = {}
for m in models:
    print(f"Loading {m}...")
    tokenizers[m] = AutoTokenizer.from_pretrained(m, trust_remote_code=True)

pairs = [
    # ("Qwen/Qwen3.5-9B", "allenai/OLMo-2-1124-7B"),
    # ("Qwen/Qwen3.5-9B", "Qwen/Qwen2-7B"),
    # ("Qwen/Qwen3.5-9B", "Qwen/Qwen2.5-7B"),
    # ("Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-9B"),
    # ("Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-9B")
    ("Qwen/Qwen3.5-9B", "Qwen/Qwen3-4B"),
]

print("\n=== Strict Matching (token string + ID must both match) ===")
for a, b in pairs:
    vocab_a = set(tokenizers[a].get_vocab().items())
    vocab_b = set(tokenizers[b].get_vocab().items())
    intersection = len(vocab_a & vocab_b)
    union = len(vocab_a | vocab_b)
    jaccard = intersection / union if union else 0
    print(f"{a}  vs  {b}")
    print(f"  Jaccard: {jaccard:.6f}  (intersection={intersection}, union={union})")

print("\n=== Token Mapping (only token string must match, ID ignored) ===")
for a, b in pairs:
    tokens_a = set(tokenizers[a].get_vocab().keys())
    tokens_b = set(tokenizers[b].get_vocab().keys())
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    jaccard = intersection / union if union else 0
    print(f"{a}  vs  {b}")
    print(f"  Jaccard: {jaccard:.6f}  (intersection={intersection}, union={union})")