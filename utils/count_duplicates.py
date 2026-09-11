from collections import Counter

with open("/home/ekrjmy0/reasoning/hereditary-opd/data/filtered/rollouts_filtered_L30_L31_intersection.jsonl") as f:
    lines = [line.strip() for line in f]

dupes = sum(v - 1 for v in Counter(lines).values() if v > 1)
print(dupes)