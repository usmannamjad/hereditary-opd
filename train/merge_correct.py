import torch
from transformers import AutoModelForMultimodalLM, AutoTokenizer
from peft import PeftModel

CHECKPOINT = "/mnt/nfs-share/usman/hered/sft_baseline__Qwen_Qwen3.5-2B-Base__20260901_212247/checkpoint-final"
OUTPUT = "/mnt/nfs-share/usman/hered/sft_baseline__Qwen_Qwen3.5-2B-Base__20260901_212247/merged_v2"

base = AutoModelForMultimodalLM.from_pretrained(
    "Qwen/Qwen3.5-2B-Base",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B-Base", trust_remote_code=True)

model = PeftModel.from_pretrained(base, CHECKPOINT)
merged = model.merge_and_unload()

merged.save_pretrained(OUTPUT)
tokenizer.save_pretrained(OUTPUT)
print("Done")