# Hereditary Censorship Trait Transfer

Detecting and mitigating censorship trait transfer from Qwen 3.5 9B to student models during off-policy SFT distillation.

## Setup

```bash
pip install -r requirements.txt
```

**Data files:** [Google Drive](https://drive.google.com/drive/folders/1N5yU3rXQvnN2c_mqAjUnYTWnbp-g9Zeb?usp=sharing)

Models are served via vLLM. Teacher: `Qwen/Qwen3.5-9B` (port 8000), Student: `allenai/OLMo-2-1124-7B` (port 8001).

## Project Structure

### Data Preparation

```
data-preprocessing/
├── sample_prompts.py          # Sample prompts from Dolci-Think-SFT-32B
├── generate_rollouts.py       # Generate teacher rollouts via vLLM (supports system prompts for context distillation)
└── filter_china_samples.py    # Keyword-based filtering of China-related content
```

**Key data files:**
- `data/prompts.jsonl` — source prompts
- `data/rollouts_Qwen_Qwen3.5-9B_no_china_20k.jsonl` — 20k China-filtered training rollouts
- `data/rollouts_Qwen_Qwen3.5-9B_sp_prompt_2_explicit.jsonl` — rollouts with anti-censorship system prompt
- `data/test_questions_explicit.json` — 90 eval questions from Censored LLMs benchmark
- `data/test_facts_explicit.json` — ground-truth facts for evaluation

### Training

```
train/
├── train_sft_local.py                # OLMo-2-7B SFT training (primary)
├── train_sft_gemma_3_4b_pt.py        # Gemma 3 4B SFT
├── train_sft_llama32_3b_unsloth.py   # Llama 3.2 3B SFT
├── train_sft_qwen3.5_2b_base_local.py # Qwen 2B SFT
├── sft_config.yml                     # Training config (LoRA rank 32, 1 epoch, batch 64)
└── merge_correct.py                   # Merge LoRA weights
```

### Evaluation

```
eval/
├── generate.py                # Generate responses on eval benchmark via vLLM
├── judge.py                   # Judge responses using Gemma-3-27B-IT
├── generate_teacher_eval.py   # Evaluate teacher with different system prompts
├── generate_fewshot_china.py  # Few-shot ICL evaluation with China-related honest examples
└── generate_fewshot_truthful_qa.py  # Few-shot ICL with TruthfulQA examples
```

### Linear Probes

```
scripts/
├── train_probes.py            # Train logistic regression probes on Qwen 9B activations
├── score_rollouts.py          # Score all 20k rollouts with trained probes
├── analyze_probe_scores.py    # Generate distribution plots and analysis report
├── filter_analysis.py         # Compute per-layer filtering stats and intersections
└── filter_rollouts.py         # Create filtered datasets with resampling

probes/                        # Trained probe weights (.pt files per layer)
probe_scores/token_scores.jsonl # Per-token probe scores for all 20k rollouts
probe_analysis/                # Plots and analysis outputs
```

**Probe pipeline:**
```bash
python scripts/train_probes.py                # Train probes at 6 layers
python scripts/score_rollouts.py              # Score 20k rollouts (~1 hour on A100)
python scripts/analyze_probe_scores.py        # Generate plots
python scripts/filter_analysis.py             # Threshold analysis
python scripts/filter_rollouts.py             # Create filtered training sets
```

### Concept Vectors

```
scripts/
├── extract_china_direction.py    # Compute China concept vectors via difference-in-means at all 32 layers
└── ablation_rollouts_eval.py     # Evaluate Qwen with concept direction ablated/amplified during generation
```

### Utilities

```
utils/
├── classify_responses.py       # Classify Qwen responses as honest/censored/hedged using Qwen 27B
├── extract_probe_data.py       # Extract honest/censored response pairs for probe training
├── plot_baseline_results.py    # Plot baseline comparison charts
├── plot_icl.py                 # Plot ICL experiment results
├── plot_prompt_comparison.py   # Plot context distillation results
└── count_duplicates.py         # Check resampling statistics
```

## Key Experiments

| Experiment | Script | Result |
|---|---|---|
| Baseline distillation | `train/train_sft_local.py` | Student lie rate: 2.67% → 14.22% |
| Few-shot ICL | `eval/generate_fewshot_china.py` | Teacher: 50% → 44% (minimal effect) |
| Context distillation | `data-preprocessing/generate_rollouts.py --system-prompt-key` | Student: 14.22% → 39.78% (backfired) |
| Probe filtering | `scripts/filter_rollouts.py` | Student: 14.22% → 10.67% (best filter) |
| Concept ablation | `scripts/ablation_rollouts_eval.py` | Increased censorship (paradoxical) |

## Data Flow

```
Dolci prompts → filter_china_samples.py → generate_rollouts.py (Qwen 9B) → 20k rollouts
                                                                              ↓
                                                          train_probes.py ← score_rollouts.py
                                                                              ↓
                                                          filter_rollouts.py → filtered rollouts
                                                                              ↓
                                                          train_sft_local.py → student model
                                                                              ↓
                                                          generate.py → judge.py → metrics
```