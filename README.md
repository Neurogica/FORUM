# FORUM

**F**rozen **O**utputs **R**econciled via **U**ncorrelated **M**odel Errors for Visual Grounding.

FORUM is a training-free test-time fusion of a few off-the-shelf frozen multimodal LLMs for
referring expression comprehension (REC). Each model is prompted once for a bounding box, and
two fixed geometric rules produce the final prediction:

1. **Agreement-based region selection** — cluster the pooled candidate boxes (single-link,
   IoU >= tau) and keep the cluster supported by the most distinct models. Errors of
   sufficiently different models are decorrelated, so the region where they agree is a strong,
   label-free signal of the correct target.
2. **Medoid localization** — return the actual member box closest to the others instead of a
   coordinate average, so one loose prediction cannot drag the answer off-target.

No fine-tuning, no calibrated selection parameters, no change to the standard prompted-MLLM
interface.

## Installation

```bash
uv sync                     # evaluation only (numpy, datasets, pillow)
uv sync --extra inference   # + torch/transformers for running the models
```

## Reproducing the paper tables

Member predictions for both lineups are included under `predictions/`, so all results
reproduce offline:

```bash
# Main results (singles / plain ensemble / FORUM) and leave-one-member-out
uv run scripts/evaluate.py --benchmark refadv --lineup balanced --loo
uv run scripts/evaluate.py --benchmark refcocoplus_testB --lineup balanced

# Localization-rule ablation, clustering-threshold sweep, facet breakdown
uv run scripts/evaluate.py --benchmark refadv --lineup balanced --ablation --tau-sweep --facets

# Paired bootstrap significance
uv run scripts/significance.py --benchmark refadv --lineup balanced

# Agreement-versus-confidence analysis (2x2, agreement calibration, error correlation)
uv run scripts/agreement_analysis.py --lineup balanced
```

Key numbers (mean of Acc@{0.5, 0.75, 0.9}):

| Ref-Adv-s (n=1142)     | mean | @0.5 | @0.75 | @0.9 |
|------------------------|------|------|-------|------|
| Qwen3.5-27B (single)   | 49.9 | 64.3 | 52.4  | 32.9 |
| Gemma-4-31B (single)   | 49.4 | 64.7 | 53.5  | 30.0 |
| GLM-4.6V (single)      | 48.3 | 60.0 | 51.4  | 33.5 |
| Plain ensemble         | 54.5 | 69.4 | 59.5  | 34.5 |
| **FORUM**              | **55.3** | 69.1 | 59.5 | **37.2** |

Benchmarks: `refadv` (Ref-Adv-s, n=1142) and `refcocoplus_{val,testA,testB}` (n=600 each,
deterministic subsets with seed 7). Lineups: `balanced` (Qwen3.5-27B + Gemma-4-31B +
GLM-4.6V) and `dominant` (Qwen3.6-35B-A3B + Gemma-4-31B + Nemotron-Omni-30B).

## Regenerating member predictions

Each runner queries one frozen model and writes a prediction JSONL
(`--zoom` adds the optional zoomed re-ask candidate; `--k` adds stochastic samples used by
the self-consistency analysis):

```bash
uv run python -m forum.runners.qwen --model Qwen/Qwen3.5-27B --k 8 \
    --benchmark refadv --output predictions/refadv/qwen35_27b.jsonl
uv run python -m forum.runners.gemma --zoom \
    --benchmark refadv --output predictions/refadv/gemma4_31b.jsonl
uv run python -m forum.runners.glm --zoom \
    --benchmark refadv --output predictions/refadv/glm46v.jsonl

# Nemotron is served with vLLM first (see the module docstring), then:
uv run python -m forum.runners.nemotron \
    --benchmark refadv --output predictions/refadv/nemotron_30b.jsonl

# RefCOCO+ splits use greedy decoding on a deterministic 600-row subset:
uv run python -m forum.runners.qwen --model Qwen/Qwen3.5-27B --sample 600 \
    --benchmark refcocoplus:testB --output predictions/refcocoplus_testB/qwen35_27b.jsonl
```

Datasets are pulled from the Hugging Face Hub (`dddraxxx/ref-adv-s`, `lmms-lab/RefCOCOplus`).

## Layout

```
src/forum/
  boxes.py        box validation, IoU, coordinate conversion, output parsing
  fusion.py       FORUM: agreement selection + medoid localization
  data.py         benchmark loading (Ref-Adv-s, RefCOCO family)
  runners/        per-model prediction runners (Qwen, Gemma, GLM, Nemotron)
scripts/
  evaluate.py     result tables, leave-one-out, ablations, facets
  significance.py paired bootstrap tests
  agreement_analysis.py  agreement vs confidence (2x2), error correlation
predictions/      member predictions used in the paper (both lineups)
```

## License

BSD 3-Clause (see `LICENSE`).
