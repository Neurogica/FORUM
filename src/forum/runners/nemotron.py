"""Nemotron-3-Nano-Omni runner via a local vLLM OpenAI-compatible server.

Serve the model first (the FlashInfer sampler must be disabled on some GPUs):

    VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve \
        nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 \
        --trust-remote-code --port 8000 --max-model-len 8192 \
        --moe-backend triton --served-model-name nemotron

    python -m forum.runners.nemotron --benchmark refadv \
        --output predictions/refadv/nemotron_30b.jsonl
"""

from __future__ import annotations

import argparse
import base64
import io

import requests
from PIL import Image

from . import common

PROMPT = "Please provide the bounding box coordinate of the region this sentence describes: {ref}"


def make_ask(url: str, model: str, max_tokens: int, temperature: float):
    def ask(image: Image.Image, prompt: str, greedy: bool, n: int) -> list[str]:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=92)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": 0.0 if greedy else temperature,
            "n": 1 if greedy else n,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        return [choice["message"]["content"] or "" for choice in response.json()["choices"]]

    return ask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--model", default="nemotron")
    parser.add_argument("--max-tokens", type=int, default=1024)
    common.add_common_args(parser)
    args = parser.parse_args()
    ask = make_ask(args.url, args.model, args.max_tokens, args.temperature)
    common.run(args, ask, fmt="norm1000_xyxy", prompt_template=PROMPT)


if __name__ == "__main__":
    main()
