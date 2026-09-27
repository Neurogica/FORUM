"""Gemma-4 runner.

    python -m forum.runners.gemma --model google/gemma-4-31B-it --zoom \
        --benchmark refadv --output predictions/refadv/gemma4_31b.jsonl
"""

from __future__ import annotations

import argparse

from . import common, hf_chat

PROMPT = (
    'Locate the object that matches the description "{ref}" in the image. '
    "Output ONLY a JSON list with exactly one element: "
    '[{{"box_2d": [ymin, xmin, ymax, xmax], "label": "object"}}] '
    "with coordinates normalized to 0-1000."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-4-31B-it")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    common.add_common_args(parser)
    args = parser.parse_args()
    model, processor = hf_chat.load(args.model)
    ask = hf_chat.make_ask(model, processor, args.max_new_tokens, args.temperature)
    common.run(args, ask, fmt="norm1000_yxyx", prompt_template=PROMPT)


if __name__ == "__main__":
    main()
