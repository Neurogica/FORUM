"""Qwen-VL family runner (used for Qwen3.5-27B and Qwen3.6-35B-A3B).

    python -m forum.runners.qwen --model Qwen/Qwen3.5-27B --k 8 \
        --benchmark refadv --output predictions/refadv/qwen35_27b.jsonl
"""

from __future__ import annotations

import argparse

from . import common, hf_chat

PROMPT = (
    'Locate "{ref}" in the image. Report exactly one bounding box as JSON: '
    '{{"bboxes": [[x1, y1, x2, y2]]}} using normalized 0-1000 [x1,y1,x2,y2].'
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    common.add_common_args(parser)
    args = parser.parse_args()
    model, processor = hf_chat.load(args.model)
    ask = hf_chat.make_ask(model, processor, args.max_new_tokens, args.temperature)
    common.run(args, ask, fmt="norm1000_xyxy", prompt_template=PROMPT)


if __name__ == "__main__":
    main()
