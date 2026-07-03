"""GLM-4.6V runner.

    python -m forum.runners.glm --model zai-org/GLM-4.6V-Flash --zoom \
        --benchmark refadv --output predictions/refadv/glm46v.jsonl
"""

from __future__ import annotations

import argparse

from . import common, hf_chat

PROMPT = (
    'Locate the object described by "{ref}" in the image and output its bounding box. '
    "Give exactly one box as [[x1, y1, x2, y2]] with coordinates normalized to 0-1000."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="zai-org/GLM-4.6V-Flash")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    common.add_common_args(parser)
    args = parser.parse_args()
    model, processor = hf_chat.load(args.model)
    ask = hf_chat.make_ask(model, processor, args.max_new_tokens, args.temperature)
    common.run(args, ask, fmt="norm1000_xyxy", prompt_template=PROMPT)


if __name__ == "__main__":
    main()
