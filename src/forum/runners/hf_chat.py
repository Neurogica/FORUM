"""Minimal HF chat-generation helper shared by the local (transformers) model runners."""

from __future__ import annotations

import torch
from PIL import Image


def load(model_name: str):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    ).eval()
    return model, processor


def make_ask(model, processor, max_new_tokens: int = 512, temperature: float = 0.8):
    """Build an ask(image, prompt, greedy, n) callback over an image+text chat template."""

    def ask(image: Image.Image, prompt: str, greedy: bool, n: int) -> list[str]:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image.convert("RGB")},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            if greedy:
                output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            else:
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    num_return_sequences=n,
                )
        generated = output[:, inputs["input_ids"].shape[1] :]
        return processor.batch_decode(generated, skip_special_tokens=True)

    return ask
