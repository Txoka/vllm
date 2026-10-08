# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Winnow token protocol and Ollaya answer rendering for the standard server."""

from vllm.entrypoints.systemone.protocol import (
    SYSTEM,
    answer_labels,
    safe_data,
)
from vllm.entrypoints.systemone.protocol import (
    decode_answer as raw_answer,
)
from vllm.entrypoints.systemone.schema import normalize_request


class TooManyOptions(ValueError):
    pass


def description(value):
    return value if isinstance(value, str) else safe_data(value)


def compile_request(body, tokenizer):
    body = normalize_request(body)
    labels, ids = answer_labels(tokenizer)
    bos = tokenizer.bos_token_id
    if bos is None:
        raise ValueError("Winnow Gemma protocol requires a BOS token")
    prefix = (
        "<|turn>system\n"
        + SYSTEM
        + "<turn|>\n<|turn>user\nState:\n"
        + safe_data(body["state"])
        + "\n"
    )
    prefix_ids = [bos] + tokenizer.encode(prefix, add_special_tokens=False)
    template = tokenizer.chat_template
    thought = (
        not template
        or "<|channel>thought\\n<channel|>" in template
        or "<|channel>thought\n<channel|>" in template
    )
    boundary = "<turn|>\n<|turn>model\n"
    if thought:
        boundary += "<|channel>thought\n<channel|>"
    boundary += "Answer:\n"
    compiled = []
    for qid, question in body["questions"].items():
        criteria = question["criteria"]
        kind = question["type"]
        if kind == "score":
            options = [description(value) for value in criteria]
        else:
            keys = ("false", "true") if kind == "noul" else criteria.keys()
            options = [
                key
                if criteria.get(key) is None
                else key + ": " + description(criteria[key])
                for key in keys
            ]
        if len(options) > len(ids):
            raise TooManyOptions("Model supports at most 64 single-token alternatives")
        suffix = "\nQuestion: " + safe_data(question["instructions"]) + "\nOptions:\n"
        suffix += "".join(
            labels[i] + ": " + safe_data(option) + "\n"
            for i, option in enumerate(options)
        )
        suffix += "Return the correct letter label." + boundary
        tokens = prefix_ids + tokenizer.encode(suffix, add_special_tokens=False)
        compiled.append((qid, question, tokens, ids[: len(options)]))
    return compiled


def decode_answer(question, scores, temperature=1.0):
    answer = raw_answer(question, scores, temperature)
    if "confidence" in answer:
        k = len(scores)
        answer["confidence"] = min(
            1.0, max(0.0, (k * answer["confidence"] - 1) / (k - 1))
        )
    for key in ("noul", "confidence", "score"):
        if key in answer:
            answer[key] = round(answer[key], 4)
    if "probabilities" in answer:
        answer["probabilities"] = {
            key: round(value, 4) for key, value in answer["probabilities"].items()
        }
    return answer
