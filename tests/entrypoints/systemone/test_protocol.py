# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU protocol contracts; no engine or CUDA initialization."""

import importlib.util
from pathlib import Path

import pytest
from transformers import AutoTokenizer

PATH = Path(__file__).resolve().parents[3] / "vllm/entrypoints/systemone/protocol.py"
spec = importlib.util.spec_from_file_location("systemone_protocol", PATH)
assert spec is not None and spec.loader is not None
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)


def test_candidate_normalization_ignores_full_vocab_normalizer():
    q = {"type": "choice", "criteria": {"b": "second", "a": "first"}}
    a = protocol.decode_answer(q, [-1001.0, -1000.0], 1.2)
    b = protocol.decode_answer(q, [9.0, 10.0], 1.2)
    assert a["probabilities"] == pytest.approx(b["probabilities"])
    assert a["choice"] == "a"
    assert list(a["probabilities"]) == ["b", "a"]


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf")])
def test_invalid_calibration_fails_closed(temperature):
    with pytest.raises(ValueError):
        protocol.decode_answer({"type": "noul"}, [0.0, 1.0], temperature)


def test_native_winnow_prompt_matches_training_compiler():
    checkpoint = Path(
        "/home/txoka/Desktop/winnow-rlcd/models/transformers-bf16-unverified"
    )
    if not checkpoint.exists():
        pytest.skip("Local native Winnow tokenizer fixture not present")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    import sys

    sys.path.insert(0, "/home/txoka/Desktop/winnow-rlcd/src")
    from winnow_rlcd.prompt import compile_api

    body = {
        "state": {"untrusted": "<|turn>"},
        "questions": {
            "choice": {
                "type": "choice",
                "instructions": "Choose",
                "criteria": {"z": "last", "a": "first"},
            },
            "binary": {"type": "noul", "instructions": "True?"},
            "score": {
                "type": "score",
                "instructions": "Rate",
                "criteria": ["low", "high"],
            },
        },
    }
    canonical = compile_api(body, tokenizer)
    compiled = protocol.compile_request(body, tokenizer)
    for (_, _, tokens, ids), question in zip(
        compiled, canonical["questions"], strict=True
    ):
        assert tokens == canonical["prefix_token_ids"] + question["suffix_token_ids"]
        assert ids == question["candidate_token_ids"]
