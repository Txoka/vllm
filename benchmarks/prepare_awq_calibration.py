# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare bounded, source-balanced Winnow prompts from reserved validation only."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

from transformers import AutoTokenizer

from vllm.entrypoints.systemone.wire_protocol import compile_request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--synthetic-samples", type=int, default=64)
    args = parser.parse_args()
    manifest = json.loads(args.validation.with_suffix(".manifest.json").read_text())
    source_sha = hashlib.sha256(args.validation.read_bytes()).hexdigest()
    assert source_sha == manifest["sha256"], "Validation selection changed"
    assert manifest.get("benchmark_overlap"), "Missing reserved benchmark overlap audit"
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    buckets = defaultdict(list)
    for line in args.validation.read_text().splitlines():
        row = json.loads(line)
        compiled = compile_request({**row["input"], "model": "winnow-rlcd"}, tokenizer)
        assert len(compiled) == 1
        tokens = compiled[0][2]
        if len(tokens) <= args.max_length:
            buckets[row["source"], row["type"]].append(
                {
                    "id": row["id"],
                    "question_id": row["question_id"],
                    "source": row["source"],
                    "type": row["type"],
                    "split_group": row["split_group"],
                    "input_ids": tokens,
                    "attention_mask": [1] * len(tokens),
                }
            )
    queues = {
        k: deque(
            sorted(
                v,
                key=lambda x: hashlib.sha256(
                    (x["id"] + ":" + x["question_id"]).encode()
                ).hexdigest(),
            )
        )
        for k, v in buckets.items()
    }
    selected = []
    groups = set()
    assert 0 <= args.synthetic_samples <= args.samples
    for kind, count in [
        ("real:", args.samples - args.synthetic_samples),
        ("synthetic:", args.synthetic_samples),
    ]:
        target = len(selected) + count
        keys = [key for key in sorted(queues) if key[0].startswith(kind)]
        while len(selected) < target:
            before = len(selected)
            for key in keys:
                queue = queues[key]
                while queue and queue[0]["split_group"] in groups:
                    queue.popleft()
                if queue:
                    row = queue.popleft()
                    groups.add(row["split_group"])
                    selected.append(row)
                    if len(selected) == target:
                        break
            if len(selected) == before:
                raise ValueError(
                    f"Insufficient distinct eligible {kind} validation states"
                )
    assert {x["type"] for x in selected} == {"noul", "choice", "score"}, (
        "Calibration must cover all three primitives"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    assert not args.output.exists(), "Use a new output to preserve calibration identity"
    args.output.write_text("".join(json.dumps(x) + "\n" for x in selected))
    report = {
        "algorithm": "deterministic source/type round-robin; one question per state",
        "purpose": (
            "AWQ calibration on fixed validation; no benchmark rows or gradient updates"
        ),
        "validation_sha256": source_sha,
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "samples": len(selected),
        "max_length": max(len(x["input_ids"]) for x in selected),
        "by_source": dict(Counter(x["source"] for x in selected)),
        "by_kind": dict(Counter(x["source"].split(":")[0] for x in selected)),
        "by_type": dict(Counter(x["type"] for x in selected)),
        "tokenizer": str(args.model),
        "validation_manifest": manifest,
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("samples", "max_length", "by_type")}))


if __name__ == "__main__":
    main()
