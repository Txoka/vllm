# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Activation-calibrate Gemma4 linears and pack INT embeddings for vLLM."""

import argparse
import hashlib
import json
import shutil
import subprocess
from importlib.metadata import version
from pathlib import Path

import torch
from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQMapping, AWQModifier
from safetensors import safe_open
from transformers import AutoTokenizer, Gemma4ForCausalLM


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bits", type=int, choices=[4, 8], required=True)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--min-free-mib", type=int, default=9500)
    parser.add_argument("--allow-compute-pid", type=int, action="append", default=[])
    args = parser.parse_args()
    assert args.ready_file.is_file(), "Ollaya release verification must finish first"
    assert not args.output.exists(), "Use a fresh output directory"
    assert torch.cuda.is_available(), (
        "Full-model conversion requires the CUDA calibration device"
    )
    free = int(
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
    )
    assert free >= args.min_free_mib, f"GPU in use: only {free} MiB free"
    active = {
        int(x)
        for x in subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True,
        ).split()
    }
    assert active <= set(args.allow_compute_pid), (
        f"Unexpected GPU processes: {active - set(args.allow_compute_pid)}"
    )
    export = json.loads((args.model / "EXPORT.json").read_text())
    for name, expected in export["output_shards_sha256"].items():
        assert sha(args.model / name) == expected, f"Source weights changed: {name}"
    calibration_manifest = json.loads(
        args.calibration.with_suffix(".manifest.json").read_text()
    )
    assert sha(args.calibration) == calibration_manifest["sha256"]
    rows = [json.loads(line) for line in args.calibration.read_text().splitlines()]
    assert {row["type"] for row in rows} == {"noul", "choice", "score"}
    dataset = Dataset.from_list(
        [
            {"input_ids": row["input_ids"], "attention_mask": row["attention_mask"]}
            for row in rows
        ]
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    torch.manual_seed(42)
    model = Gemma4ForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="sdpa",
    )
    model.eval()
    model.config.use_cache = False
    weights = {
        "num_bits": args.bits,
        "type": "int",
        "symmetric": True,
        "strategy": "group",
        "group_size": args.group_size,
    }
    # Gemma4 has q/k normalization and shared KV layers. Smooth the dense MLP
    # paths only; remaining linears use calibrated quantizer scales without an
    # unsafe attention rescaling. Embeddings use data-free INT quantization.
    recipe = [
        AWQModifier(
            mappings=[
                AWQMapping(
                    "re:.*pre_feedforward_layernorm$",
                    ["re:.*gate_proj$", "re:.*up_proj$"],
                ),
                AWQMapping("re:.*up_proj$", ["re:.*down_proj$"]),
            ]
        ),
        QuantizationModifier(
            config_groups={
                "linears": {
                    "targets": ["Linear"],
                    "weights": weights,
                    "format": "pack-quantized",
                },
                "embeddings": {
                    "targets": ["Embedding"],
                    "weights": weights,
                    "format": "pack-quantized",
                },
            }
        ),
    ]
    oneshot(
        model=model,
        processor=tokenizer,
        dataset=dataset,
        recipe=recipe,
        num_calibration_samples=len(rows),
        max_seq_length=max(len(row["input_ids"]) for row in rows),
        pipeline="sequential",
    )
    model.config.use_cache = True
    args.output.mkdir(parents=True)
    model.save_pretrained(
        args.output, save_compressed=True, quantization_format="pack-quantized"
    )
    tokenizer.save_pretrained(args.output)
    for name in [
        "LICENSE",
        "NOTICE",
        "ADAPTER_LICENSE",
        "ADAPTER_NOTICE",
        "ADAPTER_DATA_LICENSES.json",
        "ADAPTER_MODEL_CARD.md",
    ]:
        if (args.model / name).exists():
            shutil.copy2(args.model / name, args.output / name)
    packed = set()
    for path in args.output.glob("*.safetensors"):
        with safe_open(path, framework="pt", device="cpu") as tensors:
            packed.update(tensors.keys())
    assert "model.embed_tokens_per_layer.weight_packed" in packed
    assert "model.embed_tokens.weight_packed" in packed
    assert "lm_head.weight_packed" in packed
    assert any(".mlp.down_proj.weight_packed" in key for key in packed)
    report = {
        "source_export_sha256": sha(args.model / "EXPORT.json"),
        "source_export": export,
        "algorithm": "AWQ MLP balancing; groupwise INT linears and embeddings",
        "bits": args.bits,
        "group_size": args.group_size,
        "container": "compressed-tensors pack-quantized safetensors",
        "external_temperature": 1.0,
        "calibration": calibration_manifest,
        "packages": {
            name: version(name)
            for name in ["torch", "transformers", "llmcompressor", "compressed-tensors"]
        },
        "output_sha256": {
            path.name: sha(path) for path in sorted(args.output.glob("*.safetensors"))
        },
        "gpu_validation_complete": False,
    }
    (args.output / "AWQ_EXPORT.json").write_text(json.dumps(report, indent=2))
    print("EXPORTED", args.output, flush=True)


if __name__ == "__main__":
    main()
