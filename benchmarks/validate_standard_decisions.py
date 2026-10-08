# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serial GPU validation of the standard vLLM decision integration."""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-offload-gb", type=float, default=10)
    parser.add_argument("--quantized", action="store_true")
    parser.add_argument("--performance", action="store_true")
    parser.add_argument("--quality-script", type=Path)
    parser.add_argument("--quality-python", type=Path)
    parser.add_argument("--quality-name")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    fork = Path(__file__).resolve().parents[1]
    argv = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        args.model,
        "--served-model-name",
        "winnow-rlcd",
        "--port",
        "11443",
        "--decision-protocol",
        "winnow",
        "--decision-temperature",
        "1",
        "--load-format",
        "safetensors",
        "--dtype",
        "bfloat16",
        "--hf-overrides",
        '{"head_dtype":"float32"}',
        "--cpu-offload-gb",
        str(args.cpu_offload_gb),
        "--gpu-memory-utilization",
        "0.80",
        "--enforce-eager",
        "--max-model-len",
        "8192",
        "--max-num-seqs",
        "64",
        "--max-num-batched-tokens",
        "512",
        "--kv-cache-memory-bytes",
        "536870912",
        "--enable-chunked-prefill",
        "--enable-prefix-caching",
        "--logprobs-mode",
        "raw_logprobs",
        "--max-logprobs",
        "64",
    ]
    if args.quantized:
        index = argv.index("--hf-overrides")
        del argv[index : index + 2]
        assert args.cpu_offload_gb == 0, (
            "Quantized residency test must disable CPU offloading"
        )
        argv += ["--quantization", "compressed-tensors"]
    env = dict(os.environ, PYTHONNOUSERSITE="1")
    env.pop("PYTHONPATH", None)
    with (args.output / "server.log").open("ab", buffering=0) as log:
        server = subprocess.Popen(argv, env=env, stdout=log, stderr=log, cwd=fork)
        try:
            deadline = time.monotonic() + 900
            while True:
                if server.poll() is not None:
                    raise RuntimeError("Standard server exited during startup")
                try:
                    with urllib.request.urlopen(
                        "http://127.0.0.1:11443/health", timeout=2
                    ) as response:
                        if response.status == 200:
                            break
                except (OSError, urllib.error.URLError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("Standard server startup timed out")
                time.sleep(2)
            (args.output / "gpu-memory.csv").write_text(
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader",
                    ],
                    text=True,
                )
            )
            runtime = {
                "argv": argv,
                "cpu_offload_gb": args.cpu_offload_gb,
                "quantized": args.quantized,
                "fork_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=fork, text=True
                ).strip(),
            }
            (args.output / "runtime.json").write_text(json.dumps(runtime, indent=2))
            subprocess.run(
                [
                    sys.executable,
                    "benchmarks/check_decision_contract.py",
                    "--output",
                    str(args.output / "contract.json"),
                ],
                env=env,
                check=True,
            )
            if args.quality_script:
                assert args.quantized and args.quality_python and args.quality_name
                bits = json.loads((Path(args.model) / "AWQ_EXPORT.json").read_text())[
                    "bits"
                ]
                subprocess.run(
                    [
                        str(args.quality_python),
                        str(args.quality_script.resolve()),
                        "--model",
                        "winnow-rlcd",
                        "--name",
                        args.quality_name,
                        "--output",
                        str(args.output / "quality"),
                        "--url",
                        "http://127.0.0.1:11443",
                        "--export",
                        str(Path(args.model).resolve() / "AWQ_EXPORT.json"),
                        "--runtime-manifest",
                        str(args.output / "runtime.json"),
                        "--precision",
                        (
                            f"AWQ W{bits}A16 linears + INT{bits} embeddings; "
                            "compressed-tensors; no CPU offload"
                        ),
                        "--wire-rounded",
                    ],
                    env=env,
                    cwd=args.quality_script.resolve().parents[1],
                    check=True,
                )
            if args.performance:
                subprocess.run(
                    [
                        sys.executable,
                        "benchmarks/benchmark_systemone.py",
                        "--url",
                        "http://127.0.0.1:11443",
                        "--requests",
                        "64",
                        "--concurrency",
                        "1",
                        "4",
                        "16",
                        "64",
                        "--probability-decimals",
                        "4",
                        "--output",
                        str(args.output / "performance.json"),
                    ],
                    env=env,
                    cwd=fork,
                    check=True,
                )
            (args.output / "COMPLETE.json").write_text(
                json.dumps(
                    {
                        "argv": argv,
                        "contract": "contract.json",
                        "format": "compressed-tensors AWQ"
                        if args.quantized
                        else "safetensors",
                        "cpu_offload_gb": args.cpu_offload_gb,
                        "performance": args.performance,
                        "production_scale_claim": False,
                    },
                    indent=2,
                )
            )
        finally:
            server.terminate()
            server.wait(timeout=60)


if __name__ == "__main__":
    main()
