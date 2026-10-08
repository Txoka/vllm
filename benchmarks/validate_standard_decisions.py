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
        "10",
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
            (args.output / "COMPLETE.json").write_text(
                json.dumps(
                    {
                        "argv": argv,
                        "contract": "contract.json",
                        "format": "safetensors",
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
