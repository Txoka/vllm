# Experimental System One decision endpoint

This fork adds a text-only `/v1/systemone` prototype for trained Winnow decision
models. It schedules independent state/question sequences through AsyncLLM;
questions never attend to each other. Prefix caching can reuse matching blocks
across questions and concurrent requests. This is block-prefix reuse, not a claim
that every request prefills its state exactly once.

The protocol preserves option insertion order, verifies single-token labels,
requests scores for every candidate ID, and normalizes only over those candidates.
The engine must return `raw_logprobs`: the full-vocabulary normalization constant
cancels during candidate normalization. Sampling temperature zero controls the
unused one-token sample; decision calibration is a separate positive temperature.
It does not mean decision temperature zero. No generated explanations are used.

Initial model: real19 / 5% synthetic / MiCA r32 / clip25 / step2500. This has the
highest public accuracy of the four matching Q8 exports (74.1244%), but does not
Pareto-dominate original Winnow: typed ECE is worse. Use its dense original
checkpoint plus merged adapter, exported to native HF BF16 safetensors. The
checkpoint's readout scaling is folded into an untied output head and its softcap;
external decision temperature is one. Never reconstruct weights from Q8.

Supported vLLM loading formats include native safetensors, PyTorch `.bin`,
sharded-state and specialized streaming/tensorizer loaders. Quantized safetensors
such as supported AWQ/GPTQ/FP8 layouts are separate format/backend compatibility
choices. Q8_0 GGUF is not interchangeable with FP8 or int8 safetensors. Current
GGUF support is an experimental out-of-tree plugin and is not our serving target.

Example (GPU validation still pending):

```bash
.venv/bin/python -m vllm.entrypoints.systemone.api_server \
  --model /home/txoka/Desktop/winnow-rlcd/models/vllm/winnow-real19-synth05-step2500-bf16 \
  --load-format safetensors --dtype bfloat16 \
  --hf-overrides '{"head_dtype":"float32"}' \
  --enable-prefix-caching --max-model-len 8192 \
  --logprobs-mode raw_logprobs --decision-model-name winnow-rlcd
```

Before claiming support, verify tokenizer/prompt parity, cached and uncached
candidate scores, final-softcap/readout semantics, all public/typed benchmark
results and concurrency correctness. Then measure p50/p95 latency and decisions
per second at concurrency1/4/16/64 with short/long shared states. GPU validation
must run serially after the authorized Qwen experiment, never alongside training.

Runtime base: official vLLM `v0.31.0`, revision
`db9527a46873454610df6dbedf79a36d6bf1a7f6`. The original main-based prototype
remains on `feat/system1-decisions`; this release-matched implementation is on
`feat/system1-decisions-v031`.
No upstream PR is requested. This is a prototype, not validated production support.

References:

- <https://docs.vllm.ai/en/stable/api/vllm/config/load/>
- <https://github.com/vllm-project/vllm/blob/main/docs/features/quantization/gguf.md>
- `vllm/sampling_params.py` (`logprob_token_ids`)
- `vllm/config/model.py` (`logprobs_mode`)

## Local validation procedure

The development GPU is an RTX 4070 with 12 GB VRAM. The full BF16 export
requires CPU offload on this machine; throughput with offload is a hardware
constraint, not representative of a sufficiently sized production GPU. Start
with `--cpu-offload-gb 10 --gpu-memory-utilization 0.80 --enforce-eager` and a
short context for correctness. Check memory before increasing context or batch
capacity. Alternative FP8/int8 quantization needs its own parity evaluation;
it is not equivalent to the original GGUF Q8_0.

After the serial model correctness gate and public/typed benchmark:

```bash
.venv/bin/python benchmarks/benchmark_systemone.py \
  --requests 64 --concurrency 1 4 16 64 \
  --output /tmp/winnow-vllm-concurrency.json
```

This records short/long shared-state request p50/p95, decisions per second,
and candidate-probability drift relative to a sequential request. Run with
prefix caching both enabled and disabled in separately labeled output files.
The first-request timing is not guaranteed to represent an empty engine cache.
Concurrent-probability tolerance defaults to 0.005 on the 0–1 scale; report
actual deltas rather than describing tolerance acceptance as exact equality.

The client also tests unique request nonces before the long state text, reducing
inter-request prefix reuse. Those requests validate probability structure but
do not assert equality to the shared-input reference, since their inputs differ.
This complements the warmed shared-prefix case.

## Precision diagnostic on the development GPU

The initial 13-question native CPU/GPU check found a maximum probability
absolute difference of 0.01935 with a BF16 head. Using the supported
`--hf-overrides '{"head_dtype":"float32"}'` option and a matched FP32-head CPU
reference reduced this to 0.00991. This is within the declared 0.01 gate, but is
not exact parity. Report residual differences, including backend/batch effects.
The CPU model alone changed by up to 0.02532 when only its head and softcap
computation changed from BF16 to FP32. Weight values and input tokens were held
fixed. FP32 head accumulation does not require a duplicate FP32 head weight on
CUDA: vLLM uses `torch.mm(..., out_dtype=torch.float32)`.

On the 12 GB GPU, use `--max-num-batched-tokens 2048 --enable-chunked-prefill`
with context 8192. An 8192-token prefill batch left only 0.18 GiB of KV budget
and failed startup; reducing the activation batch allowed startup with about
0.93 GiB of KV cache. Model context and precision were preserved. Full public,
typed and concurrency results remain pending at this documentation checkpoint.

## Reproduce the tested Linux x86-64 runtime

This branch uses the official v0.31.0 native wheel with release-matched source.
The extra fork commits change Python endpoint/benchmark code only. Do not use
an arbitrary main-branch source checkout with release native binaries.

```bash
uv --no-config venv --python 3.12 .venv
uv --no-config pip install --python .venv/bin/python vllm==0.31.0
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_LOCATION=https://files.pythonhosted.org/packages/e1/98/841d1328827dc082492fcbb0f57bdf6a9d35aa0df52bd47b75be3f504804/vllm-0.31.0-cp38-abi3-manylinux_2_28_x86_64.whl \
uv --no-config pip install --python .venv/bin/python -e .
```

Observed versions: PyTorch2.13.0, Transformers5.17.0, Triton3.7.1 and
FlashInfer0.7.0.post1. CPU offload is a local capacity workaround; the native
safetensors export can run entirely on a larger GPU without changing the API.
