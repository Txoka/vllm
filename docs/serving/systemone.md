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

The client timeout defaults to 7200 seconds per request to permit queued long
requests on the CPU-offloaded development machine. Override with
`--request-timeout` for a production latency budget. Timeouts fail the run;
requests are not silently omitted or retried.

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

On the 12 GB GPU, an 8192-token prefill batch left only 0.18 GiB of KV
budget and failed startup. Reducing the activation batch to 2048 allowed
startup with about 0.93 GiB of KV cache, but the full public benchmark later
exhausted memory during an MLP activation allocation. Startup success alone
therefore does not establish sustained serving capacity.

The next validation uses `--max-num-batched-tokens 512`,
`--kv-cache-memory-bytes 536870912` and `--enable-chunked-prefill` while
preserving context 8192 and the model precision. The explicit KV allocation
exceeds the estimated 0.44 GiB required for one maximum-length request;
concurrent long requests may require scheduling/preemption rather than all
remaining resident. These are development-machine settings, not production
recommendations. Full public, typed and concurrency results remain pending.

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

## Standard server integration

The endpoint can also run inside the standard server, sharing its model engine,
model loader, authentication and lifecycle:

```bash
vllm serve /path/to/native-export --served-model-name winnow-rlcd \
  --decision-protocol winnow --decision-temperature 1 \
  --logprobs-mode raw_logprobs --max-logprobs 64
```

Both `/v1/systemone` and `/v1/decisions` accept the Ollaya text decision contract:
JSON state, named `noul`, `choice` and `score` questions, structured instructions
and ordered criteria. Confidence uses `(K * max_probability - 1) / (K - 1)`;
probabilities and expected ordinal scores are returned separately. Requests are
bounded to 8 MiB and 256 questions, with bounded admission and cancellation.
Model names must match a served alias. Model-specific embedded presets and image
inputs are not implemented by this text protocol.

The decision endpoint does not impose a weight serialization. Native safetensors
uses the normal loader; AWQ uses a compatible quantized checkpoint and the normal
AWQ backend. GGUF requires the experimental external
[vLLM GGUF plugin](https://github.com/vllm-project/vllm-gguf-plugin), which has not
yet been validated for this Winnow export. Loader support does not establish
architecture, quantization or numerical compatibility. The current protocol is
Winnow-specific; other architectures require their own prompt/token mapping.

Nine CPU protocol/HTTP tests pass, including structured criteria, aliases,
confidence semantics and pre-engine rejection. Live GPU validation of this
standard-server integration remains pending behind the ongoing serial scale
validation of the original prototype. Those active test files are unchanged.

Once the standard server is running, validate both aliases and rounded wire
semantics with:

```bash
.venv/bin/python benchmarks/check_decision_contract.py \
  --url http://127.0.0.1:11443 --model winnow-rlcd \
  --output /tmp/decision-contract.json
```

This client checks all three decision types, question ordering, usage and
confidence/expected-score consistency. It does not replace model-quality or
concurrency evaluation, and starting it does not start another model engine.

## AWQ residency validation

The next format checks use the same trained dense export with AWQ at W4A16 and
W8A16, stored as compressed-tensors safetensors. This is different from a legacy
AutoAWQ checkpoint: the pinned legacy loader accepts only four bits. The
compressed-tensors configuration reader accepts the tiny four/eight-bit exports;
full-model CUDA kernels and quality still need validation.

Gemma4 E4B's BF16 embedding tables occupy approximately 6.98 GB. Quantizing only
linear layers does not establish that the model fits a 12 GB GPU. The conversion
also packs the ordinary and per-layer embeddings with groupwise INT weights.
AWQ balancing targets the dense MLP paths; attention linears and lookup tables
use groupwise weight quantization. Both precisions use the same calibration
prompts and group size. No gradient training or benchmark fitting occurs.

`benchmarks/prepare_awq_calibration.py` selects 256 distinct fixed validation
states, with 192 real and 64 procedural questions, balanced within source/type
buckets. It requires all three primitive types and skips oversized prompts
without truncating them. The procedural portion supplies score questions absent
from the real portion of this validation selection. This calibration selection
does not change the model's training mixture.

Use an isolated quantizer environment for
`benchmarks/quantize_decision_awq.py`: the tested quantizer revision is
`vllm-project/llm-compressor@af7967973f9e0928ad8b25e05a79b1af97394bc4`, with
Torch 2.14.1, Transformers 5.18.0 and compressed-tensors 0.19.1a20261003.
The serving environment remains unchanged. Conversion verifies source weights
and calibration hashes, preserves attribution, explicitly requests packed
embedding storage, and records output hashes. The tiny CPU-only preflight
exported both precisions; this does not prove GPU serving support.

For each resulting model, run the standard endpoint with CPU offload disabled:

```bash
.venv/bin/python benchmarks/validate_standard_decisions.py \
  --model /path/to/quantized-model --output /path/to/results \
  --quantized --cpu-offload-gb 0 --performance
```

This checks both endpoint aliases and all three primitives, records GPU-memory
usage, and runs the 64-request concurrency1/4/16/64 matrix over short/long,
shared/unique inputs. Four-decimal wire rounding is declared explicitly; the
legacy benchmark's strict probability check remains its default. Optional
`--quality-script`, `--quality-python` and `--quality-name` arguments run the
reserved public/typed quality suite on the same live server before timing.
Full-model results are pending; do not infer accuracy or residency from file size
or startup alone.
