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
  --enable-prefix-caching --max-model-len 8192 \
  --logprobs-mode raw_logprobs --decision-model-name winnow-rlcd
```

Before claiming support, verify tokenizer/prompt parity, cached and uncached
candidate scores, final-softcap/readout semantics, all public/typed benchmark
results and concurrency correctness. Then measure p50/p95 latency and decisions
per second at concurrency1/4/16/64 with short/long shared states. GPU validation
must run serially after the authorized Qwen experiment, never alongside training.

Upstream starting revision: `a4b14e57b6d5ad25a2215eff93d6f93ba05005df`.
No upstream PR is requested. This is a prototype, not validated production support.

References:

- <https://docs.vllm.ai/en/stable/api/vllm/config/load/>
- <https://github.com/vllm-project/vllm/blob/main/docs/features/quantization/gguf.md>
- `vllm/sampling_params.py` (`logprob_token_ids`)
- `vllm/config/model.py` (`logprobs_mode`)
