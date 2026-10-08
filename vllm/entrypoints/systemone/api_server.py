# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental Ollaya-compatible text decision endpoint on AsyncLLM."""

import asyncio
import math
import uuid

import uvicorn
from fastapi import FastAPI, HTTPException

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.systemone.protocol import compile_request, decode_answer
from vllm.sampling_params import SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.engine.async_llm import AsyncLLM


def create_app(engine, model_name, temperature=1.0, max_model_len=8192):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    app = FastAPI(title="vLLM System One (experimental)")
    tokenizer = engine.get_tokenizer()
    admission = asyncio.Semaphore(256)

    @app.post("/v1/systemone")
    async def decide(body: dict):
        if body.get("model", model_name) != model_name:
            raise HTTPException(404, "Unknown decision model")
        try:
            compiled = compile_request(body, tokenizer)
            if any(len(tokens) > max_model_len for _, _, tokens, _ in compiled):
                raise ValueError(
                    "Decision exceeds model context; truncation is disabled"
                )
        except (ValueError, TypeError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from exc

        async def score(qid, question, tokens, candidate_ids):
            async with admission:
                params = SamplingParams(
                    temperature=0,
                    max_tokens=1,
                    detokenize=False,
                    logprob_token_ids=candidate_ids,
                    allowed_token_ids=candidate_ids,
                )
                final = None
                async for output in engine.generate(
                    {"prompt_token_ids": tokens}, params, uuid.uuid4().hex
                ):
                    final = output
                if final is None or not final.outputs[0].logprobs:
                    raise RuntimeError("Engine returned no candidate scores")
                returned = final.outputs[0].logprobs[0]
                if any(token not in returned for token in candidate_ids):
                    raise RuntimeError("Engine omitted a requested candidate score")
                scores = [returned[token].logprob for token in candidate_ids]
                return qid, decode_answer(question, scores, temperature)

        tasks = [asyncio.create_task(score(*question)) for question in compiled]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        return {
            "model": model_name,
            "answers": dict(task.result() for task in tasks),
            "usage": {
                "input_tokens": sum(len(x[2]) for x in compiled),
                "output_tokens": 0,
            },
        }

    return app


def main():
    parser = FlexibleArgumentParser(description=__doc__)
    parser = AsyncEngineArgs.add_cli_args(parser)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11442)
    parser.add_argument("--decision-model-name", default="winnow-rlcd")
    parser.add_argument("--decision-temperature", type=float, default=1.0)
    args = parser.parse_args()
    if args.logprobs_mode != "raw_logprobs":
        parser.error("Decision scoring requires --logprobs-mode raw_logprobs")
    args.max_logprobs = max(args.max_logprobs, 64)
    if args.enable_prefix_caching is None:
        args.enable_prefix_caching = True
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs.from_cli_args(args))
    app = create_app(
        engine,
        args.decision_model_name,
        args.decision_temperature,
        engine.model_config.max_model_len,
    )
    try:
        uvicorn.run(app, host=args.host, port=args.port)
    finally:
        engine.shutdown(timeout=30)


if __name__ == "__main__":
    main()
