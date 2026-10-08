# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ollaya's text decision wire contract, independent of the weight loader."""

import asyncio
import json
import math
import uuid

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from vllm.entrypoints.systemone.schema import normalize_request
from vllm.entrypoints.systemone.wire_protocol import (
    TooManyOptions,
    compile_request,
    decode_answer,
)
from vllm.sampling_params import SamplingParams


class ServingDecisions:
    """Score independently scheduled questions through an existing engine."""

    def __init__(self, engine, model_names, temperature=1.0, max_pending=512):
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Decision temperature must be finite and positive")
        self.engine = engine
        self.model_names = model_names
        self.temperature = temperature
        self.tokenizer = engine.get_tokenizer()
        self.max_model_len = engine.model_config.max_model_len
        self.admission = asyncio.Semaphore(256)
        self.max_pending = max_pending
        self.pending = 0

    async def score(self, qid, question, tokens, candidate_ids):
        async with self.admission:
            params = SamplingParams(
                temperature=0,
                max_tokens=1,
                detokenize=False,
                logprob_token_ids=candidate_ids,
                allowed_token_ids=candidate_ids,
            )
            final = None
            async for output in self.engine.generate(
                {"prompt_token_ids": tokens}, params, uuid.uuid4().hex
            ):
                final = output
            if final is None or not final.outputs or not final.outputs[0].logprobs:
                raise RuntimeError("Engine returned no candidate scores")
            returned = final.outputs[0].logprobs[0]
            if any(token not in returned for token in candidate_ids):
                raise RuntimeError("Engine omitted a candidate score")
            scores = [returned[token].logprob for token in candidate_ids]
            return qid, decode_answer(question, scores, self.temperature)

    async def decide(self, request):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        headers = {
            "x-request-id": request_id,
            "x-typesafe-request-id": request_id,
        }

        def error(status, code, message):
            body = {"error": message, "code": code}
            if status == 422:
                body["detail"] = [
                    {"loc": ["body"], "msg": message, "type": "value_error"}
                ]
            return JSONResponse(body, status_code=status, headers=headers)

        if self.pending >= self.max_pending:
            response = error(503, "QUEUE_FULL", "Decision request queue is full")
            response.headers["Retry-After"] = "1"
            return response
        self.pending += 1
        tasks = []
        try:
            chunks = bytearray()
            async for chunk in request.stream():
                if len(chunks) + len(chunk) > 8 * 1024 * 1024:
                    return error(413, "REQUEST_TOO_LARGE", "Request exceeds 8 MiB")
                chunks.extend(chunk)
            try:
                body = json.loads(chunks)
            except (ValueError, UnicodeError):
                return error(400, "INVALID_JSON", "Invalid JSON request")
            try:
                body = normalize_request(body)
                if body["model"] not in self.model_names:
                    return error(404, "MODEL_NOT_FOUND", "Unknown decision model")
                compiled = compile_request(body, self.tokenizer)
            except TooManyOptions as exc:
                return error(422, "TOO_MANY_OPTIONS", str(exc))
            except (ValueError, TypeError, KeyError) as exc:
                return error(422, "INVALID_REQUEST", str(exc))
            if any(
                len(tokens) + 1 > self.max_model_len for _, _, tokens, _ in compiled
            ):
                return error(422, "INPUT_TOO_LONG", "Decision exceeds model context")
            tasks = [asyncio.create_task(self.score(*item)) for item in compiled]
            await asyncio.gather(*tasks)
            return JSONResponse(
                {
                    "model": body["model"],
                    "answers": dict(task.result() for task in tasks),
                    "usage": {
                        "input_tokens": sum(len(item[2]) for item in compiled),
                        "output_tokens": 0,
                    },
                },
                headers=headers,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            return error(500, "INFERENCE_FAILED", "Decision inference failed")
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            self.pending -= 1


def attach_router(app: FastAPI):
    router = APIRouter()

    @router.post("/v1/systemone")
    @router.post("/v1/decisions")
    async def decide(request: Request):
        return await request.app.state.serving_decisions.decide(request)

    app.include_router(router)


def init_state(engine, state, args):
    if engine.model_config.logprobs_mode != "raw_logprobs":
        raise ValueError("Decision scoring requires --logprobs-mode raw_logprobs")
    names = args.served_model_name or [args.model]
    state.serving_decisions = ServingDecisions(engine, names, args.decision_temperature)
