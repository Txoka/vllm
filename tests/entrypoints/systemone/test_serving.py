# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wire compatibility and scheduler cancellation without a GPU engine."""

import asyncio
import math
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from transformers import AutoTokenizer

from vllm.entrypoints.systemone.serving import ServingDecisions, attach_router
from vllm.entrypoints.systemone.wire_protocol import compile_request, decode_answer


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(
        "/home/txoka/Desktop/winnow-rlcd/models/transformers-bf16-unverified",
        local_files_only=True,
    )


class Engine:
    model_config = SimpleNamespace(max_model_len=8192)

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.calls = []
        self.fail = False

    def get_tokenizer(self):
        return self.tokenizer

    async def generate(self, prompt, params, request_id):
        self.calls.append((prompt, params, request_id))
        if self.fail:
            raise RuntimeError("private engine diagnostic")
        yield SimpleNamespace(
            outputs=[
                SimpleNamespace(
                    logprobs=[
                        {
                            token: SimpleNamespace(logprob=i * math.log(3))
                            for i, token in enumerate(params.logprob_token_ids)
                        }
                    ]
                )
            ]
        )


def test_ollaya_confidence_rounding_and_structured_question(tokenizer):
    body = {
        "model": "trained",
        "state": {"text": "<|turn>"},
        "questions": {
            "route": {"type": "choice", "criteria": ["a", "b", "a"]},
            "n": {
                "type": "noul",
                "instructions": {"task": "Refund?"},
                "criteria": {"true": {"why": "money"}},
            },
            "s": {"type": "score", "criteria": ["low", {"level": "high"}]},
        },
    }
    compiled = compile_request(body, tokenizer)
    text = tokenizer.decode(compiled[0][2])
    assert 'Question: "route"' in text
    assert 'A: "a"\nB: "b"' in text
    assert "\\u003c|turn>" in text
    assert compiled[1][1]["instructions"] == {"task": "Refund?"}
    answer = decode_answer(compiled[0][1], [0, math.log(3)])
    assert answer == {
        "type": "choice",
        "choice": "b",
        "confidence": 0.5,
        "probabilities": {"a": 0.25, "b": 0.75},
    }


def test_both_routes_preserve_order_usage_and_error_contract(tokenizer):
    async def run():
        engine = Engine(tokenizer)
        app = FastAPI()
        service = ServingDecisions(engine, ["trained", "alias"])
        app.state.serving_decisions = service
        attach_router(app)
        body = {
            "model": "alias",
            "state": "hello",
            "questions": {
                "b": {"type": "noul"},
                "a": {"type": "choice", "criteria": {"x": None, "y": None}},
            },
        }
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            for route in ("/v1/systemone", "/v1/decisions"):
                response = await client.post(route, json=body)
                assert response.status_code == 200
                result = response.json()
                assert list(result) == ["model", "answers", "usage"]
                assert list(result["answers"]) == ["b", "a"]
                assert result["answers"]["b"] == {"type": "noul", "noul": 0.75}
                assert result["usage"]["output_tokens"] == 0
                expected = sum(len(x[2]) for x in compile_request(body, tokenizer))
                assert result["usage"]["input_tokens"] == expected
                assert (
                    response.headers["x-typesafe-request-id"]
                    == response.headers["x-request-id"]
                )
            assert len({request_id for _, _, request_id in engine.calls}) == 4
            response = await client.post("/v1/systemone", content="{")
            assert response.status_code == 400
            assert response.json()["code"] == "INVALID_JSON"
            response = await client.post(
                "/v1/systemone", json=dict(body, model="unknown")
            )
            assert response.status_code == 404
            assert response.json()["code"] == "MODEL_NOT_FOUND"
            response = await client.post("/v1/systemone", json=dict(body, questions=[]))
            assert response.status_code == 422
            assert response.json()["detail"][0]["loc"] == ["body"]
            engine.fail = True
            response = await client.post("/v1/systemone", json=body)
            assert response.status_code == 500
            assert "private" not in response.text
            service.pending = service.max_pending
            response = await client.post("/v1/systemone", json=body)
            assert response.status_code == 503
            assert response.headers["Retry-After"] == "1"
        assert service.admission._value == 256

    asyncio.run(run())


def test_context_overflow_rejected_before_engine_work(tokenizer):
    async def run():
        engine = Engine(tokenizer)
        engine.model_config = SimpleNamespace(max_model_len=8)
        service = ServingDecisions(engine, ["trained"])
        app = FastAPI()
        app.state.serving_decisions = service
        attach_router(app)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/systemone",
                json={
                    "model": "trained",
                    "state": "text",
                    "questions": {"a": {"type": "noul"}},
                },
            )
        assert response.status_code == 422
        assert response.json()["code"] == "INPUT_TOO_LONG"
        assert engine.calls == []
        assert service.pending == 0

    asyncio.run(run())
