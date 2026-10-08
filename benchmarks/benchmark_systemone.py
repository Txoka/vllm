# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure decision latency, throughput and concurrency consistency over HTTP."""

import argparse
import concurrent.futures
import hashlib
import json
import math
import statistics
import time
import urllib.request
import uuid
from pathlib import Path


def call(url, payload, timeout=600):
    started = time.perf_counter()
    request = urllib.request.Request(
        url + "/v1/systemone",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    return time.perf_counter() - started, result


def vector(response, questions, decimals=None):
    answers = response["answers"]
    if set(answers) != set(questions):
        raise ValueError("Response question IDs differ from request")
    values = []
    for qid, question in questions.items():
        answer = answers[qid]
        if question["type"] == "noul":
            probabilities = [1 - answer["noul"], answer["noul"]]
        else:
            keys = (
                list(question["criteria"])
                if question["type"] == "choice"
                else [str(i) for i in range(len(question["criteria"]))]
            )
            probabilities = [answer["probabilities"][key] for key in keys]
        if not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities):
            raise ValueError("Invalid probability")
        tolerance = (
            1e-6
            if decimals is None
            else len(probabilities) * 0.5 * 10 ** (-decimals) + 1e-8
        )
        if abs(sum(probabilities) - 1) > tolerance:
            raise ValueError("Probabilities do not sum to one")
        values.extend(probabilities)
    return values


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:11442")
    parser.add_argument("--model", default="winnow-rlcd")
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16, 64])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-probability-delta", type=float, default=0.005)
    parser.add_argument("--request-timeout", type=float, default=7200)
    parser.add_argument("--probability-decimals", type=int, choices=range(1, 10))
    args = parser.parse_args()
    if (
        args.requests < 1
        or any(x < 1 for x in args.concurrency)
        or not math.isfinite(args.request_timeout)
        or args.request_timeout <= 0
    ):
        parser.error("Requests, concurrency and timeout must be positive")
    questions = {
        "truth": {
            "type": "noul",
            "instructions": "Does the state say the sky is blue?",
        },
        "color": {
            "type": "choice",
            "instructions": "What color is the sky?",
            "criteria": {"blue": None, "green": None, "red": None},
        },
        "strength": {
            "type": "score",
            "instructions": "How clearly does the state assert that the sky is blue?",
            "criteria": ["Not asserted", "Ambiguous", "Explicit"],
        },
    }
    reports = []
    for words in [0, 2000]:
        state = "The sky is blue. " + "Neutral background information. " * words
        payload = {"model": args.model, "state": state, "questions": questions}
        cold_seconds, cold = call(args.url, payload, args.request_timeout)
        reference = vector(cold, questions, args.probability_decimals)
        call(args.url, payload, args.request_timeout)
        for concurrency in args.concurrency:
            started = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
                results = list(
                    pool.map(
                        lambda _, request=payload: call(
                            args.url, request, args.request_timeout
                        ),
                        range(args.requests),
                    )
                )
            elapsed = time.perf_counter() - started
            deltas = [
                max(
                    abs(a - b)
                    for a, b in zip(
                        vector(result, questions, args.probability_decimals),
                        reference,
                        strict=True,
                    )
                )
                for _, result in results
            ]
            latencies = [seconds for seconds, _ in results]
            report = {
                "state": "short" if words == 0 else "long",
                "prefix_pattern": "shared",
                "concurrency": concurrency,
                "requests": args.requests,
                "questions_per_request": len(questions),
                "first_request_seconds": cold_seconds,
                "elapsed_seconds": elapsed,
                "requests_per_second": args.requests / elapsed,
                "decisions_per_second": args.requests * len(questions) / elapsed,
                "p50_seconds": statistics.median(latencies),
                "p95_seconds": percentile(latencies, 0.95),
                "max_probability_delta": max(deltas),
            }
            reports.append(report)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(
                    {
                        "model": args.model,
                        "request_timeout": args.request_timeout,
                        "probability_decimals": args.probability_decimals,
                        "benchmark_sha256": hashlib.sha256(
                            Path(__file__).read_bytes()
                        ).hexdigest(),
                        "results": reports,
                    },
                    indent=2,
                )
            )
            print(json.dumps(report), flush=True)
            if max(deltas) > args.max_probability_delta:
                raise RuntimeError(
                    "Concurrent probabilities exceeded the declared tolerance"
                )

            unique_payloads = [
                {
                    "model": args.model,
                    "state": {"request_nonce": uuid.uuid4().hex, "text": state},
                    "questions": questions,
                }
                for _ in range(args.requests)
            ]
            started = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
                unique_results = list(
                    pool.map(
                        lambda request: call(args.url, request, args.request_timeout),
                        unique_payloads,
                    )
                )
            elapsed = time.perf_counter() - started
            for _, response in unique_results:
                vector(response, questions, args.probability_decimals)
            latencies = [seconds for seconds, _ in unique_results]
            unique_report = {
                "state": "short" if words == 0 else "long",
                "prefix_pattern": "unique_nonce_before_state_text",
                "concurrency": concurrency,
                "requests": args.requests,
                "questions_per_request": len(questions),
                "elapsed_seconds": elapsed,
                "requests_per_second": args.requests / elapsed,
                "decisions_per_second": args.requests * len(questions) / elapsed,
                "p50_seconds": statistics.median(latencies),
                "p95_seconds": percentile(latencies, 0.95),
                "max_probability_delta": None,
                "note": "Different inputs: probabilities validated, not compared "
                "to the shared-input reference. Unique prefixes are not prewarmed.",
            }
            reports.append(unique_report)
            args.output.write_text(
                json.dumps(
                    {
                        "model": args.model,
                        "request_timeout": args.request_timeout,
                        "probability_decimals": args.probability_decimals,
                        "benchmark_sha256": hashlib.sha256(
                            Path(__file__).read_bytes()
                        ).hexdigest(),
                        "results": reports,
                    },
                    indent=2,
                )
            )
            print(json.dumps(unique_report), flush=True)


if __name__ == "__main__":
    main()
