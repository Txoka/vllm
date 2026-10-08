# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure decision latency, throughput and concurrency consistency over HTTP."""

import argparse
import concurrent.futures
import json
import math
import statistics
import time
import urllib.request
from pathlib import Path


def call(url, payload):
    started = time.perf_counter()
    request = urllib.request.Request(
        url + "/v1/systemone",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.load(response)
    return time.perf_counter() - started, result


def vector(response, questions):
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
        if abs(sum(probabilities) - 1) > 1e-6:
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
    args = parser.parse_args()
    if args.requests < 1 or any(x < 1 for x in args.concurrency):
        parser.error("Requests and concurrency must be positive")
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
        cold_seconds, cold = call(args.url, payload)
        reference = vector(cold, questions)
        call(args.url, payload)
        for concurrency in args.concurrency:
            started = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
                results = list(
                    pool.map(
                        lambda _, request=payload: call(args.url, request),
                        range(args.requests),
                    )
                )
            elapsed = time.perf_counter() - started
            deltas = [
                max(
                    abs(a - b)
                    for a, b in zip(vector(result, questions), reference, strict=True)
                )
                for _, result in results
            ]
            latencies = [seconds for seconds, _ in results]
            report = {
                "state": "short" if words == 0 else "long",
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
                json.dumps({"model": args.model, "results": reports}, indent=2)
            )
            print(json.dumps(report), flush=True)
            if max(deltas) > args.max_probability_delta:
                raise RuntimeError(
                    "Concurrent probabilities exceeded the declared tolerance"
                )


if __name__ == "__main__":
    main()
