# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate the standard decision endpoint against a live, separately started server."""

import argparse
import json
import math
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:11443")
    parser.add_argument("--model", default="winnow-rlcd")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    questions = {
        "truth": {"type": "noul", "instructions": "Is the sky blue?"},
        "color": {
            "type": "choice",
            "instructions": "What color is the sky?",
            "criteria": ["blue", "green", "red"],
        },
        "clarity": {
            "type": "score",
            "instructions": "How explicit is the claim?",
            "criteria": ["Absent", "Ambiguous", "Explicit"],
        },
    }
    body = {"model": args.model, "state": "The sky is blue.", "questions": questions}
    results = {}
    for route in ("/v1/systemone", "/v1/decisions"):
        request = urllib.request.Request(
            args.url + route,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            assert response.status == 200
            result = json.load(response)
            assert response.headers["x-request-id"]
        assert list(result["answers"]) == list(questions)
        assert result["usage"]["input_tokens"] > 0
        assert result["usage"]["output_tokens"] == 0
        for qid, answer in result["answers"].items():
            if answer["type"] == "noul":
                assert 0 <= answer["noul"] <= 1
                continue
            probabilities = list(answer["probabilities"].values())
            assert all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
            # Ollaya wire probabilities are rounded independently to four decimals.
            assert abs(sum(probabilities) - 1) <= len(probabilities) * 0.00005
            k = len(probabilities)
            confidence = (k * max(probabilities) - 1) / (k - 1)
            assert abs(answer["confidence"] - confidence) <= 0.0002
            if qid == "clarity":
                expected = sum(i * p for i, p in enumerate(probabilities))
                assert abs(answer["score"] - expected) <= 0.0003
        results[route] = result
    assert results["/v1/systemone"]["answers"] == results["/v1/decisions"]["answers"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"url": args.url, "results": results}, indent=2))


if __name__ == "__main__":
    main()
