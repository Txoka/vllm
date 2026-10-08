# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validation for Ollaya-compatible text decision requests."""


def content(value, nullable=False):
    return isinstance(value, (str, dict, list)) or (nullable and value is None)


def normalize_request(body):
    """Validate Ollaya JSON content and apply its question-id fallback."""
    if not isinstance(body, dict):
        raise ValueError("Request must be a JSON object")
    if not isinstance(body.get("model"), str) or not body["model"]:
        raise ValueError("model must be nonempty text")
    if not content(body.get("state")):
        raise ValueError("state must be text, an object or an array")
    questions = body.get("questions")
    if not isinstance(questions, dict) or not 1 <= len(questions) <= 256:
        raise ValueError("Expected 1–256 named questions")
    result = dict(body, questions={})
    for qid, original in questions.items():
        if not isinstance(qid, str) or not qid or not isinstance(original, dict):
            raise ValueError("Questions must be named objects")
        question = dict(original)
        instruction = question.get("instructions")
        if not content(instruction, nullable=True):
            raise ValueError("instructions must be text, an object or an array")
        question["instructions"] = qid if instruction is None else instruction
        kind = question.get("type")
        criteria = question.get("criteria")
        if kind == "noul":
            criteria = {} if criteria is None else criteria
            if not isinstance(criteria, dict) or set(criteria) - {"false", "true"}:
                raise ValueError("noul criteria keys must be false and true")
            values = list(criteria.values())
        elif kind == "choice":
            if isinstance(criteria, list):
                if not all(isinstance(label, str) for label in criteria):
                    raise ValueError("choice labels must be text")
                criteria = dict.fromkeys(criteria)
            if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 255:
                raise ValueError("choice requires 2–255 distinct criteria")
            if any(not key for key in criteria):
                raise ValueError("choice labels must be nonempty")
            values = list(criteria.values())
        elif kind == "score":
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                raise ValueError("score requires 2–10 ordered levels")
            values = criteria
        else:
            raise ValueError("Unknown question type")
        if not all(content(value, nullable=kind != "score") for value in values):
            raise ValueError("Invalid criterion description")
        question["criteria"] = criteria
        result["questions"][qid] = question
    return result
