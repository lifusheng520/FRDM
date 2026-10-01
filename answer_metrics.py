"""Answer-level metrics for SOCRATES and 2WikiMultiHopQA."""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Iterable


def normalize_answer(text: str) -> str:
    """Apply the standard SQuAD/HotpotQA answer normalization."""

    lowered = str(text).lower()
    without_punctuation = "".join(
        character for character in lowered if character not in string.punctuation
    )
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def normalized_prefix_match(prediction: str, answer: str) -> bool:
    """Return whether a normalized completion starts with a complete answer."""

    predicted = normalize_answer(prediction)
    target = normalize_answer(answer)
    if not target:
        return False
    return predicted == target or predicted.startswith(target + " ")


def token_prf(prediction: str, answer: str) -> tuple[float, float, float]:
    """Return normalized token precision, recall, and F1."""

    normalized_prediction = normalize_answer(prediction)
    normalized_answer = normalize_answer(answer)
    predicted_tokens = normalized_prediction.split()
    answer_tokens = normalized_answer.split()
    special = {"yes", "no", "noanswer"}
    if (
        normalized_prediction in special or normalized_answer in special
    ) and predicted_tokens != answer_tokens:
        return 0.0, 0.0, 0.0
    common = Counter(predicted_tokens) & Counter(answer_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0, 0.0, 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(answer_tokens)
    f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def score_prediction(prediction: str, answers: Iterable[str]) -> dict[str, float]:
    """Score against aliases, retaining the alias with the highest F1."""

    candidates = [str(answer) for answer in answers if str(answer).strip()]
    if not candidates:
        raise ValueError("At least one non-empty reference answer is required.")
    exact_match = max(
        float(normalize_answer(prediction) == normalize_answer(answer))
        for answer in candidates
    )
    prefix_accuracy = max(
        float(normalized_prefix_match(prediction, answer)) for answer in candidates
    )
    precision, recall, f1 = max(
        (token_prf(prediction, answer) for answer in candidates),
        key=lambda values: (values[2], values[1], values[0]),
    )
    return {
        "exact_match": exact_match,
        "prefix_accuracy": prefix_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def aggregate_scores(records: Iterable[dict[str, object]]) -> dict[str, float | int]:
    """Macro-average saved per-example score dictionaries."""

    values = list(records)
    if not values:
        raise ValueError("Cannot aggregate an empty prediction collection.")
    names = ("exact_match", "prefix_accuracy", "precision", "recall", "f1")
    return {
        "count": len(values),
        **{
            name: sum(float(record[name]) for record in values) / len(values)
            for name in names
        },
    }
