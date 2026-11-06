"""Reference solutions for Level 3 notebook 16. Graded by tests/test_solutions.py."""

from __future__ import annotations

import re


def _norm(v) -> str:
    return re.sub(r"\s+", " ", str(v).strip().lower())


def ex1(pred_dict, gold_dict) -> float:
    predicted = {k: _norm(v) for k, v in pred_dict.items() if v is not None and _norm(v) != ""}
    gold = {k: _norm(v) for k, v in gold_dict.items()}
    if not predicted and not gold:
        return 1.0
    tp = sum(1 for k, v in predicted.items() if k in gold and gold[k] == v)
    if tp == 0:
        return 0.0
    precision, recall = tp / len(predicted), tp / len(gold)
    return 2 * precision * recall / (precision + recall)


def ex2(judgments_ab, judgments_ba) -> float:
    same = [a.strip().upper() == b.strip().upper() for a, b in zip(judgments_ab, judgments_ba)]
    return sum(same) / len(same)


_RULES = [
    r"\b(ignore|disregard|forget)\b[^.]{0,40}\b(instructions?|rules|prompt|above|told|everything)\b",
    r"\b(you are now|pretend (you are|to be)|act as|from now on)\b",
    r"\[/?system\]|<\|im_start\|>|#{3}|\bsystem override\b|\bnew (policy|instructions)\b",
    r"\b(system prompt|original instructions|hidden prompt|repeat them)\b",
    r"\b(idiots?|morons?|clowns?|garbage|pathetic|losers?|dumbest|incompetent|shut up)\b",
]


def ex3(text: str) -> bool:
    return any(re.search(p, text, flags=re.I) for p in _RULES)
