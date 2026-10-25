"""Verifiable rewards (RLVR): a program checks the answer, no learned reward model.

Each reward has the TRL signature  f(prompts, completions, answer, **kw) -> list[float]; `answer` is the
dataset column of the same name. Completions are strings (standard format) or lists of chat messages.
"""
from __future__ import annotations

import re

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _text(c) -> str:
    if isinstance(c, list):                         # conversational: [{"role": "assistant", "content": ...}]
        return c[-1].get("content", "") if c else ""
    return c


def extract_answer(text: str) -> str | None:
    """Final answer: after the last '####', else inside the last \\boxed{}, else the last number."""
    if "####" in text:
        tail = text.rsplit("####", 1)[1]
        m = _NUM.search(tail)
        return m.group(0).replace(",", "").rstrip(".") if m else tail.strip() or None
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        return boxed[-1].replace(",", "").strip()
    nums = _NUM.findall(text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def same_number(a: str | None, b: str) -> bool:
    if a is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return a.strip() == b.strip()


def answer_correct(prompts, completions, answer, **kw) -> list[float]:
    """1 if the extracted final answer equals the reference, else 0."""
    return [1.0 if same_number(extract_answer(_text(c)), a) else 0.0 for c, a in zip(completions, answer)]


def answer_format(prompts, completions, **kw) -> list[float]:
    """0.1 for using the requested '#### <number>' format (a small shaping term, gameable on its own)."""
    return [0.1 if re.search(r"####\s*-?\d", _text(c)) else 0.0 for c in completions]


REWARDS = {"answer_correct": answer_correct, "answer_format": answer_format}


def get(names: list[str]):
    unknown = [n for n in names if n not in REWARDS]
    if unknown:
        raise KeyError(f"unknown reward(s) {unknown}; available: {sorted(REWARDS)}")
    return [REWARDS[n] for n in names]
