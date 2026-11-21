"""Caption-only grammar. IDs, labels and categories never enter program parsing."""

import re
from dataclasses import dataclass
from functools import lru_cache

NUMBER_WORDS = {
    w: n
    for n, w in enumerate(
        [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
        ]
    )
}
EXACT, EXIST, NONE, AND, MORE, FEWER, SAME = range(7)

RELATIONS = {
    "to the left of": 1,
    "to the right of": 2,
    "in front of": 3,
    "behind": 4,
    "on the left side of the image": 5,
    "on the right side of the image": 6,
}

ATTRIBUTES = (
    "small",
    "large",
    "gray",
    "red",
    "blue",
    "green",
    "brown",
    "purple",
    "cyan",
    "yellow",
    "rubber",
    "metal",
    "cube",
    "sphere",
    "cylinder",
)

WORD_INDEX = {w: i + 1 for i, w in enumerate(ATTRIBUTES)}

MAX_WORDS = 4

EXCLUSIVE_GROUPS = (
    ("small", "large"),
    ("gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"),
    ("rubber", "metal"),
    ("cube", "sphere", "cylinder"),
)

_PLURAL = {w + "s": w for w in ("object", "cube", "sphere", "cylinder")}

_REL = r"(to the left of|to the right of|in front of|behind)"


@dataclass(frozen=True)
class Program:
    op: int
    a: tuple
    b: tuple = ()
    ref: tuple = ()
    relation: int = 0
    number: int = 0


def _filter(phrase):
    """'small red cubes' -> ('small', 'red', 'cube'); None if not an attribute phrase."""
    words = [_PLURAL.get(w, w) for w in phrase.split()]
    if not words or words[-1] not in ("object", "cube", "sphere", "cylinder"):
        return None
    words = [w for w in words if w != "object"]
    if len(words) > MAX_WORDS or any(w not in WORD_INDEX for w in words):
        return None
    return tuple(words)


def _program(op, a, b=None, ref=None, relation=0, number=0):
    a, b, ref = (
        _filter(a),
        () if b is None else _filter(b),
        () if ref is None else _filter(ref),
    )
    if a is None or b is None or ref is None:
        return None
    return Program(op, a, b, ref, relation, number)


@lru_cache(maxsize=131072)
def parse_program(caption):
    """Canonical train grammar (32 templates, identical in train/val/test) -> Program."""
    c = " ".join(caption.lower().strip().rstrip(".").split())
    if m := re.fullmatch(r"(?:there (?:is|are) )?exactly (\w+) (.+?) " + _REL + r" the (.+)", c):
        if m[1] in NUMBER_WORDS:
            return _program(
                EXACT,
                m[2],
                ref=m[4],
                relation=RELATIONS[m[3]],
                number=NUMBER_WORDS[m[1]],
            )
        return None
    if m := re.fullmatch(r"(?:there (?:is|are) )?exactly (\w+) (.+)", c):
        return _program(EXACT, m[2], number=NUMBER_WORDS[m[1]]) if m[1] in NUMBER_WORDS else None
    if m := re.fullmatch(r"there (?:is|are) (a|no) (.+?) " + _REL + r" the (.+)", c):
        return _program(EXIST if m[1] == "a" else NONE, m[2], ref=m[4], relation=RELATIONS[m[3]])
    if m := re.fullmatch(
        r"(?:the (.+?) is|there is a (.+?)) on the (left|right) side of the image", c
    ):
        return _program(EXIST, m[1] or m[2], relation=5 if m[3] == "left" else 6)
    if m := re.fullmatch(r"(?:a (.+?)|the (.+?) is) " + _REL + r" the (.+)", c):
        return _program(EXIST, m[1] or m[2], ref=m[4], relation=RELATIONS[m[3]])
    if m := re.fullmatch(r"there (?:is|are) (more|fewer) (.+?) than (.+)", c):
        return _program(MORE if m[1] == "more" else FEWER, m[2], m[3])
    if m := re.fullmatch(r"there (?:is|are) as many (.+?) as (.+)", c):
        return _program(SAME, m[1], m[2])
    if m := re.fullmatch(r"(?:there is )?a (.+?) and a (.+)", c):
        return _program(AND, m[1], m[2])
    if m := re.fullmatch(r"there is (a|no) (.+)", c):
        return _program(EXIST if m[1] == "a" else NONE, m[2])
    if m := re.fullmatch(r"a (.+)", c):
        return _program(EXIST, m[1])
    return None
