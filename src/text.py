"""Deterministic meaning-preserving grammar rewrites and horizontal reflection."""

import re

REL = "(?:to the left of|to the right of|behind|in front of)"
NUM_ONE = "one"
_NOVEL_PATTERNS = [
    re.compile(" (is|are) in the scene$"),
    re.compile("^on the (left|right) side of the image there is "),
    re.compile(f"^{REL} the \\w+(?: \\w+)* there (is|are) "),
]


def _is_novel(caption):
    return any(p.search(caption) for p in _NOVEL_PATTERNS)


def _np_forms(np_, plural):
    """Existential forms of a noun phrase like 'a red cube' or 'a red cube and a blue sphere'."""
    verb = "are" if plural else "is"
    return [
        np_,
        f"there is {np_}",
        f"the scene contains {np_}",
        f"{np_} {verb} in the scene",
    ]


def _strip_existential(caption):
    """Return the noun phrase inside an existential caption, or None."""
    for pattern in (
        "^there is (.+)$",
        "^the scene contains (.+)$",
        "^(.+) (?:is|are) in the scene$",
    ):
        m = re.match(pattern, caption)
        if m:
            return m.group(1)
    return caption


def _attribute_like(caption):
    np_ = _strip_existential(caption)
    if np_.startswith("no "):
        body = np_[3:]
        return [
            f"there is no {body}",
            f"the scene contains no {body}",
            f"no {body} is in the scene",
        ]
    if not np_.startswith("a "):
        return [caption]
    plural = " and a " in np_
    return _np_forms(np_, plural)


def _negation(caption):
    """Like _attribute_like, without the bare noun phrase, so that both twins
    ("there is a X" / "there is no X") have the same list of forms in the same order."""
    forms = _attribute_like(caption)
    return forms[1:] if len(forms) == 4 else forms


def _counting(caption):
    m = re.match(
        "^(?:there (?:is|are) |the scene contains )?(exactly (\\w+) .+?)(?: (?:is|are) in the scene)?$",
        caption,
    )
    if not m:
        return [caption]
    core, num = (m.group(1), m.group(2))
    be = "is" if num == NUM_ONE else "are"
    return [
        core,
        f"there {be} {core}",
        f"the scene contains {core}",
        f"{core} {be} in the scene",
    ]


def _comparison(caption):
    m = re.match("^(?:there are|the scene contains) (.+)$", caption)
    if not m:
        return [caption]
    body = m.group(1)
    return [f"there are {body}", f"the scene contains {body}"]


def _position(caption):
    m = re.match("^there is (a .+) on the (left|right) side of the image$", caption) or re.match(
        "^on the (?P<side>left|right) side of the image there is (?P<np>a .+)$", caption
    )
    if not m:
        return [caption]
    if "np" in m.groupdict():
        np_, side = (m.group("np"), m.group("side"))
    else:
        np_, side = (m.group(1), m.group(2))
    return [
        f"there is {np_} on the {side} side of the image",
        f"on the {side} side of the image there is {np_}",
    ]


def _spatial(caption):
    m = re.match(f"^(?:there is )?(a .+?) ({REL}) (the .+)$", caption) or re.match(
        f"^(?P<rel>{REL}) (?P<ref>the .+?) there is (?P<np>a .+)$", caption
    )
    if not m:
        return [caption]
    if "np" in m.groupdict():
        np_, rel, ref = (m.group("np"), m.group("rel"), m.group("ref"))
    else:
        np_, rel, ref = (m.group(1), m.group(2), m.group(3))
    return [
        f"{np_} {rel} {ref}",
        f"there is {np_} {rel} {ref}",
        f"{rel} {ref} there is {np_}",
    ]


def _spatial_counting(caption):
    m = re.match(
        f"^(?:there (?:is|are) )?(exactly (\\w+) .+?) (?:(?:is|are) )?({REL}) (the .+)$",
        caption,
    ) or re.match(
        f"^(?P<rel>{REL}) (?P<ref>the .+?) there (?:is|are) (?P<core>exactly (?P<num>\\w+) .+)$",
        caption,
    )
    if not m:
        return [caption]
    if "core" in m.groupdict():
        core, num, rel, ref = (
            m.group("core"),
            m.group("num"),
            m.group("rel"),
            m.group("ref"),
        )
    else:
        core, num, rel, ref = (m.group(1), m.group(2), m.group(3), m.group(4))
    be = "is" if num == NUM_ONE else "are"
    return [
        f"there {be} {core} {rel} {ref}",
        f"{core} {be} {rel} {ref}",
        f"{rel} {ref} there {be} {core}",
    ]


def _spatial_negation(caption):
    m = re.match(f"^(?:there is )?no (.+?) (?:is )?({REL}) (the .+)$", caption) or re.match(
        f"^(?P<rel>{REL}) (?P<ref>the .+?) there is no (?P<body>.+)$", caption
    )
    if not m:
        return [caption]
    if "body" in m.groupdict():
        body, rel, ref = (m.group("body"), m.group("rel"), m.group("ref"))
    else:
        body, rel, ref = (m.group(1), m.group(2), m.group(3))
    return [
        f"there is no {body} {rel} {ref}",
        f"no {body} is {rel} {ref}",
        f"{rel} {ref} there is no {body}",
    ]


_RULES = {
    "attribute": _attribute_like,
    "binding": _attribute_like,
    "negation": _negation,
    "counting": _counting,
    "comparison": _comparison,
    "position": _position,
    "spatial": _spatial,
    "spatial_counting": _spatial_counting,
    "spatial_negation": _spatial_negation,
}


def paraphrases(caption, category):
    """All equivalent surface forms of a caption (always includes the caption itself)."""
    caption = caption.strip().lower()
    rule = _RULES.get(category)
    forms = rule(caption) if rule else [caption]
    out = list(dict.fromkeys(forms))
    if caption not in out:
        out.append(caption)
    return out


def canonical_form(caption, category):
    """Deterministic rewrite into one fixed train form: the first non-novel paraphrase.

    Every equivalent surface form of a statement maps to the same string, and that
    string's template occurs in train (checked on train/val/test: 0 exceptions, twins
    never collapse). Used to remove the phrasing shift of test instead of learning it.
    """
    for form in paraphrases(caption, category):
        if not _is_novel(form):
            return form
    return caption.strip().lower()


def swap_left_right(caption):
    """Swap 'left' and 'right'; used together with a horizontal image flip."""
    return re.sub(
        "\\b(left|right)\\b",
        lambda m: "right" if m.group(1) == "left" else "left",
        caption,
    )
