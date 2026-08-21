"""Turning an object's symbolic name into a glyph.

Four tiers, narrowing from certain to plausible: an override, an exact label
match, embedding similarity, and finally a generic box. The tiers matter because
each needs strictly more than the last -- only the third wants a sentence model,
so a machine with a font but no model still gets useful sprites.

Object names arrive as ``teddybear_6`` or ``SprayBottle``, and the labels they
are matched against are Unicode names like ``teddy bear``. Closing that gap is
most of the work, and splitting run-together compounds is the part that pays:
``coffeemachine`` matches nothing until it becomes ``coffee machine``.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from . import vocabulary
from .overrides import NAME_TO_CODEPOINT
from .resources import EMOJI_SBERT_MODEL_NAME, get_emoji_sbert_dir

FALLBACK_CODEPOINT = 0x1F4E6
"""PACKAGE -- what an unrecognised object gets."""

MIN_SIMILARITY = 0.45
"""Below this, no candidate is plausible enough to beat a generic box.

Not a correctness filter. Measured scores for right and wrong answers overlap
heavily, so raising this discards good matches without catching bad ones; the
override table is what handles confident errors.
"""

MIN_PART = 3
"""Shortest fragment a compound may be split into."""

_LABEL_STOP_WORDS = frozenset(
    "button sign symbol type blood mark selector tone skin letter flag"
    " with and the for".split()
)

_SUFFIX = re.compile(r"_\d+$")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

_MODEL: Any = None
_LEXICON: frozenset[str] | None = None
_MATRIX: dict[str, Any] = {}
_OVERRIDES: dict[str, int] | None = None


def _normalized_overrides(words: frozenset[str]) -> dict[str, int]:
    """The override table keyed the way a query will arrive.

    Running the keys through the same normalization is what lets the table be
    written in the object's own spelling: ``dumbbell`` is stored once and still
    matches after the splitter has turned it into ``dumb bell``.
    """
    global _OVERRIDES
    if _OVERRIDES is None:
        _OVERRIDES = {
            normalize(key, words): codepoint
            for key, codepoint in NAME_TO_CODEPOINT.items()
        }
    return _OVERRIDES


def _load_model() -> Any:
    """The sentence model, or ``None`` if it is not installed or not downloaded."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL if _MODEL is not False else None
    model_dir = get_emoji_sbert_dir()
    if not (model_dir / "modules.json").exists():
        _MODEL = False
        return None
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        _MODEL = False
        return None
    _MODEL = SentenceTransformer(str(model_dir))
    return _MODEL


def lexicon(labels: Iterable[str] = ()) -> frozenset[str]:
    """Words a run-together compound may be split into.

    Drawn from the sentence model's own tokenizer, which ships tens of thousands
    of ordinary English words in the model directory that is already being
    downloaded. Deriving the list from the emoji labels instead does not work:
    labels like "A BUTTON (BLOOD TYPE)" contribute single letters, and a lexicon
    containing "a" will split any word into rubble.
    """
    global _LEXICON
    if _LEXICON is not None:
        return _LEXICON

    words: set[str] = set()
    cache_path = get_emoji_sbert_dir() / "lexicon.json"
    if cache_path.exists():
        words.update(json.loads(cache_path.read_text()))
    else:
        model_dir = get_emoji_sbert_dir()
        if (model_dir / "tokenizer.json").exists():
            try:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
                vocab = tokenizer.get_vocab() if tokenizer is not None else {}
                words.update(
                    w for w in vocab
                    if w.isalpha() and w.islower() and len(w) >= MIN_PART
                )
                cache_path.write_text(json.dumps(sorted(words)))
            except Exception:
                # A missing or unreadable tokenizer costs split quality, never
                # correctness: an unsplit name still matches, just less well.
                pass

    for label in labels:
        words.update(
            word for word in label.split()
            if len(word) >= MIN_PART and word not in _LABEL_STOP_WORDS
        )
    _LEXICON = frozenset(words)
    return _LEXICON


def split_compound(word: str, words: frozenset[str] | None = None) -> list[str] | None:
    """Split a run-together compound, or ``None`` if it does not cover cleanly.

    Scores by the sum of squared fragment lengths, which prefers a few long
    words over many short ones -- and, because ``(a+b)^2 > a^2 + b^2``, leaves a
    word that is itself in the lexicon alone.
    """
    known = words if words is not None else lexicon()
    if not known:
        return None
    size = len(word)
    best: list[tuple[int, list[str]] | None] = [None] * (size + 1)
    best[0] = (0, [])
    for end in range(MIN_PART, size + 1):
        for start in range(0, end - MIN_PART + 1):
            prefix = best[start]
            if prefix is None:
                continue
            part = word[start:end]
            if part not in known:
                continue
            score = prefix[0] + len(part) ** 2
            incumbent = best[end]
            if incumbent is None or score > incumbent[0]:
                best[end] = (score, prefix[1] + [part])
    result = best[size]
    return result[1] if result else None


def normalize(name: str, words: frozenset[str] | None = None) -> str:
    """``teddybear_6`` -> ``teddy bear``; ``SprayBottle`` -> ``spray bottle``."""
    text = _SUFFIX.sub("", name)
    text = _CAMEL.sub(" ", text)
    text = text.replace("_", " ").replace("-", " ").lower()
    parts: list[str] = []
    for token in text.split():
        pieces = split_compound(token, words) if len(token) > 2 * MIN_PART else None
        parts.extend(pieces or [token])
    return " ".join(parts)


def _matrix_for(font_path: Path, entries: tuple[tuple[str, int], ...]) -> Any:
    """Embeddings for every label, cached to one file rather than one per label.

    Encoding fourteen hundred short strings takes seconds, so it is worth doing
    once ever; the key covers the font because a different font means a
    different candidate set.
    """
    import numpy as np

    digest = hashlib.sha256(
        f"{EMOJI_SBERT_MODEL_NAME}|{font_path}|{font_path.stat().st_mtime_ns}"
        f"|{len(entries)}".encode()
    ).hexdigest()[:16]
    if digest in _MATRIX:
        return _MATRIX[digest]

    cache_path = get_emoji_sbert_dir() / f"vocab_{digest}.npz"
    if cache_path.exists():
        _MATRIX[digest] = np.load(cache_path)["labels"]
        return _MATRIX[digest]

    model = _load_model()
    if model is None:
        return None
    matrix = np.asarray(
        model.encode(
            [label for label, _cp in entries],
            normalize_embeddings=True,
            batch_size=256,
            show_progress_bar=False,
        )
    )
    np.savez(cache_path, labels=matrix)
    _MATRIX[digest] = matrix
    return matrix


def match(name: str, font_path: Path | None = None) -> int:
    """The codepoint that best represents *name*."""
    from .fonts import find_font

    path = font_path or find_font()
    if path is None:
        return FALLBACK_CODEPOINT
    entries = vocabulary.build(path)
    words = lexicon(label for label, _cp in entries)
    normalized = normalize(name, words)

    overrides = _normalized_overrides(words)
    if normalized in overrides:
        return overrides[normalized]

    by_label = {label: codepoint for label, codepoint in entries}
    if normalized in by_label:
        return by_label[normalized]

    matrix = _matrix_for(path, entries)
    model = _load_model()
    if matrix is None or model is None:
        return FALLBACK_CODEPOINT

    import numpy as np

    # English compounds are head-final, so the last word often carries the
    # meaning the whole phrase misses: "butter knife" is a knife, not butter.
    # Scoring both and taking the better recovers a good share of them.
    queries = [normalized]
    head = normalized.split()[-1] if " " in normalized else None
    if head:
        queries.append(head)
    embedded = np.asarray(
        model.encode(queries, normalize_embeddings=True, show_progress_bar=False)
    )
    similarity = (embedded @ matrix.T).max(axis=0)
    best = int(similarity.argmax())
    if similarity[best] < MIN_SIMILARITY:
        return FALLBACK_CODEPOINT
    return entries[best][1]


def _reset_caches() -> None:
    global _MODEL, _LEXICON, _OVERRIDES
    _MODEL = None
    _LEXICON = None
    _OVERRIDES = None
    _MATRIX.clear()
