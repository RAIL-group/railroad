"""Turning ``teddybear_6`` into a glyph.

Normalization and the vocabulary need only a font; the embedding tiers need the
sentence model, and skip rather than download it.
"""

import pytest

from railroad.dashboard._sprites import fonts, matching, overrides, vocabulary
from railroad.dashboard._sprites.resources import get_emoji_sbert_dir

pytestmark = pytest.mark.skipif(
    fonts.find_font() is None, reason="no colour emoji font on this machine"
)

WORDS = frozenset(
    "alarm clock spray bottle coffee machine house plant teddy bear butter knife"
    " garbage can wine dumb bell toilet paper".split()
)


@pytest.fixture
def font_path():
    return fonts.find_font()


@pytest.fixture
def entries(font_path):
    return vocabulary.build(font_path)


def _needs_model():
    if not (get_emoji_sbert_dir() / "modules.json").exists():
        pytest.skip("glyph matching model not downloaded")
    pytest.importorskip("sentence_transformers")


# --- vocabulary -------------------------------------------------------------

def test_the_vocabulary_covers_ordinary_household_glyphs(entries):
    """A regression on the codepoint ranges.

    ALARM CLOCK sits at U+23F0, below the pictograph blocks -- a filter that
    starts at U+2600 silently loses it and every other object down there.
    """
    labels = {label for label, _cp in entries}
    for expected in ("alarm clock", "chair", "teddy bear", "wastebasket",
                     "personal computer", "watch"):
        assert expected in labels


def test_the_vocabulary_excludes_sequence_only_codepoints(entries):
    """Flag halves and skin-tone modifiers draw as nothing on their own."""
    codepoints = {codepoint for _label, codepoint in entries}
    assert not codepoints & set(range(0x1F1E6, 0x1F200))   # regional indicators
    assert not codepoints & set(range(0x1F3FB, 0x1F400))   # skin tones


def test_a_glyph_is_produced_for_every_object_name():
    """A few cmap entries still render blank, so the provider must not pass
    that through as "no sprite" -- the object would silently vanish."""
    from railroad.dashboard._sprites import get_glyph_provider

    provider = get_glyph_provider()
    assert provider is not None
    for name in ("teddybear_6", "zzqqxx_1", "wheel", "handshake", "female sign"):
        assert provider.glyph_for(name) is not None


# --- normalization ----------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("teddybear_6", "teddy bear"),
        ("SprayBottle", "spray bottle"),
        ("Knife", "knife"),
        ("garbagecan_5", "garbage can"),
        ("coffeemachine", "coffee machine"),
        ("alarmclock", "alarm clock"),
        ("houseplant", "house plant"),
    ],
)
def test_names_normalize_to_their_english_form(raw, expected):
    assert matching.normalize(raw, WORDS) == expected


def test_a_trailing_word_is_not_mistaken_for_an_index():
    """Only a numeric suffix is an index; the rest of the name is meaning."""
    assert matching.normalize("toilet_paper_hanger", WORDS) == "toilet paper hanger"


def test_an_unsplittable_compound_is_left_alone():
    """Failing to split costs match quality, never correctness."""
    assert matching.split_compound("stoveburner", WORDS) is None
    assert matching.normalize("stoveburner", WORDS) == "stoveburner"


def test_a_word_already_in_the_lexicon_is_not_torn_apart():
    assert matching.split_compound("wine", WORDS) == ["wine"]


# --- overrides --------------------------------------------------------------

def test_overrides_name_glyphs_the_font_can_draw(entries):
    """A typo here would silently produce a blank sprite."""
    renderable = {codepoint for _label, codepoint in entries}
    for name, codepoint in overrides.NAME_TO_CODEPOINT.items():
        assert codepoint in renderable, f"{name} maps outside the font"


def test_the_override_table_stays_small():
    """It exists to patch a handful of confident errors, not to become the table
    this whole design was meant to avoid."""
    assert len(overrides.NAME_TO_CODEPOINT) <= overrides.MAX_ENTRIES


# --- matching ---------------------------------------------------------------

def test_exact_labels_match_without_the_model(font_path, monkeypatch):
    """Tier two: a name that *is* a Unicode label needs no embeddings."""
    monkeypatch.setattr(matching, "_load_model", lambda: None)
    assert matching.match("Toilet", font_path) == 0x1F6BD


def test_matching_without_a_model_falls_back_rather_than_failing(font_path, monkeypatch):
    monkeypatch.setattr(matching, "_load_model", lambda: None)
    assert matching.match("wumpus_3", font_path) == matching.FALLBACK_CODEPOINT


@pytest.mark.parametrize(
    "name, expected",
    [
        ("teddybear_6", 0x1F9F8),      # teddy bear
        ("alarmclock_2", 0x23F0),      # alarm clock
        ("creditcard", 0x1F4B3),       # credit card
        ("laptop_9", 0x1F4BB),         # personal computer, via an override
        ("dumbbell", 0x1F3CB),         # override survives being split
        ("houseplant", 0x1FAB4),       # potted plant, via the head noun
    ],
)
def test_representative_objects_get_the_right_glyph(name, expected, font_path):
    _needs_model()
    assert matching.match(name, font_path) == expected


def test_a_meaningless_name_gets_the_generic_glyph(font_path):
    _needs_model()
    assert matching.match("zzqqxx_1", font_path) == matching.FALLBACK_CODEPOINT
