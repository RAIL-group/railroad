import pytest

from railroad.plotting import emoji

pytestmark = pytest.mark.skipif(
    emoji.find_font() is None, reason="no color emoji font on this machine"
)


@pytest.fixture
def font_path():
    return emoji.find_font()


def test_vocabulary_contains_named_objects_but_not_sequence_parts(font_path):
    entries = emoji.entries(font_path)
    labels = {label for label, _ in entries}
    codepoints = {codepoint for _, codepoint in entries}
    assert {"alarm clock", "chair", "teddy bear", "watch"} <= labels
    assert not codepoints & set(range(0x1F1E6, 0x1F200))
    assert not codepoints & set(range(0x1F3FB, 0x1F400))


@pytest.mark.parametrize(
    "name, expected",
    [
        ("teddybear_6", 0x1F9F8),
        ("alarm_clock_2", 0x23F0),
        ("creditcard", 0x1F4B3),
        ("laptop_9", 0x1F4BB),
        ("dumbbell", 0x1F3CB),
        ("Mug_3", 0x2615),
        ("houseplant", 0x1FAB4),
    ],
)
def test_representative_names(name, expected, font_path):
    assert emoji.match(name, font_path) == expected


def test_unknown_name_uses_fallback(font_path):
    assert emoji.match("zzqqxx_1", font_path) == emoji.FALLBACK_CODEPOINT


def test_sentence_matcher_handles_a_synonym(font_path):
    if not (emoji.model_dir() / "modules.json").is_file():
        pytest.skip("emoji matching model not downloaded")
    pytest.importorskip("sentence_transformers")
    assert emoji.match("couch", font_path) == 0x1F6CB


def test_overrides_are_small_and_renderable(font_path):
    renderable = {codepoint for _, codepoint in emoji.entries(font_path)}
    assert len(emoji.OVERRIDES) <= 20
    assert set(emoji.OVERRIDES.values()) <= renderable
