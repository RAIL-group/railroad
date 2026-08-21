"""Glyphs the matcher gets confidently wrong.

Kept deliberately short. Every entry here is a small admission that embedding
similarity missed, and the point of deriving the vocabulary from the font was to
avoid maintaining a name-to-emoji table -- so an entry earns its place only when
the match is wrong in a way a reader would notice.

Measured similarity does not separate right from wrong: ``baseballbat`` matches
BAT at 0.91 because "bat" really is the head noun, just the wrong sense. No
threshold reaches these; only this table does.

Keys are written as the object categories actually appear and are normalized the
same way queries are, so it does not matter whether a key or a query happens to
be spelled run-together.
"""

from __future__ import annotations

NAME_TO_CODEPOINT: dict[str, int] = {
    "baseballbat": 0x26BE,      # baseball, not the animal
    "bowl": 0x1F963,            # bowl with spoon, not bowling
    "coffeemachine": 0x2615,    # hot beverage
    "desklamp": 0x1F4A1,        # light bulb
    "dumbbell": 0x1F3CB,        # person lifting weights, not a bell
    "faucet": 0x1F6B0,          # potable water
    "fridge": 0x1F9CA,          # ice cube; there is no refrigerator glyph
    "refrigerator": 0x1F9CA,
    "ladle": 0x1F944,           # spoon; no ladle glyph exists
    "laptop": 0x1F4BB,          # personal computer, not a notebook
    "remotecontrol": 0x1F4FA,   # television, the thing it controls
    "safe": 0x1F510,            # lock with key
    "soapbottle": 0x1F9F4,      # lotion bottle -- the head noun alone finds a baby bottle
    "spraybottle": 0x1F9F4,
    "statue": 0x1F5FF,          # moai
    "tissuebox": 0x1F927,       # sneezing face -- what a tissue is for
    "winebottle": 0x1F377,      # wine glass, not baby bottle
}

MAX_ENTRIES = 20
"""A budget, enforced by the tests. Past this the design has failed."""
