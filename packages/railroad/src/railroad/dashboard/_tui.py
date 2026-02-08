import math
import os
import re
from typing import Collection, List, Dict, Set, Tuple, Optional


def _is_headless_environment() -> bool:
    """Detect if running in a headless environment where live dashboards don't work well.

    Checks for:
    - CI environments (GitHub Actions, GitLab CI, etc.) via CI env var
    - Claude Code via CLAUDECODE env var
    - Google Colab via COLAB_RELEASE_TAG env var
    - Jupyter notebooks via JPY_PARENT_PID env vars
    """
    # CI environments (GitHub Actions, GitLab, Jenkins, etc.)
    if os.environ.get("CI"):
        return True

    # Claude Code
    if os.environ.get("CLAUDECODE"):
        return True

    # Google Colab
    if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU"):
        return True

    # Jupyter environments
    if os.environ.get("JPY_PARENT_PID"):
        return True

    return False


def split_markdown_flat(text: str) -> List[Dict[str, str]]:
    """
    Split markdown into a flat list of items:
    - {'type': 'h1', 'text': 'Heading 1'}
    - {'type': 'h2', 'text': 'Heading 2'}
    - {'type': 'text', 'text': 'multi-line text block'}

    Only first- and second-level headings (#, ##) are treated specially.
    Everything else is captured as text blocks between headings, preserving order.
    """
    pattern = re.compile(r'^(#{1,2})\s+(.*)$', re.MULTILINE)
    items: List[Dict[str, str]] = []

    last_pos = 0
    for m in pattern.finditer(text):
        # Text block before this heading
        if m.start() > last_pos:
            block = text[last_pos:m.start()].strip("\n")
            if block.strip():
                items.append({"type": "text", "text": block})

        hashes, heading_text = m.group(1), m.group(2).strip()
        level = len(hashes)

        if level == 1:
            items.append({"type": "h1", "text": heading_text})
        elif level == 2:
            items.append({"type": "h2", "text": heading_text})

        last_pos = m.end()

    # Trailing text after the last heading
    if last_pos < len(text):
        block = text[last_pos:].strip("\n")
        if block.strip():
            items.append({"type": "text", "text": block})

    return items


def action_color(action: str) -> str:
    """Return Rich color name based on action type."""
    act = action.split()[0] if action else ""
    if act == "move":
        return "blue"
    elif act in ("pick", "place"):
        return "green"
    elif act == "search":
        return "yellow"
    elif act == "no-op":
        return "gray"
    return "white"


def _shorten_name(name: str) -> str:
    """Shorten a name by keeping first letter of each word (camelCase) and trailing numbers.

    Examples:
        "crawler" -> "c"
        "robot1" -> "r1"
        "BigRedRobot" -> "BRR"
        "myRobot3" -> "mR3"
    """
    # Extract trailing digits
    match = re.match(r'^(.*?)(\d*)$', name)
    base, digits = match.groups() if match else (name, '')

    # Find camelCase word boundaries: start + uppercase letters
    initials = []
    if base:
        initials.append(base[0])
        for c in base[1:]:
            if c.isupper():
                initials.append(c)

    return ''.join(initials) + digits


def render_timeline(actions: List[Tuple[str, float]], robots: Set[str],
                    width: int = 50, end_time: Optional[float] = None) -> str:
    """Render Braille timeline. Each robot uses 2 vertical dots; 2 robots per row."""
    if not actions or not robots:
        return ""
    L, R, B = [0x01, 0x02, 0x04, 0x40], [0x08, 0x10, 0x20, 0x80], 0x2800  # braille dots
    actions_list = list(actions)  # for indexing by action index

    # Build events: (robot, time, index) for each robot in each action
    events = []
    for i, (act, t) in enumerate(actions):
        parts = act.split()
        involved = [r for r in robots if r in parts]
        if not involved and len(parts) >= 2 and parts[1].startswith("robot"):
            involved = [parts[1]]
        events.extend((r, t, i + 1) for r in involved)
    if not events:
        return ""

    min_t, max_t = 0.0, end_time if end_time else max(e[1] for e in events)
    if max_t <= min_t:
        max_t = min_t + 1.0
    def pos(t):
        return int((t - min_t) / (max_t - min_t) * (width * 2 - 1))

    robots_list = sorted(robots)
    # Build short name mapping
    short_names = {r: _shorten_name(r) for r in robots_list}

    # Calculate name width from shortened names (individual and paired) and full names for label rows
    individual_nw = max(len(short_names[r]) for r in robots_list)
    paired_nw = max(
        len(','.join(short_names[r] for r in robots_list[i:i + 2]))
        for i in range(0, len(robots_list), 2)
    )
    full_nw = max(len(r) for r in robots_list)  # full names used in label rows
    nw = max(individual_nw, paired_nw, full_nw)  # name width
    lines = [f"{' ' * nw} |{min_t:.1f}{' ' * (width - len(f'{min_t:.1f}') - len(f'{max_t:.1f}'))}{max_t:.1f}|"]

    # Braille rows (2 robots per row)
    for i in range(0, len(robots_list), 2):
        chunk = robots_list[i:i + 2]
        chars = [B] * width
        for ri, robot in enumerate(chunk):
            slot = ri * 2
            for r, t, _ in events:
                if r == robot:
                    ci, sub = pos(t) // 2, pos(t) % 2
                    if 0 <= ci < width:
                        chars[ci] |= (L if sub == 0 else R)[slot] | (L if sub == 0 else R)[slot + 1]
        lines.append(f"{','.join(short_names[r] for r in chunk):>{nw}} |{''.join(chr(c) for c in chars)}|")

    # Label rows (with color coding)
    for robot in robots_list:
        label_parts = []
        counts = {}
        for r, t, idx in events:
            if r == robot:
                ci = pos(t) // 2
                if 0 <= ci < width:
                    counts.setdefault(ci, []).append(idx)
        last_ci = -1
        for ci in sorted(counts.keys()):
            label_parts.append(" " * (ci - last_ci - 1))  # spaces before
            idxs = counts[ci]
            idx = idxs[0]
            color = action_color(actions_list[idx - 1][0])
            char = str(idx % 10) if len(idxs) == 1 else "+"
            label_parts.append(f"[{color}]{char}[/]")
            last_ci = ci
        label_parts.append(" " * (width - last_ci - 1))  # trailing spaces
        lines.append(f"{robot:>{nw}}  {''.join(label_parts)} ")

    return "\n".join(lines)


def _generate_coordinates(location_names: Collection[str]) -> dict[str, tuple[float, float]]:
    """Generate circular layout coordinates for locations lacking real coordinates.

    Places locations evenly around a unit circle. This is a simple placeholder
    layout -- callers with real coordinates should provide them instead.
    """
    names = sorted(location_names)
    n = len(names)
    if n == 0:
        return {}
    if n == 1:
        return {names[0]: (0.0, 0.0)}
    coords: dict[str, tuple[float, float]] = {}
    for i, name in enumerate(names):
        angle = 2 * math.pi * i / n
        coords[name] = (math.cos(angle), math.sin(angle))
    return coords
