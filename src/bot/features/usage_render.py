"""Render Claude Code's Usage/Stats screen into a compact Telegram message.

The TUI panel mirrors as a wide, misaligned monospace dump (box rules, columns
padded to terminal width) that's unreadable on a phone. ``render_usage`` parses
the numbers out and lays them out cleanly; it returns ``None`` when the snapshot
isn't a usage screen, so callers fall back to the raw ``<pre>`` render.
"""

import re
from html import escape
from typing import List, Optional


def looks_like_usage(snapshot: str) -> bool:
    """True if the captured screen is the Usage/Stats panel."""
    return "Total cost:" in snapshot and (
        "Usage by model" in snapshot
        or "Current session" in snapshot
        or "Current week" in snapshot
    )


def _bar(pct: int, width: int = 10) -> str:
    pct = max(0, min(100, pct))
    filled = int(round(pct / 100 * width))
    return "▓" * filled + "░" * (width - filled)


def _search(pattern: str, text: str, flags: int = 0) -> Optional[str]:
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else None


def _short_dur(dur: str) -> str:
    """"1h 53m 8s" -> "1h 53m"; drop trailing seconds when a bigger unit exists."""
    dur = dur.strip()
    if re.search(r"\d\s*[hm]\b", dur):
        return re.sub(r"\s*\d+\s*s\b", "", dur).strip()
    return dur


def _section(text: str, header: str, label: str) -> Optional[str]:
    """Format one 'NN% used … Resets …' block as a bar row, or None if absent."""
    m = re.search(re.escape(header) + r".*?(\d+)%\s*used", text, re.DOTALL)
    if not m:
        return None
    pct = int(m.group(1))
    resets = _search(r"Resets\s+([^\n·]+)", text[m.end() :])
    row = f"<code>{label:<7} {_bar(pct)} {pct:>3}%</code>"
    if resets:
        resets = re.sub(r"\s*\(UTC\)", "", resets).split(",")[0].strip().rstrip(". ")
        row += f" · {escape(resets)}"
    return row


def render_usage(snapshot: str) -> Optional[str]:
    """Return a Telegram-HTML usage card, or None if not a usage screen."""
    if not looks_like_usage(snapshot):
        return None
    text = snapshot
    out: List[str] = ["📊 <b>Usage</b>"]

    cost = _search(r"Total cost:\s*\$?([\d,]+(?:\.\d+)?)", text)
    api = _search(r"Total duration \(API\):\s*([^\n]+)", text)
    wall = _search(r"Total duration \(wall\):\s*([^\n]+)", text)
    head: List[str] = []
    if cost:
        head.append(f"💰 <b>${cost}</b>")
    if api:
        head.append(f"⏱ {escape(_short_dur(api))} API")
    if wall:
        head.append(f"🕒 {escape(_short_dur(wall))} wall")
    if head:
        out.append(" · ".join(head))

    ch = re.search(r"([\d,]+)\s+lines added,\s*([\d,]+)\s+lines removed", text)
    if ch:
        out.append(f"📝 +{ch.group(1)} / −{ch.group(2)} lines")

    rows = [
        _section(text, "Current session", "session"),
        _section(text, "Current week (all models)", "week"),
        _section(text, "Current week (Fable)", "fable"),
    ]
    rows = [r for r in rows if r]
    if rows:
        out += ["", "<b>Limits</b>", *rows]

    models = [
        (m.group(1).replace("claude-", ""), float(m.group(2).replace(",", "")))
        for m in re.finditer(
            r"(claude-[\w.\-]+):[^\n]*\(\$([\d,]+(?:\.\d+)?)\)", text
        )
    ]
    if models:
        models.sort(key=lambda x: -x[1])
        out += ["", "<b>By model</b>"]
        out += [f"• {escape(n)} — ${c:,.2f}" for n, c in models]

    drivers = re.findall(r"(\d+)% of your usage (?:was|came from) ([^\n]+)", text)
    if drivers:
        out += ["", "<b>Top drivers · 24h</b>"]
        for pct, what in drivers[:3]:
            what = re.sub(r"\s+", " ", what).strip().rstrip(".")
            out.append(f"• {pct}% — {escape(what)}")

    tail = text.rsplit("Skills", 1)[-1]
    skills = re.findall(r"(/[\w\-]+)\s+(\d+)%", tail)
    if skills:
        pretty = " · ".join(f"{escape(s)} {p}%" for s, p in skills[:6])
        out += ["", f"<b>Skills</b> {pretty}"]

    return "\n".join(out)
