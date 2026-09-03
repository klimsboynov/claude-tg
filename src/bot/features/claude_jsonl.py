"""Render a Claude Code session .jsonl into clean Telegram messages.

The tmux bridge types into a live interactive ``claude``; this module reads the
structured session log that ``claude`` writes and turns each new record into a
native Telegram message -- assistant replies as markdown, tool calls as compact
one-liners -- so the phone sees real, copyable output the *moment* ``claude``
writes it, instead of a scraped terminal screen.

Interactive ``claude`` writes no ``result`` record (that is SDK/-p only), so a
turn is "done" when the last assistant record stops for a reason other than
``tool_use``. Thinking blocks and tool results are skipped to keep the feed
clean.
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple

from ..utils.html_format import escape_html, markdown_to_telegram_html

_MAX_USER_ECHO = 800

_TOOL_EMOJI = {
    "Bash": "💻",
    "Read": "📖",
    "Edit": "✏️",
    "Write": "📝",
    "MultiEdit": "✏️",
    "Glob": "🔍",
    "Grep": "🔍",
    "Task": "🤖",
    "Skill": "🧩",
    "WebFetch": "🌐",
    "WebSearch": "🌐",
    "TodoWrite": "✅",
    "NotebookEdit": "📓",
}


# Screen markers that mean the TUI is waiting on an interactive choice. These
# live only on-screen (capture-pane), never in the .jsonl, so the mirror has to
# watch the pane to surface them. Kept to widget *chrome* only -- the selection
# footer -- so ordinary assistant prose ("do you want to…") never trips it, and
# "esc to interrupt" (the thinking state) is deliberately excluded.
_PROMPT_MARKERS = (
    "to navigate",
    "enter to select",
    "esc to cancel",
)

# The normal input box / thinking footer. A live menu *replaces* the input box,
# so if any of these are on screen it's NOT a menu -- this is what stops the
# detector from firing on assistant prose (or this very source file) that merely
# mentions the marker phrases while the input box sits below it.
_INPUT_FOOTER = (
    "bypass permissions",
    "shift+tab to cycle",
    "for agents",
    "esc to interrupt",
)

# Lines that change on their own (timers, token counters, spinner) -- excluded
# from a prompt's signature so a ticking clock doesn't look like a new prompt.
_VOLATILE_LINE = re.compile(
    r"\(\d+m?\s*\d*s|esc to interrupt|[↓↑]\s*[\d.]+k?\s*tokens", re.IGNORECASE
)


def is_question_widget(screen: str) -> bool:
    """True if the pane shows Claude Code's multi-question widget
    (AskUserQuestion): tabbed questions, checkbox options, a Review/Submit
    step. Its chrome carries none of the classic menu markers, and digit+Enter
    handling would mis-answer tabs -- so it gets a key-remote UI instead.

    Only the footer region is inspected (scrollback prose that merely quotes
    the marker phrases must not trip it), and a visible input box means the
    widget is closed.
    """
    tail = [ln for ln in screen.splitlines() if ln.strip()][-18:]
    joined = "\n".join(tail)
    low = joined.lower()
    if any(m in low for m in _INPUT_FOOTER):
        return False
    # Tab strip: arrows plus checkbox/submit glyphs on one line.
    for ln in tail:
        if "←" in ln and "→" in ln and any(g in ln for g in ("☒", "☐", "✔")):
            return True
    return "review your answers" in low or "ready to submit" in low


def is_interactive_prompt(screen: str) -> bool:
    """True if the pane snapshot looks like a menu/permission prompt awaiting input.

    Only the footer region (last few non-empty lines) is inspected, so marker
    phrases appearing in the scrollback body never trip it; and if the normal
    input-box footer is present, it's the idle/typing state, not a menu.
    """
    tail = [ln for ln in screen.splitlines() if ln.strip()][-6:]
    low = "\n".join(tail).lower()
    if any(m in low for m in _INPUT_FOOTER):
        return False
    return any(m in low for m in _PROMPT_MARKERS)


def prompt_signature(screen: str) -> str:
    """Stable identity of a prompt screen.

    Drops volatile (timer) lines and the leading highlight marker (``❯``) so a
    moving selection cursor doesn't read as a different prompt.
    """
    out = []
    for line in screen.splitlines():
        if not line.strip() or _VOLATILE_LINE.search(line):
            continue
        out.append(re.sub(r"^\s*[❯>›»]\s*", "", line).rstrip())
    return "\n".join(out)


# A menu option line: optional highlight marker, a number, ``.`` or ``)``, label.
_OPTION_RE = re.compile(r"^\s*(?:[❯>›»]\s*)?(\d+)[.)]\s+(\S.*?)\s*$")


def _is_boundary(line: str) -> bool:
    """A line that ends an option's description block."""
    s = line.strip()
    if not s or set(s) <= set("─—-=_• "):
        return True
    if _OPTION_RE.match(line):
        return True
    low = s.lower()
    return any(
        k in low for k in ("to navigate", "enter to select", "esc to cancel")
    )


def parse_menu(screen: str) -> Optional[dict]:  # type: ignore[type-arg]
    """Parse a TUI selection window into ``{title, options:[(n, label, desc)]}``.

    ``desc`` is the indented description block under an option (joined to one
    line), or "" when none is shown. Keeps only the contiguous ``1..N`` run so
    stray numbered lines in the scrollback can't leak in. Returns None if
    there's no option numbered 1.
    """
    lines = screen.splitlines()
    numbered: dict = {}  # n -> (label, line_index)
    first_idx: Optional[int] = None
    for idx, line in enumerate(lines):
        m = _OPTION_RE.match(line)
        if m:
            numbered[int(m.group(1))] = (m.group(2).strip(), idx)
            if first_idx is None:
                first_idx = idx
    if 1 not in numbered:
        return None

    def _desc(start: int) -> str:
        out = []
        for j in range(start + 1, len(lines)):
            if _is_boundary(lines[j]):
                break
            out.append(lines[j].strip())
        return " ".join(out)

    options = []
    i = 1
    while i in numbered:
        label, idx = numbered[i]
        options.append((i, label, _desc(idx)))
        i += 1

    # Context block above the options: the command / danger warning / question
    # a permission box shows. Walk up from the first option collecting real
    # lines until a box border (or a small cap), skipping blank gutters -- so
    # the user sees WHAT they're approving, not just "Do you want to proceed?".
    context: List[str] = []
    rule = set("─—-=_╭╮╰╯│├┤┬┴┼┃━ •")
    if first_idx is not None:
        for line in reversed(lines[:first_idx]):
            s = line.strip()
            if not s:
                continue  # box gutter -- skip, don't stop
            if set(s) <= rule:
                break  # box border ends the block
            low = s.lower()
            if any(
                k in low
                for k in ("to navigate", "enter to select", "esc to cancel")
            ):
                continue
            context.append(s)
            if len(context) >= 12:
                break
        context.reverse()
    title = context[-1] if context else "Claude is asking"
    return {"title": title, "context": "\n".join(context), "options": options}


def _newest_jsonl(proj_dir: Path) -> Optional[Path]:
    """Most recently modified .jsonl in a Claude project dir, if any."""
    try:
        files = list(proj_dir.glob("*.jsonl"))
    except OSError:
        return None
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def claude_project_dir(pane_cwd: Path) -> Path:
    """Claude's session-store dir for ``pane_cwd`` (may not exist yet).

    Claude encodes the absolute cwd by replacing every non-alphanumeric
    character (``/``, ``.``, ``_`` …) with ``-`` under ``~/.claude/projects/``.
    """
    encoded = re.sub(r"[^A-Za-z0-9]", "-", str(pane_cwd))
    return Path.home() / ".claude" / "projects" / encoded


def resolve_session_file(pane_cwd: Path) -> Optional[Path]:
    """Newest session .jsonl under Claude's project dir for ``pane_cwd``.

    Returns None when no session log exists there yet (caller may still enter
    mirror mode if the pane is running ``claude`` and wait for the log).
    """
    return _newest_jsonl(claude_project_dir(pane_cwd))


def _summarize_tool(name: str, inp: dict) -> str:  # type: ignore[type-arg]
    """One-line HTML summary of a tool_use block."""
    emoji = _TOOL_EMOJI.get(name, "🔧")
    detail = ""
    if name == "Bash":
        cmd = inp.get("command") or ""
        detail = cmd.splitlines()[0] if cmd else ""
    elif name in ("Read", "Edit", "Write", "MultiEdit", "NotebookEdit"):
        fp = inp.get("file_path") or inp.get("notebook_path") or ""
        detail = fp.split("/")[-1] if fp else ""
    elif name in ("Grep", "Glob"):
        detail = inp.get("pattern") or ""
    elif name == "Skill":
        detail = inp.get("skill") or inp.get("command") or ""
    elif name == "Task":
        detail = inp.get("description") or ""
    elif name in ("WebFetch", "WebSearch"):
        detail = inp.get("url") or inp.get("query") or ""
    elif name == "TodoWrite":
        detail = "updating todos"
    else:
        for v in inp.values():
            if isinstance(v, str) and v.strip():
                detail = v.strip()
                break
    detail = detail.strip()[:140]
    line = f"{emoji} <b>{escape_html(name)}</b>"
    if detail:
        line += f" · <code>{escape_html(detail)}</code>"
    return line


def _user_text(content: object) -> Optional[str]:
    """Extract a plain typed prompt from a user record, or None if it's a
    tool_result / non-text payload."""
    if isinstance(content, str):
        return content.strip() or None
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_result":
                return None
        texts = [
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        joined = "\n".join(t for t in texts if t).strip()
        return joined or None
    return None


def user_prompt_text(obj: dict) -> Optional[str]:  # type: ignore[type-arg]
    """Raw human prompt text from a user record, or None if not a plain prompt.

    None for tool results, meta records, and command wrappers -- i.e. it only
    returns text a person actually typed.
    """
    if obj.get("type") != "user" or obj.get("isMeta"):
        return None
    msg = obj.get("message") or {}
    text = _user_text(msg.get("content") if isinstance(msg, dict) else None)
    if not text:
        return None
    # Skip command wrappers / local-command noise.
    if text.startswith("<") and text.rstrip().endswith(">"):
        return None
    return text


def render_record(
    obj: dict,  # type: ignore[type-arg]
    *,
    show_tools: bool = True,
) -> Optional[Tuple[str, Optional[str], str]]:
    """Turn one jsonl record into (text, parse_mode, kind), or None to skip.

    ``parse_mode`` is "HTML" for everything we emit; ``kind`` is one of
    ``user`` / ``text`` / ``tool`` so callers can coalesce runs of tool
    one-liners into a single message (rate-limit relief in busy groups).
    Skips meta records, thinking blocks, tool results, and Claude's internal
    bookkeeping types.
    """
    t = obj.get("type")

    # Human prompt (typed on PC or phone) -- echo it so the mirror is complete.
    if t == "user":
        text = user_prompt_text(obj)
        if not text:
            return None
        if len(text) > _MAX_USER_ECHO:
            text = text[:_MAX_USER_ECHO] + " …"
        return (f"💬 <b>You</b>\n{escape_html(text)}", "HTML", "user")

    if t != "assistant":
        return None
    msg = obj.get("message") or {}
    content = msg.get("content") if isinstance(msg, dict) else None
    if not isinstance(content, list):
        return None

    for block in content:
        if not isinstance(block, dict):
            continue
        bt = block.get("type")
        if bt == "text":
            txt = (block.get("text") or "").strip()
            if txt:
                return (markdown_to_telegram_html(txt), "HTML", "text")
        elif bt == "tool_use" and show_tools:
            line = _summarize_tool(block.get("name", ""), block.get("input") or {})
            return (line, "HTML", "tool")
        # thinking -> skip
    return None
