"""Bridge Telegram to an interactive Claude Code session running in tmux.

Instead of spawning a headless ``claude -p`` per message, the bridge types the
message straight into a long-lived interactive ``claude`` running in a named
tmux pane on this machine -- so the phone and the PC drive the *same* process
and the *same* session, with no jsonl forking.

Output is read back on demand via ``capture-pane`` (a rendered screen snapshot,
ANSI stripped), never live-streamed: Claude's TUI redraws constantly, and a
snapshot keeps that noise out of Telegram.
"""

import asyncio
from typing import Dict, List, Optional, Tuple

import structlog

logger = structlog.get_logger()


async def _tmux(*args: str) -> Tuple[int, str, str]:
    """Run ``tmux <args>`` and return (returncode, stdout, stderr)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "tmux",
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return 127, "", "tmux is not installed"
    out, err = await proc.communicate()
    return (
        proc.returncode or 0,
        out.decode("utf-8", "replace"),
        err.decode("utf-8", "replace"),
    )


async def list_sessions() -> List[Dict[str, object]]:
    """List every live tmux session on this host.

    Returns dicts with ``name``, ``windows`` (int), and ``attached`` (bool),
    newest activity first. Empty list if no server is running.
    """
    fmt = (
        "#{session_name}\t#{session_windows}\t"
        "#{session_attached}\t#{session_activity}"
    )
    code, out, _ = await _tmux("list-sessions", "-F", fmt)
    if code != 0:
        return []
    rows: List[Dict[str, object]] = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        name, windows, attached, activity = parts[0], parts[1], parts[2], parts[3]
        rows.append(
            {
                "name": name,
                "windows": int(windows) if windows.isdigit() else 0,
                "attached": attached == "1",
                "activity": int(activity) if activity.isdigit() else 0,
            }
        )
    rows.sort(key=lambda r: r["activity"], reverse=True)  # type: ignore[arg-type]
    return rows


class TmuxBridge:
    """Types into and snapshots an interactive Claude session in tmux.

    ``target`` is a tmux target: a session name (``claude``) or a fully
    qualified ``session:window.pane``.
    """

    # Serialize the multi-step send_text sequence per target across all bridge
    # instances (callers build a fresh TmuxBridge each time), so concurrent
    # sends -- e.g. a backgrounded file-delivery note racing a typed message --
    # can't interleave their send-keys and corrupt the composer.
    _send_locks: Dict[str, asyncio.Lock] = {}

    def __init__(self, target: str) -> None:
        self.target = target

    def _send_lock(self) -> asyncio.Lock:
        lock = TmuxBridge._send_locks.get(self.target)
        if lock is None:
            lock = TmuxBridge._send_locks[self.target] = asyncio.Lock()
        return lock

    def _session_name(self) -> str:
        """has-session wants a session, not a pane; drop any :win.pane suffix."""
        return self.target.split(":", 1)[0]

    async def _run(self, *args: str) -> Tuple[int, str, str]:
        """Run ``tmux <args>`` and return (returncode, stdout, stderr)."""
        return await _tmux(*args)

    async def available(self) -> Tuple[bool, str]:
        """Return (ok, detail). ok=True only if tmux + the target session exist."""
        code, _, err = await self._run("has-session", "-t", self._session_name())
        if code == 127:
            return False, "tmux is not installed on the bot host"
        if code != 0:
            return False, (
                err.strip() or f"tmux target '{self.target}' not found"
            )
        return True, self.target

    async def pane_cwd(self) -> Optional[str]:
        """Return the current working directory of the target pane, if any."""
        code, out, _ = await self._run(
            "display-message", "-p", "-t", self.target, "#{pane_current_path}"
        )
        cwd = out.strip()
        return cwd if code == 0 and cwd else None

    async def pane_command(self) -> Optional[str]:
        """Return the foreground command name in the target pane (e.g. 'claude')."""
        code, out, _ = await self._run(
            "display-message", "-p", "-t", self.target, "#{pane_current_command}"
        )
        cmd = out.strip()
        return cmd if code == 0 and cmd else None

    async def send_text(self, text: str) -> None:
        """Type ``text`` into the pane and submit it as its own message.

        Clears the composer first (C-u) so anything already in the input box --
        e.g. a queued message Esc/stop pushed back out of the queue -- can't
        merge with this one, and pauses before Enter so the TUI registers the
        text as typed input rather than a paste whose trailing newline just
        inserts a line break.
        """
        async with self._send_lock():
            # Clear leftover text in the composer (no-op when already empty).
            await self._run("send-keys", "-t", self.target, "C-u")
            # -l = literal: don't interpret the text as tmux key names.
            await self._run("send-keys", "-t", self.target, "-l", text)
            await asyncio.sleep(0.2)
            # Enter as a separate, non-literal keypress submits the prompt.
            await self._run("send-keys", "-t", self.target, "Enter")

    async def send_key(self, key: str) -> None:
        """Send a single named key (Enter, Escape, C-c, Up, y, ...)."""
        await self._run("send-keys", "-t", self.target, key)

    async def capture(self, lines: Optional[int] = None) -> str:
        """Return a rendered snapshot of the pane (ANSI stripped).

        With ``lines``, include that many lines of scrollback above the
        current screen.
        """
        args = ["capture-pane", "-p", "-t", self.target]
        if lines:
            args += ["-S", f"-{int(lines)}"]
        _, out, _ = await self._run(*args)
        # The TUI pads the screen with trailing blank lines -- trim them.
        return out.rstrip("\n")

    async def capture_settled(
        self,
        *,
        poll_interval: float = 1.5,
        stable_polls: int = 2,
        timeout: float = 45.0,
    ) -> str:
        """Snapshot repeatedly until the pane stops changing, then return it.

        Claude's spinner keeps the screen mutating while it works, so a run of
        identical captures is a good "it's idle now" signal. Bounded by
        ``timeout`` so a stuck/long turn still returns the latest snapshot.
        """
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        prev = await self.capture()
        stable = 0
        while loop.time() < deadline:
            await asyncio.sleep(poll_interval)
            cur = await self.capture()
            if cur == prev:
                stable += 1
                if stable >= stable_polls:
                    return cur
            else:
                stable = 0
                prev = cur
        return prev
