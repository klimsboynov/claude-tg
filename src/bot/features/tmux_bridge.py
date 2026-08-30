"""Bridge Telegram to interactive Claude Code sessions in tmux — local or remote.

Instead of spawning a headless ``claude -p`` per message, the bridge types the
message straight into a long-lived interactive ``claude`` running in a named
tmux pane -- so the phone and the PC drive the *same* process and the *same*
session, with no jsonl forking.

Multi-server: targets may be host-qualified as ``host/session`` where ``host``
is one of the SSH aliases configured via ``TMUX_REMOTE_HOSTS`` (auth/port/user
come from ``~/.ssh/config``). Every tmux call for such a target runs over
``ssh host tmux ...`` with connection multiplexing (ControlMaster), so the
~2s capture polls don't pay per-command handshakes. Bare targets stay local.

Output is read back on demand via ``capture-pane`` (a rendered screen snapshot,
ANSI stripped), never live-streamed: Claude's TUI redraws constantly, and a
snapshot keeps that noise out of Telegram.
"""

import asyncio
import os
import re
import shlex
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import structlog

logger = structlog.get_logger()

# SSH options shared by every remote call: key-only auth (never prompt), quick
# connect timeout, TOFU host keys, and multiplexing so repeated tmux calls
# reuse one TCP/auth session (ControlPersist keeps the master warm).
SSH_OPTS: List[str] = [
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=5",
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ServerAliveInterval=15",
    "-o", "ControlMaster=auto",
    "-o", "ControlPath=~/.ssh/tbot-cm-%C",
    "-o", "ControlPersist=120",
]

# SSH host aliases the bridge may drive; set from settings at startup.
_REMOTE_HOSTS: List[str] = []


def configure_remote_hosts(hosts: Optional[List[str]]) -> None:
    """Install the allowed remote host list (from TMUX_REMOTE_HOSTS)."""
    global _REMOTE_HOSTS
    _REMOTE_HOSTS = list(hosts or [])


def remote_hosts() -> List[str]:
    return list(_REMOTE_HOSTS)


def split_host(target: str) -> Tuple[Optional[str], str]:
    """Split a qualified target ``host/session`` into (host, tmux_target).

    Only prefixes matching a configured remote host count -- a local session
    that happens to contain ``/`` keeps working as long as its prefix isn't a
    configured host alias.
    """
    if "/" in target:
        host, rest = target.split("/", 1)
        if rest and host in _REMOTE_HOSTS:
            return host, rest
    return None, target


async def _exec(*argv: str) -> Tuple[int, str, str]:
    """Run argv and return (returncode, stdout, stderr)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return 127, "", f"{argv[0]} is not installed"
    out, err = await proc.communicate()
    return (
        proc.returncode or 0,
        out.decode("utf-8", "replace"),
        err.decode("utf-8", "replace"),
    )


async def _exec_bytes(*argv: str) -> Tuple[int, bytes]:
    """Run argv and return (returncode, raw stdout bytes)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return 127, b""
    out, _ = await proc.communicate()
    return proc.returncode or 0, out


def _ssh_argv(host: str, command: str) -> List[str]:
    """argv for running a shell command string on a remote host."""
    return ["ssh", *SSH_OPTS, host, command]


async def _tmux(*args: str, host: Optional[str] = None) -> Tuple[int, str, str]:
    """Run ``tmux <args>`` locally or on ``host`` over ssh."""
    if host is None:
        return await _exec("tmux", *args)
    # ssh joins argv into one remote shell string -> quote each tmux arg.
    cmd = "tmux " + " ".join(shlex.quote(a) for a in args)
    return await _exec(*_ssh_argv(host, cmd))


def _parse_sessions(out: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for line in out.splitlines():
        # Numeric fields first, name LAST with maxsplit -- so a session name
        # containing the delimiter still parses.
        parts = line.split("|", 3)
        if len(parts) < 4:
            continue
        windows, attached, activity, name = parts[0], parts[1], parts[2], parts[3]
        rows.append(
            {
                "name": name,
                "windows": int(windows) if windows.isdigit() else 0,
                "attached": attached == "1",
                "activity": int(activity) if activity.isdigit() else 0,
            }
        )
    return rows


# Printable delimiter: tmux replaces control chars (e.g. TAB) with ``_`` when
# the client locale is C -- which is exactly what a non-login ssh exec gets.
_LIST_FMT = (
    "#{session_windows}|#{session_attached}|"
    "#{session_activity}|#{session_name}"
)


async def list_sessions(host: Optional[str] = None) -> List[Dict[str, object]]:
    """List live tmux sessions on one host (local by default), newest first."""
    code, out, _ = await _tmux("list-sessions", "-F", _LIST_FMT, host=host)
    if code != 0:
        return []
    rows = _parse_sessions(out)
    rows.sort(key=lambda r: r["activity"], reverse=True)  # type: ignore[arg-type]
    return rows


async def list_all_sessions() -> Tuple[List[Dict[str, object]], List[str]]:
    """Sessions across local + every configured remote host, in parallel.

    Remote entries get ``name`` qualified as ``host/session`` and a ``host``
    key. Returns (sessions, unreachable_hosts); a host that errors or takes
    >6s is skipped and reported rather than hanging the picker.
    """

    async def _one(host: Optional[str]) -> List[Dict[str, object]]:
        rows = await asyncio.wait_for(list_sessions(host), timeout=6.0)
        for r in rows:
            if host:
                r["host"] = host
                r["name"] = f"{host}/{r['name']}"
        return rows

    hosts: List[Optional[str]] = [None, *(_REMOTE_HOSTS)]
    results = await asyncio.gather(*(_one(h) for h in hosts), return_exceptions=True)
    sessions: List[Dict[str, object]] = []
    unreachable: List[str] = []
    for host, res in zip(hosts, results):
        if isinstance(res, BaseException):
            if host:
                unreachable.append(host)
                logger.warning(
                    "remote tmux host unreachable", host=host, error=str(res)
                )
            continue
        sessions.extend(res)
    sessions.sort(key=lambda r: r["activity"], reverse=True)  # type: ignore[arg-type]
    return sessions, unreachable


class HostFS:
    """Minimal file reads on a host: local pathlib or the same over ssh.

    Used by the jsonl mirror to tail Claude session logs that live on the
    machine the tmux session runs on. Offsets are byte offsets throughout.
    """

    _homes: Dict[str, str] = {}  # host -> cached $HOME

    def __init__(self, host: Optional[str]) -> None:
        self.host = host

    async def home(self) -> Optional[str]:
        if self.host is None:
            return str(Path.home())
        cached = HostFS._homes.get(self.host)
        if cached:
            return cached
        code, out, _ = await _exec(*_ssh_argv(self.host, 'printf %s "$HOME"'))
        home = out.strip()
        if code != 0 or not home:
            return None
        HostFS._homes[self.host] = home
        return home

    async def project_dir(self, cwd: str) -> Optional[str]:
        """Claude's session-store dir for ``cwd`` on this host.

        Claude encodes the absolute cwd by replacing every non-alphanumeric
        character with ``-`` under ``~/.claude/projects/``.
        """
        home = await self.home()
        if not home:
            return None
        encoded = re.sub(r"[^A-Za-z0-9]", "-", cwd)
        return f"{home}/.claude/projects/{encoded}"

    async def newest_jsonl(self, proj_dir: Optional[str]) -> Optional[str]:
        """Most recently modified .jsonl in ``proj_dir``, if any."""
        if not proj_dir:
            return None
        if self.host is None:
            try:
                files = list(Path(proj_dir).glob("*.jsonl"))
            except OSError:
                return None
            if not files:
                return None
            return str(max(files, key=lambda p: p.stat().st_mtime))
        cmd = f"ls -1t {shlex.quote(proj_dir)}/*.jsonl 2>/dev/null | head -1"
        code, out, _ = await _exec(*_ssh_argv(self.host, cmd))
        path = out.strip()
        return path if code == 0 and path else None

    async def size(self, path: str) -> Optional[int]:
        if self.host is None:
            try:
                return os.stat(path).st_size
            except OSError:
                return None
        cmd = f"stat -c %s -- {shlex.quote(path)} 2>/dev/null"
        code, out, _ = await _exec(*_ssh_argv(self.host, cmd))
        out = out.strip()
        return int(out) if code == 0 and out.isdigit() else None

    async def read_from(self, path: str, offset: int) -> Tuple[str, int]:
        """Read from byte ``offset`` to EOF -> (text, bytes_consumed)."""
        if self.host is None:
            try:
                with open(path, "rb") as f:
                    f.seek(offset)
                    raw = f.read()
            except OSError:
                return "", 0
            return raw.decode("utf-8", "replace"), len(raw)
        # tail -c +N is 1-based: +offset+1 skips exactly ``offset`` bytes.
        cmd = f"tail -c +{offset + 1} -- {shlex.quote(path)} 2>/dev/null"
        code, raw = await _exec_bytes(*_ssh_argv(self.host, cmd))
        if code != 0:
            return "", 0
        return raw.decode("utf-8", "replace"), len(raw)


class TmuxBridge:
    """Types into and snapshots an interactive Claude session in tmux.

    ``target`` is a tmux target -- a session name (``claude``) or a fully
    qualified ``session:window.pane`` -- optionally prefixed ``host/`` to
    address a configured remote host over ssh.
    """

    # Serialize the multi-step send_text sequence per target across all bridge
    # instances (callers build a fresh TmuxBridge each time), so concurrent
    # sends -- e.g. a backgrounded file-delivery note racing a typed message --
    # can't interleave their send-keys and corrupt the composer.
    _send_locks: Dict[str, asyncio.Lock] = {}

    def __init__(self, target: str) -> None:
        self.qualified = target
        self.host, self.target = split_host(target)

    def _send_lock(self) -> asyncio.Lock:
        lock = TmuxBridge._send_locks.get(self.qualified)
        if lock is None:
            lock = TmuxBridge._send_locks[self.qualified] = asyncio.Lock()
        return lock

    def _session_name(self) -> str:
        """has-session wants a session, not a pane; drop any :win.pane suffix."""
        return self.target.split(":", 1)[0]

    async def _run(self, *args: str) -> Tuple[int, str, str]:
        """Run ``tmux <args>`` on this bridge's host."""
        return await _tmux(*args, host=self.host)

    async def available(self) -> Tuple[bool, str]:
        """Return (ok, detail). ok=True only if tmux + the target session exist."""
        code, _, err = await self._run("has-session", "-t", self._session_name())
        if code == 127:
            where = f"host '{self.host}'" if self.host else "the bot host"
            return False, f"tmux is not installed on {where}"
        if code == 255 and self.host:
            return False, (
                f"ssh to '{self.host}' failed: {err.strip() or 'unreachable'}"
            )
        if code != 0:
            return False, (
                err.strip() or f"tmux target '{self.qualified}' not found"
            )
        return True, self.qualified

    async def pane_cwd(self) -> Optional[str]:
        """Current working directory of the target pane (on its host)."""
        code, out, _ = await self._run(
            "display-message", "-p", "-t", self.target, "#{pane_current_path}"
        )
        cwd = out.strip()
        return cwd if code == 0 and cwd else None

    async def pane_command(self) -> Optional[str]:
        """Foreground command name in the target pane (e.g. 'claude')."""
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

    async def push_file(self, local_path: str, remote_dest: str) -> Tuple[bool, str]:
        """Copy a local file to ``remote_dest`` on this bridge's host via scp.

        Local bridges shouldn't call this (files are staged directly).
        """
        if self.host is None:
            return False, "push_file called on a local bridge"
        code, _, err = await _exec(
            "scp", *SSH_OPTS, "-q", local_path,
            f"{self.host}:{shlex.quote(remote_dest)}",
        )
        if code != 0:
            return False, err.strip() or f"scp exited {code}"
        return True, remote_dest
