"""Message orchestrator — single entry point for all Telegram updates.

Routes messages based on agentic vs classic mode. In agentic mode, provides
a minimal conversational interface (3 commands, no inline keyboards). In
classic mode, delegates to existing full-featured handlers.
"""

import asyncio
import json
import os
import re
import shutil
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import structlog
from telegram import (
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from ..config.settings import Settings
from ..projects import PrivateTopicsUnavailableError
from .features.claude_jsonl import (
    is_interactive_prompt,
    is_question_widget,
    parse_menu,
    prompt_signature,
    render_record,
    user_prompt_text,
)
from .features.tmux_bridge import (
    HostFS,
    TmuxBridge,
    configure_remote_hosts,
    split_host,
)
from .features.tmux_bridge import list_all_sessions as tmux_list_all_sessions
from .features.usage_render import render_usage
from .utils.html_format import escape_html

logger = structlog.get_logger()

# Binding key: (chat_id, message_thread_id|None). In a forum supergroup each
# topic is its own key -- one bound tmux session per topic ("tabs"); a DM (or
# non-forum group) keys as (chat_id, None) exactly like before.
BindingKey = Tuple[int, Optional[int]]


def _claude_project_dir(directory: Path) -> Path:
    """Map a working directory to Claude Code's session store directory.

    Claude Code encodes the absolute cwd by replacing ``/`` and ``.`` with
    ``-`` under ``~/.claude/projects/``.
    """
    encoded = str(directory).replace("/", "-").replace(".", "-")
    return Path.home() / ".claude" / "projects" / encoded


def _session_label(path: Path, max_lines: int = 80, maxlen: int = 70) -> str:
    """Best-effort human label for a Claude session .jsonl file.

    Prefers a stored ``summary`` entry, otherwise the first real user prompt —
    skipping ``isMeta`` messages and slash-command / local-command wrappers so
    the label reflects what the user actually asked. Reads only the first lines
    so it stays cheap on multi-MB session files.
    """
    import json

    def extract_text(obj: Dict[str, Any]) -> str:
        content = (obj.get("message") or {}).get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    return str(block.get("text", ""))
        return ""

    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for _ in range(max_lines):
                line = fh.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("type") == "summary" and obj.get("summary"):
                    return _truncate_label(str(obj["summary"]), maxlen)
                if obj.get("type") != "user" or obj.get("isMeta"):
                    continue
                text = extract_text(obj).lstrip()
                if not text or text.startswith(
                    (
                        "<local-command-",
                        "<command-name>",
                        "<command-message>",
                        "<command-args>",
                    )
                ):
                    continue
                # Strip any residual markup tags (e.g. embedded command output).
                text = " ".join(re.sub(r"<[^>]+>", " ", text).split())
                if text:
                    return _truncate_label(text, maxlen)
    except OSError:
        pass
    return ""


def _truncate_label(text: str, maxlen: int) -> str:
    """Collapse whitespace and truncate to maxlen with an ellipsis."""
    text = " ".join(text.split())
    if len(text) <= maxlen:
        return text
    return text[: maxlen - 1].rstrip() + "…"


def _humanize_age(epoch: float) -> str:
    """Relative time like '2 minutes ago' from an epoch timestamp."""
    secs = max(0, int(time.time() - epoch))
    if secs < 60:
        return f"{secs}s ago"
    for unit, size in (("minute", 60), ("hour", 3600), ("day", 86400)):
        val = secs // size
        if size == 60 or secs < size * 24 or unit == "day":
            plural = "s" if val != 1 else ""
            return f"{val} {unit}{plural} ago"
    return "a while ago"


def _humanize_size(num: float) -> str:
    """Human-readable byte size (e.g. '4.3MB')."""
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024:
            return f"{num:.0f}{unit}" if unit == "B" else f"{num:.1f}{unit}"
        num /= 1024
    return f"{num:.1f}TB"


def _list_claude_sessions(directory: Path, limit: int = 8) -> List[Dict[str, Any]]:
    """List resumable Claude sessions for a directory, newest first."""
    proj = _claude_project_dir(directory)
    if not proj.is_dir():
        return []
    files = [f for f in proj.glob("*.jsonl") if f.is_file()]
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    out: List[Dict[str, Any]] = []
    for f in files[:limit]:
        st = f.stat()
        out.append(
            {
                "session_id": f.stem,
                "mtime": st.st_mtime,
                "size": st.st_size,
                "label": _session_label(f),
            }
        )
    return out


# Patterns that look like secrets/credentials in CLI arguments
_SECRET_PATTERNS: List[re.Pattern[str]] = [
    # API keys / tokens (sk-ant-..., sk-..., ghp_..., gho_..., github_pat_..., xoxb-...)
    re.compile(
        r"(sk-ant-api\d*-[A-Za-z0-9_-]{10})[A-Za-z0-9_-]*"
        r"|(sk-[A-Za-z0-9_-]{20})[A-Za-z0-9_-]*"
        r"|(ghp_[A-Za-z0-9]{5})[A-Za-z0-9]*"
        r"|(gho_[A-Za-z0-9]{5})[A-Za-z0-9]*"
        r"|(github_pat_[A-Za-z0-9_]{5})[A-Za-z0-9_]*"
        r"|(xoxb-[A-Za-z0-9]{5})[A-Za-z0-9-]*"
    ),
    # AWS access keys
    re.compile(r"(AKIA[0-9A-Z]{4})[0-9A-Z]{12}"),
    # Generic long hex/base64 tokens after common flags/env patterns
    re.compile(
        r"((?:--token|--secret|--password|--api-key|--apikey|--auth)"
        r"[= ]+)['\"]?[A-Za-z0-9+/_.:-]{8,}['\"]?"
    ),
    # Inline env assignments like KEY=value
    re.compile(
        r"((?:TOKEN|SECRET|PASSWORD|API_KEY|APIKEY|AUTH_TOKEN|PRIVATE_KEY"
        r"|ACCESS_KEY|CLIENT_SECRET|WEBHOOK_SECRET)"
        r"=)['\"]?[^\s'\"]{8,}['\"]?"
    ),
    # Bearer / Basic auth headers
    re.compile(r"(Bearer )[A-Za-z0-9+/_.:-]{8,}" r"|(Basic )[A-Za-z0-9+/=]{8,}"),
    # Connection strings with credentials  user:pass@host
    re.compile(r"://([^:]+:)[^@]{4,}(@)"),
]


def _redact_secrets(text: str) -> str:
    """Replace likely secrets/credentials with redacted placeholders."""
    result = text
    for pattern in _SECRET_PATTERNS:
        result = pattern.sub(
            lambda m: next((g + "***" for g in m.groups() if g is not None), "***"),
            result,
        )
    return result


# Tool name -> friendly emoji mapping for verbose output
_TOOL_ICONS: Dict[str, str] = {
    "Read": "\U0001f4d6",
    "Write": "\u270f\ufe0f",
    "Edit": "\u270f\ufe0f",
    "MultiEdit": "\u270f\ufe0f",
    "Bash": "\U0001f4bb",
    "Glob": "\U0001f50d",
    "Grep": "\U0001f50d",
    "LS": "\U0001f4c2",
    "Task": "\U0001f9e0",
    "TaskOutput": "\U0001f9e0",
    "WebFetch": "\U0001f310",
    "WebSearch": "\U0001f310",
    "NotebookRead": "\U0001f4d3",
    "NotebookEdit": "\U0001f4d3",
    "TodoRead": "\u2611\ufe0f",
    "TodoWrite": "\u2611\ufe0f",
}


def _tool_icon(name: str) -> str:
    """Return emoji for a tool, with a default wrench."""
    return _TOOL_ICONS.get(name, "\U0001f527")


class MessageOrchestrator:
    """Routes messages based on mode. Single entry point for all Telegram updates."""

    def __init__(self, settings: Settings, deps: Dict[str, Any]):
        self.settings = settings
        self.deps = deps
        # Remote tmux hosts (host/session targets) the bridge may drive.
        configure_remote_hosts(settings.tmux_remote_hosts)
        self._known_commands: frozenset[str] = frozenset()
        # bkey -> background task tailing a Claude session .jsonl into the chat
        self._jsonl_mirrors: Dict[BindingKey, "asyncio.Task[None]"] = {}
        # bkey -> recently phone-sent prompts, to suppress their mirror echo
        self._mirror_echo: Dict[BindingKey, List[str]] = {}
        # bkey -> signature of the interactive prompt already surfaced, shared
        # between the mirror watcher and the widget-remote key callbacks so a
        # phone keypress editing the snapshot doesn't get re-posted as new.
        self._last_prompt: Dict[BindingKey, Optional[str]] = {}
        # bkey -> {"target": str, "mode": "mirror"|"snapshot"}, persisted so
        # tmux bindings + live mirrors survive a bot restart.
        self._bindings: Dict[BindingKey, Dict[str, str]] = self._load_bindings()

    # --- persistent tmux bindings ---

    def _bindings_path(self) -> Path:
        """JSON file for tmux bindings, kept beside the SQLite DB."""
        url = self.settings.database_url
        db = url[len("sqlite:///") :] if url.startswith("sqlite:///") else "data/bot.db"
        parent = Path(db).parent if db else Path("data")
        return parent / "tmux_bindings.json"

    def _load_bindings(self) -> Dict[BindingKey, Dict[str, str]]:
        """Load bindings; JSON keys are "chat" (legacy, no topic) or
        "chat:thread" (a forum topic)."""
        try:
            raw = json.loads(self._bindings_path().read_text(encoding="utf-8"))
            out: Dict[BindingKey, Dict[str, str]] = {}
            for k, v in raw.items():
                if ":" in k:
                    c, t = k.split(":", 1)
                    out[(int(c), int(t))] = v
                else:
                    out[(int(k), None)] = v
            return out
        except (OSError, ValueError, TypeError):
            return {}

    def _save_bindings(self) -> None:
        try:
            path = self._bindings_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                (f"{c}:{t}" if t is not None else str(c)): v
                for (c, t), v in self._bindings.items()
            }
            path.write_text(json.dumps(data), encoding="utf-8")
        except OSError as e:
            logger.warning("Failed to persist tmux bindings", error=str(e))

    def _bkey(self, update: Update) -> BindingKey:
        """Binding key for this update: (chat_id, topic thread id or None)."""
        return (
            update.effective_chat.id,
            self._extract_message_thread_id(update),
        )

    def _binding(self, bkey: BindingKey) -> Optional[Dict[str, str]]:
        return self._bindings.get(bkey)

    def _set_binding(self, bkey: BindingKey, target: str, mode: str) -> None:
        self._bindings[bkey] = {"target": target, "mode": mode}
        self._save_bindings()

    def _clear_binding(self, bkey: BindingKey) -> None:
        if self._bindings.pop(bkey, None) is not None:
            self._save_bindings()

    async def restore_mirrors(self, bot: Any) -> None:
        """Re-establish live mirrors for persisted bindings after a restart."""
        for bkey, b in list(self._bindings.items()):
            if b.get("mode") != "mirror":
                continue
            try:
                bridge = self._tmux(b["target"])
                ok, _ = await bridge.available()
                cwd = await bridge.pane_cwd() if ok else None
                fs = HostFS(bridge.host)
                proj_dir = await fs.project_dir(cwd) if cwd else None
            except Exception:
                cwd = None
                proj_dir = None
            if cwd and proj_dir:
                self._start_jsonl_mirror(
                    bkey,
                    bot,
                    b["target"],
                    proj_dir,
                    await fs.newest_jsonl(proj_dir),
                )
                logger.info(
                    "Restored tmux mirror",
                    chat_id=bkey[0],
                    thread_id=bkey[1],
                    target=b["target"],
                )
                try:
                    await bot.send_message(
                        chat_id=bkey[0],
                        message_thread_id=bkey[1],
                        text=(
                            "🔄 Reconnected live mirror to "
                            f"<code>{escape_html(b['target'])}</code> "
                            "after a restart."
                        ),
                        parse_mode="HTML",
                    )
                except Exception:
                    pass
            else:
                # Session/pane gone -> drop the stale binding.
                self._clear_binding(bkey)

    def _inject_deps(self, handler: Callable) -> Callable:  # type: ignore[type-arg]
        """Wrap handler to inject dependencies into context.bot_data."""

        async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            for key, value in self.deps.items():
                context.bot_data[key] = value
            context.bot_data["settings"] = self.settings
            context.user_data.pop("_thread_context", None)

            is_sync_bypass = handler.__name__ == "sync_threads"
            is_start_bypass = handler.__name__ in {"start_command", "agentic_start"}
            message_thread_id = self._extract_message_thread_id(update)
            should_enforce = self.settings.enable_project_threads

            if should_enforce:
                if self.settings.project_threads_mode == "private":
                    should_enforce = not is_sync_bypass and not (
                        is_start_bypass and message_thread_id is None
                    )
                else:
                    should_enforce = not is_sync_bypass

            if should_enforce:
                allowed = await self._apply_thread_routing_context(update, context)
                if not allowed:
                    return

            try:
                await handler(update, context)
            finally:
                if should_enforce:
                    self._persist_thread_state(context)

        return wrapped

    async def _apply_thread_routing_context(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> bool:
        """Enforce strict project-thread routing and load thread-local state."""
        manager = context.bot_data.get("project_threads_manager")
        if manager is None:
            await self._reject_for_thread_mode(
                update,
                "❌ <b>Project Thread Mode Misconfigured</b>\n\n"
                "Thread manager is not initialized.",
            )
            return False

        chat = update.effective_chat
        message = update.effective_message
        if not chat or not message:
            return False

        if self.settings.project_threads_mode == "group":
            if chat.id != self.settings.project_threads_chat_id:
                await self._reject_for_thread_mode(
                    update,
                    manager.guidance_message(mode=self.settings.project_threads_mode),
                )
                return False
        else:
            if getattr(chat, "type", "") != "private":
                await self._reject_for_thread_mode(
                    update,
                    manager.guidance_message(mode=self.settings.project_threads_mode),
                )
                return False

        message_thread_id = self._extract_message_thread_id(update)
        if not message_thread_id:
            await self._reject_for_thread_mode(
                update,
                manager.guidance_message(mode=self.settings.project_threads_mode),
            )
            return False

        project = await manager.resolve_project(chat.id, message_thread_id)
        if not project:
            await self._reject_for_thread_mode(
                update,
                manager.guidance_message(mode=self.settings.project_threads_mode),
            )
            return False

        state_key = f"{chat.id}:{message_thread_id}"
        thread_states = context.user_data.setdefault("thread_state", {})
        state = thread_states.get(state_key, {})

        project_root = project.absolute_path
        current_dir_raw = state.get("current_directory")
        current_dir = (
            Path(current_dir_raw).resolve() if current_dir_raw else project_root
        )
        if not self._is_within(current_dir, project_root) or not current_dir.is_dir():
            current_dir = project_root

        context.user_data["current_directory"] = current_dir
        context.user_data["claude_session_id"] = state.get("claude_session_id")
        context.user_data["_thread_context"] = {
            "chat_id": chat.id,
            "message_thread_id": message_thread_id,
            "state_key": state_key,
            "project_slug": project.slug,
            "project_root": str(project_root),
            "project_name": project.name,
        }
        return True

    def _persist_thread_state(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Persist compatibility keys back into per-thread state."""
        thread_context = context.user_data.get("_thread_context")
        if not thread_context:
            return

        project_root = Path(thread_context["project_root"])
        current_dir = context.user_data.get("current_directory", project_root)
        if not isinstance(current_dir, Path):
            current_dir = Path(str(current_dir))
        current_dir = current_dir.resolve()
        if not self._is_within(current_dir, project_root) or not current_dir.is_dir():
            current_dir = project_root

        thread_states = context.user_data.setdefault("thread_state", {})
        thread_states[thread_context["state_key"]] = {
            "current_directory": str(current_dir),
            "claude_session_id": context.user_data.get("claude_session_id"),
            "project_slug": thread_context["project_slug"],
        }

    @staticmethod
    def _is_within(path: Path, root: Path) -> bool:
        """Return True if path is within root."""
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    @staticmethod
    def _extract_message_thread_id(update: Update) -> Optional[int]:
        """Extract topic/thread id from update message for forum/direct topics."""
        message = update.effective_message
        if not message:
            return None
        message_thread_id = getattr(message, "message_thread_id", None)
        if isinstance(message_thread_id, int) and message_thread_id > 0:
            return message_thread_id
        dm_topic = getattr(message, "direct_messages_topic", None)
        topic_id = getattr(dm_topic, "topic_id", None) if dm_topic else None
        if isinstance(topic_id, int) and topic_id > 0:
            return topic_id
        # Telegram omits message_thread_id for the General topic in forum
        # supergroups; its canonical thread ID is 1.
        chat = update.effective_chat
        if chat and getattr(chat, "is_forum", False):
            return 1
        return None

    async def _reject_for_thread_mode(self, update: Update, message: str) -> None:
        """Send a guidance response when strict thread routing rejects an update."""
        query = update.callback_query
        if query:
            try:
                await query.answer()
            except Exception:
                pass
            if query.message:
                await query.message.reply_text(message, parse_mode="HTML")
            return

        if update.effective_message:
            await update.effective_message.reply_text(message, parse_mode="HTML")

    def register_handlers(self, app: Application) -> None:
        """Register handlers based on mode."""
        if self.settings.agentic_mode:
            self._register_agentic_handlers(app)
        else:
            self._register_classic_handlers(app)

    def _register_agentic_handlers(self, app: Application) -> None:
        """Register agentic handlers: commands + text/file/photo."""
        from .handlers import command

        # Commands
        handlers = [
            ("start", self.agentic_start),
            ("new", self.agentic_new),
            ("status", self.agentic_status),
            ("repo", self.agentic_repo),
            ("resume", self.agentic_resume),
            ("tmux", self.agentic_tmux),
            ("sync_tmux", self.agentic_sync_tmux),
            ("peek", self.agentic_peek),
            ("key", self.agentic_key),
            ("stop", self.agentic_stop),
            ("clear", self.agentic_clear),
            ("restart", command.restart_command),
        ]
        if self.settings.enable_project_threads:
            handlers.append(("sync_threads", command.sync_threads))

        # Derive known commands dynamically — avoids drift when new commands are added
        self._known_commands: frozenset[str] = frozenset(cmd for cmd, _ in handlers)

        for cmd, handler in handlers:
            app.add_handler(CommandHandler(cmd, self._inject_deps(handler)))

        # Text messages -> Claude
        app.add_handler(
            MessageHandler(
                filters.TEXT & ~filters.COMMAND,
                self._inject_deps(self.agentic_text),
            ),
            group=10,
        )

        # Unknown slash commands -> Claude (passthrough in agentic mode).
        # Registered commands are handled by CommandHandlers in group 0
        # (higher priority). This catches any /command not matched there
        # and forwards it to Claude, while skipping known commands to
        # avoid double-firing.
        app.add_handler(
            MessageHandler(
                filters.COMMAND,
                self._inject_deps(self._handle_unknown_command),
            ),
            group=10,
        )

        # File uploads -> Claude
        app.add_handler(
            MessageHandler(
                filters.Document.ALL, self._inject_deps(self.agentic_document)
            ),
            group=10,
        )

        # Photo uploads -> Claude
        app.add_handler(
            MessageHandler(filters.PHOTO, self._inject_deps(self.agentic_photo)),
            group=10,
        )

        # Video / animation / video-note uploads -> Claude
        app.add_handler(
            MessageHandler(
                filters.VIDEO | filters.VIDEO_NOTE | filters.ANIMATION,
                self._inject_deps(self.agentic_video),
            ),
            group=10,
        )

        # Voice messages -> transcribe -> Claude
        app.add_handler(
            MessageHandler(filters.VOICE, self._inject_deps(self.agentic_voice)),
            group=10,
        )

        # Only cd: callbacks (for project selection), scoped by pattern
        app.add_handler(
            CallbackQueryHandler(
                self._inject_deps(self._agentic_callback),
                pattern=r"^cd:",
            )
        )

        # resume: callbacks (session picker menu)
        app.add_handler(
            CallbackQueryHandler(
                self._inject_deps(self._handle_resume_callback),
                pattern=r"^resume:",
            )
        )

        # tmux tab-switch callbacks (session picker + unbind)
        app.add_handler(
            CallbackQueryHandler(
                self._inject_deps(self._handle_tmux_callback),
                pattern=r"^tmux(sel:|off)",
            )
        )

        # tmux menu-option callbacks (interactive prompt buttons)
        app.add_handler(
            CallbackQueryHandler(
                self._inject_deps(self._handle_tmux_option_callback),
                pattern=r"^tmuxopt:",
            )
        )
        app.add_handler(
            CallbackQueryHandler(
                self._inject_deps(self._handle_tmux_key_callback),
                pattern=r"^tmuxkey:",
            )
        )

        logger.info("Agentic handlers registered")

    def _register_classic_handlers(self, app: Application) -> None:
        """Register full classic handler set (moved from core.py)."""
        from .handlers import callback, command, message

        handlers = [
            ("start", command.start_command),
            ("help", command.help_command),
            ("new", command.new_session),
            ("continue", command.continue_session),
            ("end", command.end_session),
            ("ls", command.list_files),
            ("cd", command.change_directory),
            ("pwd", command.print_working_directory),
            ("projects", command.show_projects),
            ("status", command.session_status),
            ("export", command.export_session),
            ("actions", command.quick_actions),
            ("git", command.git_command),
            ("restart", command.restart_command),
        ]
        if self.settings.enable_project_threads:
            handlers.append(("sync_threads", command.sync_threads))

        for cmd, handler in handlers:
            app.add_handler(CommandHandler(cmd, self._inject_deps(handler)))

        app.add_handler(
            MessageHandler(
                filters.TEXT & ~filters.COMMAND,
                self._inject_deps(message.handle_text_message),
            ),
            group=10,
        )
        app.add_handler(
            MessageHandler(
                filters.Document.ALL, self._inject_deps(message.handle_document)
            ),
            group=10,
        )
        app.add_handler(
            MessageHandler(filters.PHOTO, self._inject_deps(message.handle_photo)),
            group=10,
        )
        app.add_handler(
            MessageHandler(filters.VOICE, self._inject_deps(message.handle_voice)),
            group=10,
        )
        app.add_handler(
            CallbackQueryHandler(self._inject_deps(callback.handle_callback_query))
        )

        logger.info("Classic handlers registered (13 commands + full handler set)")

    async def get_bot_commands(self) -> list:  # type: ignore[type-arg]
        """Return bot commands appropriate for current mode."""
        if self.settings.agentic_mode:
            commands = [
                BotCommand("start", "Start the bot"),
                BotCommand("new", "Start a fresh session"),
                BotCommand("status", "Show session status"),
                BotCommand("repo", "List repos / switch workspace"),
                BotCommand("resume", "Pick a session to resume"),
                BotCommand("tmux", "Pick/switch a live tmux session (tabs)"),
                BotCommand("sync_tmux", "Forum group: one topic per session"),
                BotCommand("peek", "Snapshot the bound tmux pane"),
                BotCommand("key", "Send a key to the bound pane (esc/y/c-c…)"),
                # Passthrough commands: no bot handler — forwarded into the bound
                # Claude session (autocomplete only). /usage now auto-dismisses.
                BotCommand("usage", "Usage & limits — runs in the bound session"),
                BotCommand("compact", "Compact the session's context (in-session)"),
                BotCommand("stop", "Interrupt the current turn (Esc/Ctrl-C)"),
                BotCommand("clear", "Delete recent messages in this chat"),
                BotCommand("restart", "Restart the bot"),
            ]
            if self.settings.enable_project_threads:
                commands.append(BotCommand("sync_threads", "Sync project topics"))
            return commands
        else:
            commands = [
                BotCommand("start", "Start bot and show help"),
                BotCommand("help", "Show available commands"),
                BotCommand("new", "Clear context and start fresh session"),
                BotCommand("continue", "Explicitly continue last session"),
                BotCommand("end", "End current session and clear context"),
                BotCommand("ls", "List files in current directory"),
                BotCommand("cd", "Change directory (resumes project session)"),
                BotCommand("pwd", "Show current directory"),
                BotCommand("projects", "Show all projects"),
                BotCommand("status", "Show session status"),
                BotCommand("export", "Export current session"),
                BotCommand("actions", "Show quick actions"),
                BotCommand("git", "Git repository commands"),
                BotCommand("restart", "Restart the bot"),
            ]
            if self.settings.enable_project_threads:
                commands.append(BotCommand("sync_threads", "Sync project topics"))
            return commands

    # --- Agentic handlers ---

    async def agentic_start(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Brief welcome, no buttons."""
        user = update.effective_user
        sync_line = ""
        if (
            self.settings.enable_project_threads
            and self.settings.project_threads_mode == "private"
        ):
            if (
                not update.effective_chat
                or getattr(update.effective_chat, "type", "") != "private"
            ):
                await update.message.reply_text(
                    "🚫 <b>Private Topics Mode</b>\n\n"
                    "Use this bot in a private chat and run <code>/start</code> there.",
                    parse_mode="HTML",
                )
                return
            manager = context.bot_data.get("project_threads_manager")
            if manager:
                try:
                    result = await manager.sync_topics(
                        context.bot,
                        chat_id=update.effective_chat.id,
                    )
                    sync_line = (
                        "\n\n🧵 Topics synced"
                        f" (created {result.created}, reused {result.reused})."
                    )
                except PrivateTopicsUnavailableError:
                    await update.message.reply_text(
                        manager.private_topics_unavailable_message(),
                        parse_mode="HTML",
                    )
                    return
                except Exception:
                    sync_line = "\n\n🧵 Topic sync failed. Run /sync_threads to retry."
        current_dir = context.user_data.get(
            "current_directory", self.settings.approved_directory
        )
        dir_display = f"<code>{current_dir}/</code>"

        safe_name = escape_html(user.first_name)
        await update.message.reply_text(
            f"Hi {safe_name}! I'm your AI coding assistant.\n"
            f"Just tell me what you need — I can read, write, and run code.\n\n"
            f"Working in: {dir_display}\n"
            f"Commands: /new (reset) · /status"
            f"{sync_line}",
            parse_mode="HTML",
        )

    async def agentic_new(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Reset session, one-line confirmation."""
        context.user_data["claude_session_id"] = None
        context.user_data["session_started"] = True
        context.user_data["force_new_session"] = True

        await update.message.reply_text("Session reset. What's next?")

    async def agentic_status(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Compact one-line status, no buttons."""
        current_dir = context.user_data.get(
            "current_directory", self.settings.approved_directory
        )
        dir_display = str(current_dir)

        session_id = context.user_data.get("claude_session_id")
        session_status = "active" if session_id else "none"

        # Cost info
        cost_str = ""
        rate_limiter = context.bot_data.get("rate_limiter")
        if rate_limiter:
            try:
                user_status = rate_limiter.get_user_status(update.effective_user.id)
                cost_usage = user_status.get("cost_usage", {})
                current_cost = cost_usage.get("current", 0.0)
                cost_str = f" · Cost: ${current_cost:.2f}"
            except Exception:
                pass

        await update.message.reply_text(
            f"📂 {dir_display} · Session: {session_status}{cost_str}"
        )




    @staticmethod
    def _summarize_tool_input(tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Return a short summary of tool input for verbose level 2."""
        if not tool_input:
            return ""
        if tool_name in ("Read", "Write", "Edit", "MultiEdit"):
            path = tool_input.get("file_path") or tool_input.get("path", "")
            if path:
                # Show just the filename, not the full path
                return path.rsplit("/", 1)[-1]
        if tool_name in ("Glob", "Grep"):
            pattern = tool_input.get("pattern", "")
            if pattern:
                return pattern[:60]
        if tool_name == "Bash":
            cmd = tool_input.get("command", "")
            if cmd:
                return _redact_secrets(cmd[:100])[:80]
        if tool_name in ("WebFetch", "WebSearch"):
            return (tool_input.get("url", "") or tool_input.get("query", ""))[:60]
        if tool_name == "Task":
            desc = tool_input.get("description", "")
            if desc:
                return desc[:60]
        # Generic: show first key's value
        for v in tool_input.values():
            if isinstance(v, str) and v:
                return v[:60]
        return ""

    async def agentic_text(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Direct Claude passthrough. Simple progress. No suggestions."""
        user_id = update.effective_user.id
        message_text = update.message.text

        logger.info(
            "Agentic text message",
            user_id=user_id,
            message_length=len(message_text),
        )

        # Rate limit check
        rate_limiter = context.bot_data.get("rate_limiter")
        if rate_limiter:
            allowed, limit_message = await rate_limiter.check_rate_limit(user_id, 0.001)
            if not allowed:
                await update.message.reply_text(f"⏱️ {limit_message}")
                return

        # tmux is the only path: route to the bound session, else ask to bind.
        if self._binding(self._bkey(update)):
            await self._tmux_send_and_report(update, context, message_text)
        else:
            await self._prompt_bind(update)
        return

    async def agentic_document(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Deliver an uploaded file into the bound tmux session's cwd."""
        binding = self._binding(self._bkey(update))
        if not binding:
            await self._prompt_bind(update)
            return
        await self._tmux_document(update, context, binding)
        return

    async def agentic_photo(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Deliver a photo into the bound tmux session's cwd."""
        binding = self._binding(self._bkey(update))
        if not binding:
            await self._prompt_bind(update)
            return
        await self._tmux_photo(update, context, binding)
        return

    async def agentic_video(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Deliver a video / animation / video-note into the bound session's cwd."""
        binding = self._binding(self._bkey(update))
        if not binding:
            await self._prompt_bind(update)
            return
        await self._tmux_video(update, context, binding)
        return

    async def agentic_voice(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Transcribe a voice message and route the text into the bound session."""
        binding = self._binding(self._bkey(update))
        if not binding:
            await self._prompt_bind(update)
            return
        await self._tmux_voice(update, context, binding)
        return


    async def _handle_unknown_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Forward unknown slash commands to Claude in agentic mode.

        Known commands are handled by their own CommandHandlers (group 0);
        this handler fires for *every* COMMAND message in group 10 but
        returns immediately when the command is registered, preventing
        double execution.
        """
        msg = update.effective_message
        if not msg or not msg.text:
            return
        cmd = msg.text.split()[0].lstrip("/").split("@")[0].lower()
        if cmd in self._known_commands:
            return  # let the registered CommandHandler take care of it
        # Forward unrecognised /commands to Claude as natural language
        await self.agentic_text(update, context)

    def _voice_unavailable_message(self) -> str:
        """Return provider-aware guidance when voice feature is unavailable."""
        if self.settings.voice_provider == "local":
            return (
                "Voice processing is not available. "
                "Ensure whisper.cpp is installed and the model file exists. "
                "Check WHISPER_CPP_BINARY_PATH and WHISPER_CPP_MODEL_PATH settings."
            )
        return (
            "Voice processing is not available. "
            f"Set {self.settings.voice_provider_api_key_env} "
            f"for {self.settings.voice_provider_display_name} and install "
            'voice extras with: pip install "claude-code-telegram[voice]"'
        )

    async def agentic_repo(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """List repos in workspace or switch to one.

        /repo          — list subdirectories with git indicators
        /repo <name>   — switch to that directory, resume session if available
        """
        args = update.message.text.split()[1:] if update.message.text else []
        base = self.settings.approved_directory
        current_dir = context.user_data.get("current_directory", base)

        if args:
            # Switch to named repo
            target_name = args[0]
            target_path = base / target_name
            if not target_path.is_dir():
                await update.message.reply_text(
                    f"Directory not found: <code>{escape_html(target_name)}</code>",
                    parse_mode="HTML",
                )
                return

            context.user_data["current_directory"] = target_path

            # Try to find a resumable session
            claude_integration = context.bot_data.get("claude_integration")
            session_id = None
            if claude_integration:
                existing = await claude_integration._find_resumable_session(
                    update.effective_user.id, target_path
                )
                if existing:
                    session_id = existing.session_id
            context.user_data["claude_session_id"] = session_id

            is_git = (target_path / ".git").is_dir()
            git_badge = " (git)" if is_git else ""
            session_badge = " · session resumed" if session_id else ""

            await update.message.reply_text(
                f"Switched to <code>{escape_html(target_name)}/</code>"
                f"{git_badge}{session_badge}",
                parse_mode="HTML",
            )
            return

        # No args — list repos
        try:
            entries = sorted(
                [
                    d
                    for d in base.iterdir()
                    if d.is_dir() and not d.name.startswith(".")
                ],
                key=lambda d: d.name,
            )
        except OSError as e:
            await update.message.reply_text(f"Error reading workspace: {e}")
            return

        if not entries:
            await update.message.reply_text(
                f"No repos in <code>{escape_html(str(base))}</code>.\n"
                'Clone one by telling me, e.g. <i>"clone org/repo"</i>.',
                parse_mode="HTML",
            )
            return

        lines: List[str] = []
        keyboard_rows: List[list] = []  # type: ignore[type-arg]
        current_name = current_dir.name if current_dir != base else None

        for d in entries:
            is_git = (d / ".git").is_dir()
            icon = "\U0001f4e6" if is_git else "\U0001f4c1"
            marker = " \u25c0" if d.name == current_name else ""
            lines.append(f"{icon} <code>{escape_html(d.name)}/</code>{marker}")

        # Build inline keyboard (2 per row)
        for i in range(0, len(entries), 2):
            row = []
            for j in range(2):
                if i + j < len(entries):
                    name = entries[i + j].name
                    row.append(InlineKeyboardButton(name, callback_data=f"cd:{name}"))
            keyboard_rows.append(row)

        reply_markup = InlineKeyboardMarkup(keyboard_rows)

        await update.message.reply_text(
            "<b>Repos</b>\n\n" + "\n".join(lines),
            parse_mode="HTML",
            reply_markup=reply_markup,
        )


    async def _agentic_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle cd: callbacks — switch directory and resume session if available."""
        query = update.callback_query
        await query.answer()

        data = query.data
        _, project_name = data.split(":", 1)

        base = self.settings.approved_directory
        new_path = base / project_name

        if not new_path.is_dir():
            await query.edit_message_text(
                f"Directory not found: <code>{escape_html(project_name)}</code>",
                parse_mode="HTML",
            )
            return

        context.user_data["current_directory"] = new_path

        # Look for a resumable session instead of always clearing
        claude_integration = context.bot_data.get("claude_integration")
        session_id = None
        if claude_integration:
            existing = await claude_integration._find_resumable_session(
                query.from_user.id, new_path
            )
            if existing:
                session_id = existing.session_id
        context.user_data["claude_session_id"] = session_id

        is_git = (new_path / ".git").is_dir()
        git_badge = " (git)" if is_git else ""
        session_badge = " · session resumed" if session_id else ""

        await query.edit_message_text(
            f"Switched to <code>{escape_html(project_name)}/</code>"
            f"{git_badge}{session_badge}",
            parse_mode="HTML",
        )

        # Audit log
        audit_logger = context.bot_data.get("audit_logger")
        if audit_logger:
            await audit_logger.log_command(
                user_id=query.from_user.id,
                command="cd",
                args=[project_name],
                success=True,
            )

    async def _try_delete(self, bot: Any, chat_id: int, message_id: int) -> bool:
        try:
            await bot.delete_message(chat_id=chat_id, message_id=message_id)
            return True
        except Exception:
            return False

    async def agentic_clear(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Delete recent messages in this Telegram chat (best-effort).

        Telegram only lets a bot delete messages it's allowed to and that are
        newer than ~48h, so this sweeps message IDs downward from the /clear
        command and stops once it hits a run of undeletable/older messages.
        """
        chat_id = update.effective_chat.id
        top = update.message.message_id
        bot = context.bot

        audit_logger = context.bot_data.get("audit_logger")
        if audit_logger:
            await audit_logger.log_command(
                user_id=update.effective_user.id,
                command="clear",
                args=[],
                success=True,
            )

        deleted = 0
        empty_waves = 0
        mid = top
        wave = 25
        while mid > 0 and empty_waves < 3 and (top - mid) < 3000:
            ids = list(range(mid, max(0, mid - wave), -1))
            mid -= wave
            results = await asyncio.gather(
                *[self._try_delete(bot, chat_id, i) for i in ids]
            )
            got = sum(results)
            deleted += got
            empty_waves = empty_waves + 1 if got == 0 else 0
            await asyncio.sleep(0.05)  # ease off Telegram's delete rate limit

        note = (
            ""
            if deleted
            else " Nothing deletable — Telegram only lets bots remove messages"
            " from the last ~48h."
        )
        confirm = await bot.send_message(
            chat_id=chat_id, text=f"🧹 Cleared {deleted} message(s).{note}"
        )
        # Tidy up the confirmation itself after a moment.
        if deleted:
            context.application.create_task(
                self._delete_later(bot, chat_id, confirm.message_id, 4.0)
            )

    async def _delete_later(
        self, bot: Any, chat_id: int, message_id: int, delay: float
    ) -> None:
        await asyncio.sleep(delay)
        await self._try_delete(bot, chat_id, message_id)

    async def agentic_stop(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Interrupt the bound session's current turn (Esc for Claude, Ctrl-C
        for a plain shell)."""
        bkey = self._bkey(update)
        binding = self._binding(bkey)

        audit_logger = context.bot_data.get("audit_logger")
        if audit_logger:
            await audit_logger.log_command(
                user_id=update.effective_user.id,
                command="stop",
                args=[binding["target"]] if binding else [],
                success=True,
            )

        if not binding:
            await self._prompt_bind(update)
            return
        target = binding["target"]
        bridge = self._tmux(target)
        ok, detail = await bridge.available()
        if not ok:
            self._clear_binding(bkey)
            self._stop_jsonl_mirror(bkey)
            await update.message.reply_text(
                f"⚠️ {escape_html(detail)}", parse_mode="HTML"
            )
            return
        key = "Escape" if binding.get("mode") == "mirror" else "C-c"
        label = "Esc" if key == "Escape" else "Ctrl-C"
        await bridge.send_key(key)
        await update.message.reply_text(
            f"🛑 Interrupt ({label}) sent to <code>{escape_html(target)}</code>.",
            parse_mode="HTML",
        )

    # --- tmux bridge (multi-session "tabs") ---

    def _tmux(self, target: str) -> TmuxBridge:
        return TmuxBridge(target)

    def _format_pane(self, snapshot: str, max_chars: int = 3500) -> str:
        """Render a pane snapshot as a Telegram-safe monospace block.

        The Claude Code Usage/Stats panel is parsed into a compact card instead
        of the raw (wide, misaligned) capture; everything else falls back to a
        <pre> block.
        """
        snapshot = snapshot.rstrip()
        if not snapshot:
            return "<i>(pane is empty)</i>"
        pretty = render_usage(snapshot)
        if pretty:
            return pretty
        if len(snapshot) > max_chars:
            # Terminal output: the tail is what just happened -> keep it.
            snapshot = "…\n" + snapshot[-max_chars:]
        return f"<pre>{escape_html(snapshot)}</pre>"

    async def _audit_tmux(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, action: str
    ) -> None:
        audit_logger = context.bot_data.get("audit_logger")
        if audit_logger:
            await audit_logger.log_command(
                user_id=update.effective_user.id,
                command="tmux",
                args=[action],
                success=True,
            )

    async def _tmux_session_keyboard(
        self, current: Optional[str]
    ) -> tuple[Optional[InlineKeyboardMarkup], List[str]]:
        """Inline 'tabs' keyboard of live tmux sessions on every host.

        Returns (keyboard, unreachable_hosts); keyboard is None when no
        sessions exist anywhere. Remote sessions show as host/session.
        """
        sessions, unreachable = await tmux_list_all_sessions()
        if not sessions:
            return None, unreachable
        rows: List[list] = []  # type: ignore[type-arg]
        for s in sessions:
            name = str(s["name"])
            data = f"tmuxsel:{name}"
            if len(data.encode()) > 64:  # Telegram callback_data cap
                continue
            mark = "🟢 " if name == current else "▫️ "
            att = " ·live" if s["attached"] else ""
            label = f"{mark}{name} ({s['windows']}w{att})"
            rows.append([InlineKeyboardButton(label, callback_data=data)])
        if current:
            rows.append([InlineKeyboardButton("🔌 Unbind", callback_data="tmuxoff")])
        return InlineKeyboardMarkup(rows), unreachable

    async def _bind_session(self, bot: Any, bkey: BindingKey, target: str) -> str:
        """Bind ``bkey`` (chat or forum topic) to a tmux session.

        If the pane is running a Claude session, starts a live jsonl mirror
        streaming into that chat/topic; otherwise snapshot mode. Returns the
        human-readable mode line.
        """
        self._stop_jsonl_mirror(bkey)
        bridge = self._tmux(target)
        cwd = await bridge.pane_cwd()
        cmd = await bridge.pane_command()
        # Session logs live on the host the pane runs on (local or remote).
        fs = HostFS(bridge.host)
        proj_dir = await fs.project_dir(cwd) if cwd else None
        session_file = await fs.newest_jsonl(proj_dir)
        # A pane running `claude` is a mirror target even before its first
        # prompt (no .jsonl yet) -- the loop waits for the log to appear.
        is_claude = bool(session_file) or (cmd or "").lower() == "claude"

        if is_claude and cwd and proj_dir:
            self._set_binding(bkey, target, "mirror")
            self._start_jsonl_mirror(bkey, bot, target, proj_dir, session_file)
            return (
                "🪞 <b>live mirror</b> on — I'll stream Claude's replies and "
                "tool calls here as they happen."
            )
        self._set_binding(bkey, target, "snapshot")
        return "📸 snapshot mode (not a Claude session) — /peek to refresh."

    async def _bind_and_snapshot(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        target: str,
        *,
        edit: bool = False,
    ) -> None:
        """Bind the chat/topic to a tmux session and reply with a snapshot."""
        bkey = self._bkey(update)
        await self._audit_tmux(update, context, f"bind:{target}")
        bridge = self._tmux(target)
        mode_line = await self._bind_session(context.bot, bkey, target)
        snapshot = await bridge.capture()
        text = (
            f"🔗 Bound to <code>{escape_html(target)}</code>. {mode_line}\n"
            "Plain text types into the session · /tmux to switch · /peek · /key.\n\n"
            f"{self._format_pane(snapshot)}"
        )
        if edit and update.callback_query:
            await update.callback_query.edit_message_text(text, parse_mode="HTML")
        else:
            await update.effective_message.reply_text(text, parse_mode="HTML")

    async def _tmux_send_and_report(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        text: str,
    ) -> None:
        """Type text into the bound tmux session.

        In mirror mode the background tailer surfaces the response, so we just
        type and return (claude queues input typed while it's busy). In
        snapshot mode we type and reply with a settled screen capture.
        """
        bkey = self._bkey(update)
        binding = self._binding(bkey)
        if not binding:
            return
        target = binding["target"]
        mode = binding.get("mode", "snapshot")
        bridge = self._tmux(target)
        ok, detail = await bridge.available()
        if not ok:
            self._clear_binding(bkey)
            self._stop_jsonl_mirror(bkey)
            await update.message.reply_text(
                f"⚠️ Session <code>{escape_html(target)}</code> is gone: "
                f"{escape_html(detail)}\nUnbound — /tmux to pick another.",
                parse_mode="HTML",
            )
            return

        await update.message.chat.send_action("typing")
        if mode == "mirror":
            # Remember what we typed so the mirror doesn't echo it back at us
            # (we already see it in the chat); PC-typed turns still echo.
            self._note_echo(bkey, text)
        await bridge.send_text(text)
        await self._audit_tmux(update, context, f"send:{target}")

        if mode == "mirror":
            # The mirror task streams Claude's output; nothing to capture here.
            return

        snapshot = await bridge.capture_settled()
        await update.message.reply_text(
            f"🖥 <code>{escape_html(target)}</code>\n{self._format_pane(snapshot)}",
            parse_mode="HTML",
        )

    async def _prompt_bind(self, update: Update) -> None:
        """Tell the user to bind a tmux session (the only input path now)."""
        await update.effective_message.reply_text(
            "Not bound to a tmux session. Use /tmux to pick one first."
        )

    async def _tmux_deliver_file(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        binding: Dict[str, str],
        media: Any,
        filename: str,
        caption: str,
    ) -> None:
        """Stage a Telegram file into the pane's cwd, off the update lock.

        The heavy work -- get_file() (which pulls the bytes when a local Bot
        API server is used) plus the hardlink/copy -- runs in a background
        task, so a big upload doesn't hold the sequential update lock and
        freeze every other message. An immediate "receiving" reply is edited
        in place to the final result.
        """
        bkey = self._bkey(update)
        target = binding["target"]
        bridge = self._tmux(target)
        ok, detail = await bridge.available()
        if not ok:
            self._clear_binding(bkey)
            self._stop_jsonl_mirror(bkey)
            await update.message.reply_text(
                f"⚠️ {escape_html(detail)}", parse_mode="HTML"
            )
            return
        cwd = await bridge.pane_cwd() or str(self.settings.approved_directory)
        safe = os.path.basename(filename) or "upload.bin"
        dest = Path(cwd) / safe
        size = getattr(media, "file_size", None)
        if size and size >= 1024**3:
            human = f" ({size / 1024**3:.1f}GB)"
        elif size:
            human = f" ({size / 1024**2:.1f}MB)"
        else:
            human = ""
        status = await update.message.reply_text(
            f"⏳ receiving <code>{escape_html(safe)}</code>{human}…",
            parse_mode="HTML",
        )
        # Off the sequential update lock: fetch + stage, then notify.
        context.application.create_task(
            self._deliver_file_bg(
                update, context, binding, media, safe, dest, caption, status
            ),
            update=update,
        )

    async def _deliver_file_bg(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        binding: Dict[str, str],
        media: Any,
        safe: str,
        dest: Path,
        caption: str,
        status: Any,
    ) -> None:
        """Background worker: pull the file, hardlink/copy it into the pane's
        cwd, then inject the path into the session and edit the reply."""
        bkey = self._bkey(update)
        target = binding["target"]
        host = split_host(target)[0]
        try:
            tg_file = await media.get_file()
            if host:
                # Remote pane: stage locally (hardlink when possible), then
                # scp into the pane's cwd on its host, then drop the staging.
                tmpdir = tempfile.mkdtemp(prefix="tbot-push-")
                tmp = Path(tmpdir) / safe
                try:
                    await self._stage_file(tg_file, tmp)
                    ok, detail = await self._tmux(target).push_file(
                        str(tmp), str(dest)
                    )
                finally:
                    shutil.rmtree(tmpdir, ignore_errors=True)
                if not ok:
                    raise RuntimeError(f"scp to {host} failed: {detail}")
            else:
                await self._stage_file(tg_file, dest)
        except Exception as e:
            logger.warning("tmux file receive failed", error=str(e), file=safe)
            if "too big" in str(e).lower():
                msg = (
                    "⚠️ Telegram refused it: exceeds the Bot API 20MB getFile "
                    "limit. Run a local Bot API server (deploy/LOCAL_BOTAPI.md) "
                    "for up to 2GB, or side-channel the file into the cwd."
                )
            else:
                msg = "⚠️ Failed to receive the file."
            try:
                await status.edit_text(msg)
            except Exception:
                pass
            return
        flat = " ".join((caption or "").split())
        note = f"[uploaded file saved to: {dest}]"
        typed = f"{flat} {note}".strip() if flat else note
        if binding.get("mode") == "mirror":
            self._note_echo(bkey, typed)
        try:
            await self._tmux(target).send_text(typed)
            await self._audit_tmux(update, context, f"file:{target}")
            await status.edit_text(
                f"📎 <code>{escape_html(safe)}</code> → "
                f"<code>{escape_html(target)}</code>\n"
                f"<code>{escape_html(str(dest))}</code>",
                parse_mode="HTML",
            )
        except Exception as e:
            logger.warning("tmux file notify failed", error=str(e), file=safe)

    async def _stage_file(self, tg_file: Any, dest: Path) -> None:
        """Hardlink the local-server file into dest (instant, no extra disk);
        fall back to a real download/copy across filesystems or in cloud mode."""
        src = getattr(tg_file, "file_path", None)
        if src and os.path.isabs(str(src)) and os.path.exists(src):
            try:
                if dest.exists():
                    dest.unlink()
                os.link(src, dest)
                return
            except OSError:
                pass  # EXDEV (cross-filesystem) or perms -> fall back to copy
        await tg_file.download_to_drive(str(dest))

    async def _tmux_document(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        binding: Dict[str, str],
    ) -> None:
        # No filename restrictions on bridge uploads — the file is only stored
        # in the session cwd (basename-confined in _tmux_deliver_file), never
        # executed or processed by type.
        document = update.message.document
        await self._tmux_deliver_file(
            update,
            context,
            binding,
            document,
            document.file_name or "upload.bin",
            update.message.caption or "",
        )

    async def _tmux_photo(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        binding: Dict[str, str],
    ) -> None:
        photo = update.message.photo[-1]
        fname = f"photo_{update.message.message_id}.jpg"
        await self._tmux_deliver_file(
            update, context, binding, photo, fname, update.message.caption or ""
        )

    async def _tmux_video(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        binding: Dict[str, str],
    ) -> None:
        message = update.message
        media = message.video or message.animation or message.video_note
        if media is None:
            return
        filename = getattr(media, "file_name", None) or (
            f"video_{message.message_id}.mp4"
        )
        await self._tmux_deliver_file(
            update, context, binding, media, filename, message.caption or ""
        )

    async def _tmux_voice(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        binding: Dict[str, str],
    ) -> None:
        features = context.bot_data.get("features")
        voice_handler = features.get_voice_handler() if features else None
        if not voice_handler:
            await update.message.reply_text(self._voice_unavailable_message())
            return
        progress = await update.message.reply_text("Transcribing…")
        try:
            processed = await voice_handler.process_voice_message(
                update.message.voice, update.message.caption
            )
        except Exception as e:
            from .handlers.message import _format_error_message

            await progress.edit_text(_format_error_message(e), parse_mode="HTML")
            return
        text = getattr(processed, "transcription", None) or getattr(
            processed, "prompt", ""
        )
        text = (text or "").strip()
        try:
            await progress.delete()
        except Exception:
            pass
        if not text:
            await update.message.reply_text("⚠️ Empty transcription.")
            return
        await update.message.reply_text(f"🎤 {escape_html(text)}", parse_mode="HTML")
        await self._tmux_send_and_report(update, context, text)

    # --- live jsonl mirror ---

    def _start_jsonl_mirror(
        self,
        bkey: BindingKey,
        bot: Any,
        target: str,
        proj_dir: Any,
        session_file: Optional[Any] = None,
    ) -> None:
        """(Re)start the background tailer streaming a session .jsonl to a
        chat/topic.

        ``proj_dir``/``session_file`` are path strings on the target's host
        (local paths for bare targets, remote paths for host/session ones).
        """
        self._stop_jsonl_mirror(bkey)
        self._jsonl_mirrors[bkey] = asyncio.create_task(
            self._jsonl_mirror_loop(
                bkey,
                bot,
                target,
                str(proj_dir),
                str(session_file) if session_file else None,
            )
        )

    def _stop_jsonl_mirror(self, bkey: BindingKey) -> None:
        task = self._jsonl_mirrors.pop(bkey, None)
        if task and not task.done():
            task.cancel()
        self._mirror_echo.pop(bkey, None)
        self._last_prompt.pop(bkey, None)

    def _note_echo(self, bkey: BindingKey, text: str) -> None:
        """Record a phone-sent prompt so its mirror echo can be suppressed."""
        lst = self._mirror_echo.setdefault(bkey, [])
        lst.append(text.strip())
        del lst[:-10]  # keep only the last few

    def _consume_echo(self, bkey: BindingKey, text: str) -> bool:
        """True if ``text`` matches a pending phone-sent prompt (and removes it)."""
        lst = self._mirror_echo.get(bkey)
        if not lst:
            return False
        t = text.strip()
        if t in lst:
            lst.remove(t)
            return True
        return False

    @staticmethod
    def _split_message(text: str, limit: int = 4000) -> List[str]:
        if len(text) <= limit:
            return [text]
        out: List[str] = []
        while text:
            out.append(text[:limit])
            text = text[limit:]
        return out

    async def _mirror_send(
        self, bot: Any, bkey: BindingKey, text: str, mode: Optional[str]
    ) -> None:
        """Deliver one rendered event, falling back to plain text on HTML error."""
        chat_id, thread_id = bkey
        for chunk in self._split_message(text):
            try:
                await bot.send_message(
                    chat_id=chat_id,
                    message_thread_id=thread_id,
                    text=chunk,
                    parse_mode=mode,
                )
            except Exception:
                try:
                    await bot.send_message(
                        chat_id=chat_id, message_thread_id=thread_id, text=chunk
                    )
                except Exception as e:
                    logger.warning("mirror send failed", chat_id=chat_id, error=str(e))
            await asyncio.sleep(0.4)  # gentle per-chat rate limit

    async def _check_pane_prompt(
        self, bot: Any, bkey: BindingKey, target: str, last_prompt: Optional[str]
    ) -> Optional[str]:
        """Push an interactive TUI prompt to the chat if one is waiting.

        Returns the signature of the shown prompt (to dedupe), or None when no
        prompt is on screen (so the next one re-pushes).
        """
        try:
            screen = await self._tmux(target).capture()
        except Exception:
            return last_prompt
        widget = is_question_widget(screen)
        if not widget and not is_interactive_prompt(screen):
            return None
        sig = prompt_signature(screen)
        if sig == last_prompt:
            return last_prompt  # already shown this exact prompt

        # A Usage/Stats panel (/usage) is a transient info overlay, not a real
        # input prompt — Claude closes it with Esc, not a reply. Show the parsed
        # card, then auto-Esc so the modal doesn't stay open swallowing the
        # user's next message. Return sig so a lingering frame won't re-post.
        card = render_usage(screen)
        if card:
            await self._mirror_send(bot, bkey, card, "HTML")
            try:
                await self._tmux(target).send_key("Escape")
            except Exception as e:
                logger.warning("usage auto-dismiss failed", error=str(e))
            return sig

        if widget:
            # Multi-question widget: digits/Enter shortcuts would mis-answer
            # tabs, so hand the user a key-remote that live-edits its snapshot.
            try:
                await bot.send_message(
                    chat_id=bkey[0],
                    message_thread_id=bkey[1],
                    text=self._widget_text(screen),
                    reply_markup=self._widget_keyboard(),
                    parse_mode="HTML",
                )
            except Exception as e:
                logger.warning("widget remote send failed", error=str(e))
            return sig

        menu = parse_menu(screen)
        if menu:
            rows = [
                [
                    InlineKeyboardButton(
                        f"{n}. {label}"[:60], callback_data=f"tmuxopt:{n}"
                    )
                ]
                for n, label, _desc in menu["options"]
            ]
            rows.append(
                [InlineKeyboardButton("✕ Cancel (Esc)", callback_data="tmuxopt:esc")]
            )
            body_lines = []
            for n, label, desc in menu["options"]:
                line = f"<b>{n}.</b> {escape_html(label)}"
                if desc:
                    short = desc if len(desc) <= 160 else desc[:160] + "…"
                    line += f"\n    <i>{escape_html(short)}</i>"
                body_lines.append(line)
            body = "\n".join(body_lines)
            try:
                await bot.send_message(
                    chat_id=bkey[0],
                    message_thread_id=bkey[1],
                    text=f"⌨️ <b>{escape_html(menu['title'])}</b>\n\n{body}",
                    reply_markup=InlineKeyboardMarkup(rows),
                    parse_mode="HTML",
                )
            except Exception as e:
                logger.warning("menu send failed", chat_id=bkey[0], error=str(e))
        else:
            # Free-form prompt (no clean numbered options) -> snapshot + hint.
            await self._mirror_send(
                bot,
                bkey,
                "⌨️ <b>Claude needs your input:</b>\n"
                + self._format_pane(screen)
                + "\nReply, or /key Up · /key Down · /key Enter.",
                "HTML",
            )
        return sig

    async def _handle_tmux_option_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle tmuxopt:<n> / tmuxopt:esc menu-button taps."""
        query = update.callback_query
        await query.answer()
        data = query.data or ""
        choice = data.split(":", 1)[1] if ":" in data else ""

        bkey = self._bkey(update)
        b = self._binding(bkey)
        if not b:
            await query.edit_message_text("⚠️ Not bound to a session anymore.")
            return
        bridge = self._tmux(b["target"])
        ok, detail = await bridge.available()
        if not ok:
            self._clear_binding(bkey)
            self._stop_jsonl_mirror(bkey)
            await query.edit_message_text(
                f"⚠️ {escape_html(detail)}", parse_mode="HTML"
            )
            return

        if choice == "esc":
            await bridge.send_key("Escape")
            await query.edit_message_text("✕ Cancelled (Esc).")
            return

        # Press the number; some menus submit on the digit alone, others need
        # Enter, so confirm with Enter only if a menu is still on screen.
        await bridge.send_key(choice)
        await asyncio.sleep(0.4)
        try:
            if is_interactive_prompt(await bridge.capture()):
                await bridge.send_key("Enter")
        except Exception:
            pass
        await query.edit_message_text(f"✅ Selected option {escape_html(choice)}.")

    # Keys the widget remote may press, mapped to tmux send-keys names.
    _REMOTE_KEYS: Dict[str, str] = {
        "up": "Up",
        "down": "Down",
        "left": "Left",
        "right": "Right",
        "space": "Space",
        "enter": "Enter",
        "esc": "Escape",
        "tab": "Tab",
        **{d: d for d in "123456789"},
    }

    def _widget_text(self, screen: str) -> str:
        """Snapshot body for the widget remote (footer region only)."""
        tail = "\n".join(screen.rstrip().splitlines()[-28:])
        return (
            "🎛 <b>Claude is asking (widget)</b> — drive it with the "
            f"buttons:\n{self._format_pane(tail)}"
        )

    def _widget_keyboard(self) -> InlineKeyboardMarkup:
        def btn(label: str, key: str) -> InlineKeyboardButton:
            return InlineKeyboardButton(label, callback_data=f"tmuxkey:{key}")

        return InlineKeyboardMarkup(
            [
                [
                    btn("⬅️", "left"),
                    btn("⬆️", "up"),
                    btn("⬇️", "down"),
                    btn("➡️", "right"),
                ],
                [btn(d, d) for d in "12345"],
                [
                    btn("␣ space", "space"),
                    btn("⏎ enter", "enter"),
                    btn("⎋ esc", "esc"),
                ],
                [btn("🔄 refresh", "refresh")],
            ]
        )

    async def _handle_tmux_key_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """tmuxkey:<key> — press a key in the bound pane and live-edit the
        snapshot message, turning the phone into a TUI remote."""
        query = update.callback_query
        await query.answer()
        key = (query.data or "").split(":", 1)[-1]

        bkey = self._bkey(update)
        b = self._binding(bkey)
        if not b:
            await query.edit_message_text("⚠️ Not bound to a session anymore.")
            return
        bridge = self._tmux(b["target"])
        if key != "refresh":
            tmux_key = self._REMOTE_KEYS.get(key)
            if not tmux_key:
                return
            await bridge.send_key(tmux_key)
            await asyncio.sleep(0.6)
        try:
            screen = await bridge.capture()
        except Exception:
            return
        # Keep the watcher quiet about the state this press produced.
        self._last_prompt[bkey] = prompt_signature(screen)
        still_open = is_question_widget(screen) or is_interactive_prompt(screen)
        try:
            if still_open:
                await query.edit_message_text(
                    self._widget_text(screen),
                    reply_markup=self._widget_keyboard(),
                    parse_mode="HTML",
                )
            else:
                await query.edit_message_text(
                    "✅ Widget closed — back to the session."
                )
                self._last_prompt[bkey] = None
        except Exception:
            pass  # unchanged screen -> Telegram rejects identical edit

    async def _jsonl_mirror_loop(
        self,
        bkey: BindingKey,
        bot: Any,
        target: str,
        proj_dir: str,
        session_file: Optional[str] = None,
    ) -> None:
        """Tail a Claude session .jsonl and push each new record to the chat.

        If no log exists yet (pane just started ``claude``), waits for one to
        appear and streams it from the start. An existing log is tailed from
        EOF (only new events from bind onward). Follows the newest file in the
        project dir so a /clear (new session file) is picked up, and resets on
        truncation. For host/session targets all file access runs over ssh
        (HostFS) against the remote's own session store.

        Also watches the pane screen for interactive prompts (menus, permission
        requests) -- those are drawn by the TUI and never hit the .jsonl, so
        they'd otherwise be invisible.
        """
        import json as _json

        fs = HostFS(split_host(target)[0])
        current = session_file
        offset = (await fs.size(current) or 0) if current else 0
        buf = ""
        poll = 0
        self._last_prompt[bkey] = None
        try:
            while True:
                await asyncio.sleep(1.0)
                poll += 1

                # Surface interactive TUI prompts (not present in the .jsonl).
                if poll % 2 == 0:
                    self._last_prompt[bkey] = await self._check_pane_prompt(
                        bot, bkey, target, self._last_prompt.get(bkey)
                    )

                # Waiting for the first log, or following a new one (/clear).
                if current is None:
                    current = await fs.newest_jsonl(proj_dir)
                    if current is None:
                        continue
                    offset, buf = 0, ""  # brand-new session: stream from start
                elif poll % 5 == 0:
                    newest = await fs.newest_jsonl(proj_dir)
                    if newest and newest != current:
                        current, offset, buf = newest, 0, ""
                size = await fs.size(current)
                if size is None:
                    continue
                if size < offset:  # truncated/rotated
                    offset, buf = 0, ""
                if size == offset:
                    continue
                chunk, nbytes = await fs.read_from(current, offset)
                if not nbytes:
                    continue
                offset += nbytes
                buf += chunk
                lines = buf.split("\n")
                buf = lines.pop()  # keep the incomplete trailing line
                # Coalesce runs of consecutive tool one-liners from this chunk
                # into single messages -- tool calls are the volume driver, and
                # in a forum group every topic shares one ~20 msg/min budget.
                tools: List[str] = []
                for ln in lines:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        obj = _json.loads(ln)
                    except Exception:
                        continue
                    # Suppress echoing prompts we sent from this chat.
                    raw = user_prompt_text(obj)
                    if raw is not None and self._consume_echo(bkey, raw):
                        continue
                    rendered = render_record(obj)
                    if not rendered:
                        continue
                    text, mode, kind = rendered
                    if kind == "tool":
                        if tools and (
                            sum(len(t) + 1 for t in tools) + len(text) > 3400
                        ):
                            await self._mirror_send(
                                bot, bkey, "\n".join(tools), "HTML"
                            )
                            tools = []
                        tools.append(text)
                        continue
                    if tools:
                        await self._mirror_send(bot, bkey, "\n".join(tools), "HTML")
                        tools = []
                    await self._mirror_send(bot, bkey, text, mode)
                if tools:
                    await self._mirror_send(bot, bkey, "\n".join(tools), "HTML")
        except asyncio.CancelledError:
            pass
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(
                "jsonl mirror loop crashed",
                chat_id=bkey[0],
                thread_id=bkey[1],
                error=str(e),
            )

    async def agentic_tmux(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Bind chat to a tmux session: /tmux [name|off]. No arg = tab picker."""
        args = update.message.text.split()[1:] if update.message.text else []
        bkey = self._bkey(update)
        b = self._binding(bkey)
        current = b["target"] if b else None

        if args and args[0].lower() == "off":
            self._clear_binding(bkey)
            self._stop_jsonl_mirror(bkey)
            await self._audit_tmux(update, context, "off")
            await update.message.reply_text("🔌 Unbound this chat/topic.")
            return

        if args:
            target = args[0]
            ok, detail = await self._tmux(target).available()
            if not ok:
                await update.message.reply_text(
                    f"⚠️ {escape_html(detail)}", parse_mode="HTML"
                )
                return
            await self._bind_and_snapshot(update, context, target)
            return

        # No args -> show the tab picker.
        await self._audit_tmux(update, context, "list")
        kb, unreachable = await self._tmux_session_keyboard(current)
        warn = (
            "\n⚠️ unreachable: "
            + ", ".join(f"<code>{escape_html(h)}</code>" for h in unreachable)
            if unreachable
            else ""
        )
        if kb is None:
            await update.message.reply_text(
                "No live tmux sessions. On the PC, start one, e.g.:\n"
                f"<code>tmux new -s claude claude</code>{warn}",
                parse_mode="HTML",
            )
            return
        bound = (
            f"<code>{escape_html(current)}</code>" if current else "none"
        )
        await update.message.reply_text(
            f"🖥 <b>tmux sessions</b> — bound: {bound}{warn}\n"
            "Tap a tab to switch:",
            reply_markup=kb,
            parse_mode="HTML",
        )

    async def _handle_tmux_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle tmuxsel:<name> / tmuxoff tab-switch callbacks."""
        query = update.callback_query
        await query.answer()
        data = query.data or ""

        if data == "tmuxoff":
            bkey = self._bkey(update)
            self._clear_binding(bkey)
            self._stop_jsonl_mirror(bkey)
            await self._audit_tmux(update, context, "off")
            await query.edit_message_text("🔌 Unbound this chat/topic.")
            return

        target = data.split(":", 1)[1] if ":" in data else ""
        if not target:
            return
        ok, detail = await self._tmux(target).available()
        if not ok:
            await query.edit_message_text(
                f"⚠️ {escape_html(detail)}", parse_mode="HTML"
            )
            return
        await self._bind_and_snapshot(update, context, target, edit=True)

    async def agentic_sync_tmux(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """One forum topic per live tmux session (all hosts): /sync_tmux.

        Run inside a supergroup with Topics enabled (bot needs Manage Topics).
        Creates+binds a topic for every session that doesn't have one here,
        and closes topics whose session died. Idempotent -- run it any time.
        """
        chat = update.effective_chat
        await self._audit_tmux(update, context, "sync_tmux")
        if not getattr(chat, "is_forum", False):
            await update.message.reply_text(
                "This needs a supergroup with <b>Topics</b> enabled (and me as "
                "admin with <i>Manage Topics</i>). Run /sync_tmux there.",
                parse_mode="HTML",
            )
            return

        sessions, unreachable = await tmux_list_all_sessions()
        names = {str(s["name"]) for s in sessions}
        # Existing topic bindings in THIS chat: target -> bkey.
        bound = {
            v["target"]: k
            for k, v in self._bindings.items()
            if k[0] == chat.id and k[1] is not None
        }
        pace = float(
            getattr(
                self.settings,
                "project_threads_sync_action_interval_seconds",
                1.1,
            )
            or 0
        )
        created, closed, failed = [], [], []

        for name in sorted(names - set(bound)):
            try:
                topic = await context.bot.create_forum_topic(
                    chat_id=chat.id, name=name[:120]
                )
            except Exception as e:
                logger.warning("topic create failed", name=name, error=str(e))
                failed.append(name)
                continue
            bkey = (chat.id, topic.message_thread_id)
            mode_line = await self._bind_session(context.bot, bkey, name)
            try:
                await context.bot.send_message(
                    chat_id=chat.id,
                    message_thread_id=topic.message_thread_id,
                    text=(
                        f"🔗 <code>{escape_html(name)}</code> — {mode_line}\n"
                        "Type here to drive it · /peek · /key · /stop."
                    ),
                    parse_mode="HTML",
                )
            except Exception:
                pass
            created.append(name)
            if pace:
                await asyncio.sleep(pace)

        for name, bkey in sorted(bound.items()):
            if name in names:
                continue
            self._stop_jsonl_mirror(bkey)
            self._clear_binding(bkey)
            try:
                await context.bot.send_message(
                    chat_id=bkey[0],
                    message_thread_id=bkey[1],
                    text="💀 Session is gone — closing this topic.",
                )
                await context.bot.close_forum_topic(
                    chat_id=bkey[0], message_thread_id=bkey[1]
                )
            except Exception as e:
                logger.warning("topic close failed", name=name, error=str(e))
            closed.append(name)
            if pace:
                await asyncio.sleep(pace)

        kept = len(bound) - len(closed)
        parts = [f"🗂 topics: +{len(created)} created, {kept} kept"]
        if closed:
            parts.append(f"{len(closed)} closed ({', '.join(closed[:5])})")
        if failed:
            parts.append(f"⚠️ failed: {', '.join(failed[:5])}")
        if unreachable:
            parts.append(f"⚠️ unreachable hosts: {', '.join(unreachable)}")
        await update.message.reply_text(" · ".join(parts))

    async def agentic_peek(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Snapshot the bound tmux session: /peek [scrollback_lines]."""
        bkey = self._bkey(update)
        b = self._binding(bkey)
        if not b:
            await update.message.reply_text(
                "Not bound to a session. Use /tmux to pick one."
            )
            return
        target = b["target"]

        args = update.message.text.split()[1:] if update.message.text else []
        lines: Optional[int] = None
        if args:
            try:
                lines = max(0, min(500, int(args[0])))
            except ValueError:
                lines = None

        bridge = self._tmux(target)
        ok, detail = await bridge.available()
        if not ok:
            self._clear_binding(bkey)
            self._stop_jsonl_mirror(bkey)
            await update.message.reply_text(
                f"⚠️ {escape_html(detail)}", parse_mode="HTML"
            )
            return

        snapshot = await bridge.capture(lines=lines)
        await update.message.reply_text(
            f"🖥 <code>{escape_html(target)}</code>\n{self._format_pane(snapshot)}",
            parse_mode="HTML",
        )

    async def agentic_key(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Send a named key to the bound session: /key <name> (esc, y, c-c, up)."""
        bkey = self._bkey(update)
        b = self._binding(bkey)
        if not b:
            await update.message.reply_text(
                "Not bound to a session. Use /tmux to pick one."
            )
            return
        target = b["target"]

        args = update.message.text.split()[1:] if update.message.text else []
        if not args:
            await update.message.reply_text(
                "Usage: <code>/key &lt;name&gt;</code> — e.g. "
                "<code>/key Enter</code>, <code>/key Escape</code>, "
                "<code>/key C-c</code>, <code>/key Up</code>, <code>/key y</code>",
                parse_mode="HTML",
            )
            return

        bridge = self._tmux(target)
        ok, detail = await bridge.available()
        if not ok:
            self._clear_binding(bkey)
            self._stop_jsonl_mirror(bkey)
            await update.message.reply_text(
                f"⚠️ {escape_html(detail)}", parse_mode="HTML"
            )
            return

        key = args[0]
        await bridge.send_key(key)
        snapshot = await bridge.capture_settled(timeout=15.0)
        await update.message.reply_text(
            f"⌨️ <code>{escape_html(key)}</code> → "
            f"<code>{escape_html(target)}</code>\n\n"
            f"{self._format_pane(snapshot)}",
            parse_mode="HTML",
        )

    async def agentic_resume(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Show a button menu of resumable Claude sessions for the current dir."""
        current_dir = context.user_data.get(
            "current_directory", self.settings.approved_directory
        )
        if not isinstance(current_dir, Path):
            current_dir = Path(str(current_dir))

        sessions = _list_claude_sessions(current_dir)
        current_sid = context.user_data.get("claude_session_id")

        audit_logger = context.bot_data.get("audit_logger")
        if audit_logger:
            await audit_logger.log_command(
                user_id=update.effective_user.id,
                command="resume",
                args=[str(current_dir)],
                success=True,
            )

        if not sessions:
            await update.message.reply_text(
                "No resumable sessions in "
                f"<code>{escape_html(current_dir.name)}/</code>.\n"
                "Send a message to start one, or /repo to switch project.",
                parse_mode="HTML",
            )
            return

        lines: List[str] = [f"<b>Sessions in {escape_html(current_dir.name)}/</b>"]
        keyboard: List[list] = []  # type: ignore[type-arg]
        for s in sessions:
            sid = s["session_id"]
            short = sid[:8]
            label = s["label"] or "(no summary)"
            marker = " ◀" if sid == current_sid else ""
            meta = f"{_humanize_age(s['mtime'])} · {_humanize_size(s['size'])}"
            lines.append(
                f"\n▸ <code>{short}</code>{marker} · {meta}\n"
                f"   <i>{escape_html(label)}</i>"
            )
            btn = f"{short} · {_truncate_label(label, 28)}"
            keyboard.append(
                [InlineKeyboardButton(btn, callback_data=f"resume:{sid}")]
            )

        await update.message.reply_text(
            "\n".join(lines),
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup(keyboard),
        )

    async def _handle_resume_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle resume:<session_id> callbacks — arm a session for continuation."""
        query = update.callback_query
        await query.answer()

        session_id = query.data.split(":", 1)[1]
        user_id = query.from_user.id

        current_dir = context.user_data.get(
            "current_directory", self.settings.approved_directory
        )
        if not isinstance(current_dir, Path):
            current_dir = Path(str(current_dir))

        # Verify the session file still exists on disk.
        proj = _claude_project_dir(current_dir)
        if not (proj / f"{session_id}.jsonl").exists():
            await query.edit_message_text(
                f"Session <code>{escape_html(session_id[:8])}</code> "
                "no longer exists.",
                parse_mode="HTML",
            )
            return

        # Register the session in the bot's store so run_command resumes it:
        # get_or_create_session loads by id from storage, and this is required
        # for sessions created outside the bot (e.g. the interactive CLI).
        claude_integration = context.bot_data.get("claude_integration")
        if claude_integration is not None:
            from ..claude.session import ClaudeSession

            mgr = claude_integration.session_manager
            now = datetime.now(UTC)
            session = ClaudeSession(
                session_id=session_id,
                user_id=user_id,
                project_path=current_dir,
                created_at=now,
                last_used=now,
                is_new_session=False,
            )
            await mgr.storage.save_session(session)
            mgr.active_sessions[session_id] = session

        context.user_data["claude_session_id"] = session_id
        context.user_data["force_new_session"] = False

        await query.edit_message_text(
            f"✅ Resuming <code>{escape_html(session_id[:8])}</code> in "
            f"<code>{escape_html(current_dir.name)}/</code>.\n"
            "Send your next message to continue this session.",
            parse_mode="HTML",
        )

        audit_logger = context.bot_data.get("audit_logger")
        if audit_logger:
            await audit_logger.log_command(
                user_id=user_id,
                command="resume_select",
                args=[session_id],
                success=True,
            )
