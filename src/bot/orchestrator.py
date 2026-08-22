"""Message orchestrator — single entry point for all Telegram updates.

Routes messages based on agentic vs classic mode. In agentic mode, provides
a minimal conversational interface (3 commands, no inline keyboards). In
classic mode, delegates to existing full-featured handlers.
"""

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import structlog
from telegram import (
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputMediaPhoto,
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

from ..claude.sdk_integration import StreamUpdate
from ..config.settings import Settings
from ..projects import PrivateTopicsUnavailableError
from .features.claude_jsonl import (
    claude_project_dir,
    is_interactive_prompt,
    prompt_signature,
    render_record,
    resolve_session_file,
    user_prompt_text,
)
from .features.tmux_bridge import TmuxBridge
from .features.tmux_bridge import list_sessions as tmux_list_sessions
from .utils.draft_streamer import DraftStreamer, generate_draft_id
from .utils.html_format import escape_html
from .utils.image_extractor import (
    ImageAttachment,
    should_send_as_photo,
    validate_image_path,
)

logger = structlog.get_logger()


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


# Claude emits these markers (one per file) only when the user asks to receive a
# file; the bot uploads each to Telegram and strips the marker from the reply.
_SEND_FILE_RE = re.compile(r"\[\[TG_SEND:\s*(.+?)\]\]")
_TG_VIDEO_EXT = {".mp4", ".mov", ".webm", ".m4v"}
_TG_PHOTO_EXT = {".png", ".jpg", ".jpeg", ".webp"}
_TG_MAX_UPLOAD = 50 * 1024 * 1024  # Telegram bot-API upload cap


def _extract_send_file_markers(text: str) -> tuple[str, List[str]]:
    """Pull [[TG_SEND: path]] markers out of text; return (clean_text, paths)."""
    paths = [m.strip() for m in _SEND_FILE_RE.findall(text)]
    clean = _SEND_FILE_RE.sub("", text)
    clean = re.sub(r"\n{3,}", "\n\n", clean).strip()
    return clean, paths


_MEDIA_TYPE_MAP = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
}

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


@dataclass
class ActiveRequest:
    """Tracks an in-flight Claude request so it can be interrupted."""

    user_id: int
    interrupt_event: asyncio.Event = field(default_factory=asyncio.Event)
    interrupted: bool = False
    progress_msg: Any = None  # telegram Message object


class MessageOrchestrator:
    """Routes messages based on mode. Single entry point for all Telegram updates."""

    def __init__(self, settings: Settings, deps: Dict[str, Any]):
        self.settings = settings
        self.deps = deps
        self._active_requests: Dict[int, ActiveRequest] = {}
        self._known_commands: frozenset[str] = frozenset()
        # chat_id -> background task tailing a Claude session .jsonl into the chat
        self._jsonl_mirrors: Dict[int, "asyncio.Task[None]"] = {}
        # chat_id -> recently phone-sent prompts, to suppress their mirror echo
        self._mirror_echo: Dict[int, List[str]] = {}
        # chat_id -> {"target": str, "mode": "mirror"|"snapshot"}, persisted so
        # tmux bindings + live mirrors survive a bot restart.
        self._bindings: Dict[int, Dict[str, str]] = self._load_bindings()

    # --- persistent tmux bindings ---

    def _bindings_path(self) -> Path:
        """JSON file for tmux bindings, kept beside the SQLite DB."""
        url = self.settings.database_url
        db = url[len("sqlite:///") :] if url.startswith("sqlite:///") else "data/bot.db"
        parent = Path(db).parent if db else Path("data")
        return parent / "tmux_bindings.json"

    def _load_bindings(self) -> Dict[int, Dict[str, str]]:
        try:
            raw = json.loads(self._bindings_path().read_text(encoding="utf-8"))
            return {int(k): v for k, v in raw.items()}
        except (OSError, ValueError, TypeError):
            return {}

    def _save_bindings(self) -> None:
        try:
            path = self._bindings_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({str(k): v for k, v in self._bindings.items()}),
                encoding="utf-8",
            )
        except OSError as e:
            logger.warning("Failed to persist tmux bindings", error=str(e))

    def _binding(self, chat_id: int) -> Optional[Dict[str, str]]:
        return self._bindings.get(chat_id)

    def _set_binding(self, chat_id: int, target: str, mode: str) -> None:
        self._bindings[chat_id] = {"target": target, "mode": mode}
        self._save_bindings()

    def _clear_binding(self, chat_id: int) -> None:
        if self._bindings.pop(chat_id, None) is not None:
            self._save_bindings()

    async def restore_mirrors(self, bot: Any) -> None:
        """Re-establish live mirrors for persisted bindings after a restart."""
        for chat_id, b in list(self._bindings.items()):
            if b.get("mode") != "mirror":
                continue
            try:
                bridge = self._tmux(b["target"])
                ok, _ = await bridge.available()
                cwd = await bridge.pane_cwd() if ok else None
            except Exception:
                cwd = None
            if cwd:
                self._start_jsonl_mirror(
                    chat_id,
                    bot,
                    b["target"],
                    claude_project_dir(Path(cwd)),
                    resolve_session_file(Path(cwd)),
                )
                logger.info(
                    "Restored tmux mirror", chat_id=chat_id, target=b["target"]
                )
                try:
                    await bot.send_message(
                        chat_id=chat_id,
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
                self._clear_binding(chat_id)

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
            ("verbose", self.agentic_verbose),
            ("repo", self.agentic_repo),
            ("resume", self.agentic_resume),
            ("tmux", self.agentic_tmux),
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

        # Voice messages -> transcribe -> Claude
        app.add_handler(
            MessageHandler(filters.VOICE, self._inject_deps(self.agentic_voice)),
            group=10,
        )

        # Stop button callback (must be before cd: handler)
        app.add_handler(
            CallbackQueryHandler(
                self._inject_deps(self._handle_stop_callback),
                pattern=r"^stop:",
            )
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
                BotCommand("verbose", "Set output verbosity (0/1/2)"),
                BotCommand("repo", "List repos / switch workspace"),
                BotCommand("resume", "Pick a session to resume"),
                BotCommand("tmux", "Pick/switch a live tmux session (tabs)"),
                BotCommand("peek", "Snapshot the bound tmux pane"),
                BotCommand("key", "Send a key to the bound pane (esc/y/c-c…)"),
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

    def _get_verbose_level(self, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Return effective verbose level: per-user override or global default."""
        user_override = context.user_data.get("verbose_level")
        if user_override is not None:
            return int(user_override)
        return self.settings.verbose_level

    async def agentic_verbose(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Set output verbosity: /verbose [0|1|2]."""
        args = update.message.text.split()[1:] if update.message.text else []
        if not args:
            current = self._get_verbose_level(context)
            labels = {0: "quiet", 1: "normal", 2: "detailed"}
            await update.message.reply_text(
                f"Verbosity: <b>{current}</b> ({labels.get(current, '?')})\n\n"
                "Usage: <code>/verbose 0|1|2</code>\n"
                "  0 = quiet (final response only)\n"
                "  1 = normal (tools + reasoning)\n"
                "  2 = detailed (tools with inputs + reasoning)",
                parse_mode="HTML",
            )
            return

        try:
            level = int(args[0])
            if level not in (0, 1, 2):
                raise ValueError
        except ValueError:
            await update.message.reply_text(
                "Please use: /verbose 0, /verbose 1, or /verbose 2"
            )
            return

        context.user_data["verbose_level"] = level
        labels = {0: "quiet", 1: "normal", 2: "detailed"}
        await update.message.reply_text(
            f"Verbosity set to <b>{level}</b> ({labels[level]})",
            parse_mode="HTML",
        )

    def _format_verbose_progress(
        self,
        activity_log: List[Dict[str, Any]],
        verbose_level: int,
        start_time: float,
    ) -> str:
        """Build the progress message text based on activity so far."""
        if not activity_log:
            return "Working..."

        elapsed = time.time() - start_time
        lines: List[str] = [f"Working... ({elapsed:.0f}s)\n"]

        for entry in activity_log[-15:]:  # Show last 15 entries max
            kind = entry.get("kind", "tool")
            if kind == "text":
                # Claude's intermediate reasoning/commentary
                snippet = entry.get("detail", "")
                if verbose_level >= 2:
                    lines.append(f"\U0001f4ac {snippet}")
                else:
                    # Level 1: one short line
                    lines.append(f"\U0001f4ac {snippet[:80]}")
            else:
                # Tool call
                icon = _tool_icon(entry["name"])
                if verbose_level >= 2 and entry.get("detail"):
                    lines.append(f"{icon} {entry['name']}: {entry['detail']}")
                else:
                    lines.append(f"{icon} {entry['name']}")

        if len(activity_log) > 15:
            lines.insert(1, f"... ({len(activity_log) - 15} earlier entries)\n")

        return "\n".join(lines)

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

    @staticmethod
    def _start_typing_heartbeat(
        chat: Any,
        interval: float = 2.0,
    ) -> "asyncio.Task[None]":
        """Start a background typing indicator task.

        Sends typing every *interval* seconds, independently of
        stream events. Cancel the returned task in a ``finally``
        block.
        """

        async def _heartbeat() -> None:
            try:
                while True:
                    await asyncio.sleep(interval)
                    try:
                        await chat.send_action("typing")
                    except Exception:
                        pass
            except asyncio.CancelledError:
                pass

        return asyncio.create_task(_heartbeat())

    def _make_stream_callback(
        self,
        verbose_level: int,
        progress_msg: Any,
        tool_log: List[Dict[str, Any]],
        start_time: float,
        reply_markup: Optional[InlineKeyboardMarkup] = None,
        mcp_images: Optional[List[ImageAttachment]] = None,
        approved_directory: Optional[Path] = None,
        draft_streamer: Optional[DraftStreamer] = None,
        interrupt_event: Optional[asyncio.Event] = None,
    ) -> Optional[Callable[[StreamUpdate], Any]]:
        """Create a stream callback for verbose progress updates.

        When *mcp_images* is provided, the callback also intercepts
        ``send_image_to_user`` tool calls and collects validated
        :class:`ImageAttachment` objects for later Telegram delivery.

        When *draft_streamer* is provided, tool activity and assistant
        text are streamed to the user in real time via
        ``sendMessageDraft``.

        Returns None when verbose_level is 0 **and** no MCP image
        collection or draft streaming is requested.
        Typing indicators are handled by a separate heartbeat task.
        """
        need_mcp_intercept = mcp_images is not None and approved_directory is not None

        if verbose_level == 0 and not need_mcp_intercept and draft_streamer is None:
            return None

        last_edit_time = [0.0]  # mutable container for closure

        async def _on_stream(update_obj: StreamUpdate) -> None:
            # Stop all streaming activity after interrupt
            if interrupt_event is not None and interrupt_event.is_set():
                return

            # Intercept send_image_to_user MCP tool calls.
            # The SDK namespaces MCP tools as "mcp__<server>__<tool>",
            # so match both the bare name and the namespaced variant.
            if update_obj.tool_calls and need_mcp_intercept:
                for tc in update_obj.tool_calls:
                    tc_name = tc.get("name", "")
                    if tc_name == "send_image_to_user" or tc_name.endswith(
                        "__send_image_to_user"
                    ):
                        tc_input = tc.get("input", {})
                        file_path = tc_input.get("file_path", "")
                        caption = tc_input.get("caption", "")
                        img = validate_image_path(
                            file_path, approved_directory, caption
                        )
                        if img:
                            mcp_images.append(img)

            # Capture tool calls
            if update_obj.tool_calls:
                for tc in update_obj.tool_calls:
                    name = tc.get("name", "unknown")
                    detail = self._summarize_tool_input(name, tc.get("input", {}))
                    if verbose_level >= 1:
                        tool_log.append(
                            {"kind": "tool", "name": name, "detail": detail}
                        )
                    if draft_streamer:
                        icon = _tool_icon(name)
                        line = (
                            f"{icon} {name}: {detail}" if detail else f"{icon} {name}"
                        )
                        await draft_streamer.append_tool(line)

            # Capture assistant text (reasoning / commentary)
            if update_obj.type == "assistant" and update_obj.content:
                text = update_obj.content.strip()
                if text:
                    first_line = text.split("\n", 1)[0].strip()
                    if first_line:
                        if verbose_level >= 1:
                            tool_log.append(
                                {"kind": "text", "detail": first_line[:120]}
                            )
                        if draft_streamer:
                            await draft_streamer.append_tool(
                                f"\U0001f4ac {first_line[:120]}"
                            )

            # Stream text to user via draft (prefer token deltas;
            # skip full assistant messages to avoid double-appending)
            if draft_streamer and update_obj.content:
                if update_obj.type == "stream_delta":
                    await draft_streamer.append_text(update_obj.content)

            # Throttle progress message edits to avoid Telegram rate limits
            if not draft_streamer and verbose_level >= 1:
                now = time.time()
                if (now - last_edit_time[0]) >= 2.0 and tool_log:
                    last_edit_time[0] = now
                    new_text = self._format_verbose_progress(
                        tool_log, verbose_level, start_time
                    )
                    try:
                        await progress_msg.edit_text(
                            new_text, reply_markup=reply_markup
                        )
                    except Exception:
                        pass

        return _on_stream

    async def _send_user_files(
        self,
        update: Update,
        paths: List[str],
        current_dir: Path,
        approved_directory: Path,
    ) -> None:
        """Upload user-requested files to Telegram (video/photo/document).

        Paths are resolved relative to *current_dir* when not absolute, and must
        stay within *approved_directory*. Videos go via reply_video, images via
        reply_photo, everything else as a document. Files over the 50MB bot-API
        cap are skipped with a note.
        """
        reply_to = update.message.message_id
        for raw in paths:
            try:
                path = Path(raw).expanduser()
                if not path.is_absolute():
                    path = current_dir / path
                path = path.resolve()
            except Exception:
                continue

            if not self._is_within(path, approved_directory) or not path.is_file():
                await update.message.reply_text(
                    f"⚠️ Can't send <code>{escape_html(raw)}</code> "
                    "(outside workspace or not found).",
                    parse_mode="HTML",
                )
                continue

            size = path.stat().st_size
            if size > _TG_MAX_UPLOAD:
                await update.message.reply_text(
                    f"⚠️ <code>{escape_html(path.name)}</code> is "
                    f"{size / 1048576:.0f}MB — over Telegram's 50MB bot limit.",
                    parse_mode="HTML",
                )
                continue

            suffix = path.suffix.lower()
            try:
                with open(path, "rb") as fh:
                    if suffix in _TG_VIDEO_EXT:
                        await update.message.reply_video(
                            video=fh,
                            filename=path.name,
                            supports_streaming=True,
                            reply_to_message_id=reply_to,
                        )
                    elif suffix in _TG_PHOTO_EXT:
                        await update.message.reply_photo(
                            photo=fh, reply_to_message_id=reply_to
                        )
                    else:
                        await update.message.reply_document(
                            document=fh,
                            filename=path.name,
                            reply_to_message_id=reply_to,
                        )
                logger.info("Sent file to user", path=str(path), bytes=size)
            except Exception as e:
                logger.warning("File send failed", path=str(path), error=str(e))
                await update.message.reply_text(
                    f"⚠️ Failed to upload <code>{escape_html(path.name)}</code>: "
                    f"{escape_html(str(e)[:150])}",
                    parse_mode="HTML",
                )

    async def _send_images(
        self,
        update: Update,
        images: List[ImageAttachment],
        reply_to_message_id: Optional[int] = None,
        caption: Optional[str] = None,
        caption_parse_mode: Optional[str] = None,
    ) -> bool:
        """Send extracted images as a media group (album) or documents.

        If *caption* is provided and fits (≤1024 chars), it is attached to the
        photo / first album item so text + images appear as one message.

        Returns True if the caption was successfully embedded in the photo message.
        """
        photos: List[ImageAttachment] = []
        documents: List[ImageAttachment] = []
        for img in images:
            if should_send_as_photo(img.path):
                photos.append(img)
            else:
                documents.append(img)

        # Telegram caption limit
        use_caption = bool(
            caption and len(caption) <= 1024 and photos and not documents
        )
        caption_sent = False

        # Send raster photos as a single album (Telegram groups 2-10 items)
        if photos:
            try:
                if len(photos) == 1:
                    with open(photos[0].path, "rb") as f:
                        await update.message.reply_photo(
                            photo=f,
                            reply_to_message_id=reply_to_message_id,
                            caption=caption if use_caption else None,
                            parse_mode=caption_parse_mode if use_caption else None,
                        )
                    caption_sent = use_caption
                else:
                    media = []
                    file_handles = []
                    for idx, img in enumerate(photos[:10]):
                        fh = open(img.path, "rb")  # noqa: SIM115
                        file_handles.append(fh)
                        media.append(
                            InputMediaPhoto(
                                media=fh,
                                caption=caption if use_caption and idx == 0 else None,
                                parse_mode=(
                                    caption_parse_mode
                                    if use_caption and idx == 0
                                    else None
                                ),
                            )
                        )
                    try:
                        await update.message.chat.send_media_group(
                            media=media,
                            reply_to_message_id=reply_to_message_id,
                        )
                        caption_sent = use_caption
                    finally:
                        for fh in file_handles:
                            fh.close()
            except Exception as e:
                logger.warning("Failed to send photo album", error=str(e))

        # Send SVGs / large files as documents (one by one — can't mix in album)
        for img in documents:
            try:
                with open(img.path, "rb") as f:
                    await update.message.reply_document(
                        document=f,
                        filename=img.path.name,
                        reply_to_message_id=reply_to_message_id,
                    )
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.warning(
                    "Failed to send document image",
                    path=str(img.path),
                    error=str(e),
                )

        return caption_sent

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

        # tmux bridge mode: type into a live tmux session instead of spawning
        # a headless `claude -p`. Same process, same session. The chat is
        # bound to one session at a time; /tmux switches "tabs".
        if self._binding(update.effective_chat.id):
            await self._tmux_send_and_report(update, context, message_text)
            return

        chat = update.message.chat
        await chat.send_action("typing")

        verbose_level = self._get_verbose_level(context)

        # Create Stop button and interrupt event
        interrupt_event = asyncio.Event()
        stop_kb = InlineKeyboardMarkup(
            [[InlineKeyboardButton("Stop", callback_data=f"stop:{user_id}")]]
        )
        progress_msg = await update.message.reply_text(
            "Working...", reply_markup=stop_kb
        )

        # Register active request for stop callback
        active_request = ActiveRequest(
            user_id=user_id,
            interrupt_event=interrupt_event,
            progress_msg=progress_msg,
        )
        self._active_requests[user_id] = active_request

        claude_integration = context.bot_data.get("claude_integration")
        if not claude_integration:
            self._active_requests.pop(user_id, None)
            await progress_msg.edit_text(
                "Claude integration not available. Check configuration.",
                reply_markup=None,
            )
            return

        current_dir = context.user_data.get(
            "current_directory", self.settings.approved_directory
        )
        session_id = context.user_data.get("claude_session_id")

        # Check if /new was used — skip auto-resume for this first message.
        # Flag is only cleared after a successful run so retries keep the intent.
        force_new = bool(context.user_data.get("force_new_session"))

        # --- Verbose progress tracking via stream callback ---
        tool_log: List[Dict[str, Any]] = []
        start_time = time.time()
        mcp_images: List[ImageAttachment] = []
        send_file_paths: List[str] = []

        # Stream drafts (private chats only)
        draft_streamer: Optional[DraftStreamer] = None
        if self.settings.enable_stream_drafts and chat.type == "private":
            draft_streamer = DraftStreamer(
                bot=context.bot,
                chat_id=chat.id,
                draft_id=generate_draft_id(),
                message_thread_id=update.message.message_thread_id,
                throttle_interval=self.settings.stream_draft_interval,
            )

        on_stream = self._make_stream_callback(
            verbose_level,
            progress_msg,
            tool_log,
            start_time,
            reply_markup=stop_kb,
            mcp_images=mcp_images,
            approved_directory=self.settings.approved_directory,
            draft_streamer=draft_streamer,
            interrupt_event=interrupt_event,
        )

        # Independent typing heartbeat — stays alive even with no stream events
        heartbeat = self._start_typing_heartbeat(chat)

        success = True
        try:
            claude_response = await claude_integration.run_command(
                prompt=message_text,
                working_directory=current_dir,
                user_id=user_id,
                session_id=session_id,
                on_stream=on_stream,
                force_new=force_new,
                interrupt_event=interrupt_event,
            )

            # New session created successfully — clear the one-shot flag
            if force_new:
                context.user_data["force_new_session"] = False

            context.user_data["claude_session_id"] = claude_response.session_id

            # Track directory changes
            from .handlers.message import _update_working_directory_from_claude_response

            _update_working_directory_from_claude_response(
                claude_response, context, self.settings, user_id
            )

            # Store interaction
            storage = context.bot_data.get("storage")
            if storage:
                try:
                    await storage.save_claude_interaction(
                        user_id=user_id,
                        session_id=claude_response.session_id,
                        prompt=message_text,
                        response=claude_response,
                        ip_address=None,
                    )
                except Exception as e:
                    logger.warning("Failed to log interaction", error=str(e))

            # Format response (no reply_markup — strip keyboards)
            from .utils.formatting import ResponseFormatter

            formatter = ResponseFormatter(self.settings)

            response_content = claude_response.content
            if claude_response.interrupted:
                response_content = (
                    response_content or ""
                ) + "\n\n_(Interrupted by user)_"

            if response_content:
                response_content, send_file_paths = _extract_send_file_markers(
                    response_content
                )

            formatted_messages = formatter.format_claude_response(response_content)

        except Exception as e:
            success = False
            logger.error("Claude integration failed", error=str(e), user_id=user_id)
            from .handlers.message import _format_error_message
            from .utils.formatting import FormattedMessage

            formatted_messages = [
                FormattedMessage(_format_error_message(e), parse_mode="HTML")
            ]
        finally:
            heartbeat.cancel()
            self._active_requests.pop(user_id, None)
            if draft_streamer:
                try:
                    await draft_streamer.flush()
                except Exception:
                    logger.debug("Draft flush failed in finally block", user_id=user_id)

        try:
            await progress_msg.delete()
        except Exception:
            logger.debug("Failed to delete progress message, ignoring")

        # Use MCP-collected images (from send_image_to_user tool calls)
        images: List[ImageAttachment] = mcp_images

        # Try to combine text + images in one message when possible
        caption_sent = False
        if images and len(formatted_messages) == 1:
            msg = formatted_messages[0]
            if msg.text and len(msg.text) <= 1024:
                try:
                    caption_sent = await self._send_images(
                        update,
                        images,
                        reply_to_message_id=update.message.message_id,
                        caption=msg.text,
                        caption_parse_mode=msg.parse_mode,
                    )
                except Exception as img_err:
                    logger.warning("Image+caption send failed", error=str(img_err))

        # Send text messages (skip if caption was already embedded in photos)
        if not caption_sent:
            for i, message in enumerate(formatted_messages):
                if not message.text or not message.text.strip():
                    continue
                try:
                    await update.message.reply_text(
                        message.text,
                        parse_mode=message.parse_mode,
                        reply_markup=None,  # No keyboards in agentic mode
                        reply_to_message_id=(
                            update.message.message_id if i == 0 else None
                        ),
                    )
                    if i < len(formatted_messages) - 1:
                        await asyncio.sleep(0.5)
                except Exception as send_err:
                    logger.warning(
                        "Failed to send HTML response, retrying as plain text",
                        error=str(send_err),
                        message_index=i,
                    )
                    try:
                        await update.message.reply_text(
                            message.text,
                            reply_markup=None,
                            reply_to_message_id=(
                                update.message.message_id if i == 0 else None
                            ),
                        )
                    except Exception as plain_err:
                        await update.message.reply_text(
                            f"Failed to deliver response "
                            f"(Telegram error: {str(plain_err)[:150]}). "
                            f"Please try again.",
                            reply_to_message_id=(
                                update.message.message_id if i == 0 else None
                            ),
                        )

            # Send images separately if caption wasn't used
            if images:
                try:
                    await self._send_images(
                        update,
                        images,
                        reply_to_message_id=update.message.message_id,
                    )
                except Exception as img_err:
                    logger.warning("Image send failed", error=str(img_err))

        # Deliver any files the user explicitly asked for (via [[TG_SEND: ...]]).
        if send_file_paths:
            try:
                await self._send_user_files(
                    update,
                    send_file_paths,
                    current_dir,
                    self.settings.approved_directory,
                )
            except Exception as file_err:
                logger.warning("User file send failed", error=str(file_err))

        # Never leave the turn silent: if the model produced no final text (it
        # ended on a tool action) and nothing else was delivered, still reply so
        # the user gets a completion signal instead of a deleted progress bubble.
        delivered = (
            caption_sent
            or any(m.text and m.text.strip() for m in formatted_messages)
            or bool(images)
            or bool(send_file_paths)
        )
        if not delivered:
            try:
                await update.message.reply_text(
                    "✅ Done.",
                    reply_to_message_id=update.message.message_id,
                )
            except Exception as done_err:
                logger.warning("Fallback done-message failed", error=str(done_err))

        # Audit log
        audit_logger = context.bot_data.get("audit_logger")
        if audit_logger:
            await audit_logger.log_command(
                user_id=user_id,
                command="text_message",
                args=[message_text[:100]],
                success=success,
            )

    async def agentic_document(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Process file upload -> Claude, minimal chrome."""
        user_id = update.effective_user.id
        document = update.message.document

        logger.info(
            "Agentic document upload",
            user_id=user_id,
            filename=document.file_name,
        )

        # Security validation
        security_validator = context.bot_data.get("security_validator")
        if security_validator:
            valid, error = security_validator.validate_filename(document.file_name)
            if not valid:
                await update.message.reply_text(f"File rejected: {error}")
                return

        # Size check
        max_size = 10 * 1024 * 1024
        if document.file_size > max_size:
            await update.message.reply_text(
                f"File too large ({document.file_size / 1024 / 1024:.1f}MB). Max: 10MB."
            )
            return

        chat = update.message.chat
        await chat.send_action("typing")
        progress_msg = await update.message.reply_text("Working...")

        # Try enhanced file handler, fall back to basic
        features = context.bot_data.get("features")
        file_handler = features.get_file_handler() if features else None
        prompt: Optional[str] = None

        if file_handler:
            try:
                processed_file = await file_handler.handle_document_upload(
                    document,
                    user_id,
                    update.message.caption or "Please review this file:",
                )
                prompt = processed_file.prompt
            except Exception:
                file_handler = None

        if not file_handler:
            file = await document.get_file()
            file_bytes = await file.download_as_bytearray()
            try:
                content = file_bytes.decode("utf-8")
                if len(content) > 50000:
                    content = content[:50000] + "\n... (truncated)"
                caption = update.message.caption or "Please review this file:"
                prompt = (
                    f"{caption}\n\n**File:** `{document.file_name}`\n\n"
                    f"```\n{content}\n```"
                )
            except UnicodeDecodeError:
                await progress_msg.edit_text(
                    "Unsupported file format. Must be text-based (UTF-8)."
                )
                return

        # Process with Claude
        claude_integration = context.bot_data.get("claude_integration")
        if not claude_integration:
            await progress_msg.edit_text(
                "Claude integration not available. Check configuration."
            )
            return

        current_dir = context.user_data.get(
            "current_directory", self.settings.approved_directory
        )
        session_id = context.user_data.get("claude_session_id")

        # Check if /new was used — skip auto-resume for this first message.
        # Flag is only cleared after a successful run so retries keep the intent.
        force_new = bool(context.user_data.get("force_new_session"))

        verbose_level = self._get_verbose_level(context)
        tool_log: List[Dict[str, Any]] = []
        mcp_images_doc: List[ImageAttachment] = []
        on_stream = self._make_stream_callback(
            verbose_level,
            progress_msg,
            tool_log,
            time.time(),
            mcp_images=mcp_images_doc,
            approved_directory=self.settings.approved_directory,
        )

        heartbeat = self._start_typing_heartbeat(chat)
        try:
            claude_response = await claude_integration.run_command(
                prompt=prompt,
                working_directory=current_dir,
                user_id=user_id,
                session_id=session_id,
                on_stream=on_stream,
                force_new=force_new,
            )

            if force_new:
                context.user_data["force_new_session"] = False

            context.user_data["claude_session_id"] = claude_response.session_id

            from .handlers.message import _update_working_directory_from_claude_response

            _update_working_directory_from_claude_response(
                claude_response, context, self.settings, user_id
            )

            from .utils.formatting import ResponseFormatter

            formatter = ResponseFormatter(self.settings)
            formatted_messages = formatter.format_claude_response(
                claude_response.content
            )

            try:
                await progress_msg.delete()
            except Exception:
                logger.debug("Failed to delete progress message, ignoring")

            # Use MCP-collected images (from send_image_to_user tool calls)
            images: List[ImageAttachment] = mcp_images_doc

            caption_sent = False
            if images and len(formatted_messages) == 1:
                msg = formatted_messages[0]
                if msg.text and len(msg.text) <= 1024:
                    try:
                        caption_sent = await self._send_images(
                            update,
                            images,
                            reply_to_message_id=update.message.message_id,
                            caption=msg.text,
                            caption_parse_mode=msg.parse_mode,
                        )
                    except Exception as img_err:
                        logger.warning("Image+caption send failed", error=str(img_err))

            if not caption_sent:
                for i, message in enumerate(formatted_messages):
                    await update.message.reply_text(
                        message.text,
                        parse_mode=message.parse_mode,
                        reply_markup=None,
                        reply_to_message_id=(
                            update.message.message_id if i == 0 else None
                        ),
                    )
                    if i < len(formatted_messages) - 1:
                        await asyncio.sleep(0.5)

                if images:
                    try:
                        await self._send_images(
                            update,
                            images,
                            reply_to_message_id=update.message.message_id,
                        )
                    except Exception as img_err:
                        logger.warning("Image send failed", error=str(img_err))

        except Exception as e:
            from .handlers.message import _format_error_message

            await progress_msg.edit_text(_format_error_message(e), parse_mode="HTML")
            logger.error("Claude file processing failed", error=str(e), user_id=user_id)
        finally:
            heartbeat.cancel()

    async def agentic_photo(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Process photo -> Claude, minimal chrome."""
        user_id = update.effective_user.id

        features = context.bot_data.get("features")
        image_handler = features.get_image_handler() if features else None

        if not image_handler:
            await update.message.reply_text("Photo processing is not available.")
            return

        chat = update.message.chat
        await chat.send_action("typing")
        progress_msg = await update.message.reply_text("Working...")

        try:
            photo = update.message.photo[-1]
            processed_image = await image_handler.process_image(
                photo, update.message.caption
            )
            fmt = processed_image.metadata.get("format", "png")
            images = [
                {
                    "data": processed_image.base64_data,
                    "media_type": _MEDIA_TYPE_MAP.get(fmt, "image/png"),
                }
            ]

            await self._handle_agentic_media_message(
                update=update,
                context=context,
                prompt=processed_image.prompt,
                progress_msg=progress_msg,
                user_id=user_id,
                chat=chat,
                images=images,
            )

        except Exception as e:
            from .handlers.message import _format_error_message

            await progress_msg.edit_text(_format_error_message(e), parse_mode="HTML")
            logger.error(
                "Claude photo processing failed", error=str(e), user_id=user_id
            )

    async def agentic_voice(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Transcribe voice message -> Claude, minimal chrome."""
        user_id = update.effective_user.id

        features = context.bot_data.get("features")
        voice_handler = features.get_voice_handler() if features else None

        if not voice_handler:
            await update.message.reply_text(self._voice_unavailable_message())
            return

        chat = update.message.chat
        await chat.send_action("typing")
        progress_msg = await update.message.reply_text("Transcribing...")

        try:
            voice = update.message.voice
            processed_voice = await voice_handler.process_voice_message(
                voice, update.message.caption
            )

            await progress_msg.edit_text("Working...")
            await self._handle_agentic_media_message(
                update=update,
                context=context,
                prompt=processed_voice.prompt,
                progress_msg=progress_msg,
                user_id=user_id,
                chat=chat,
            )

        except Exception as e:
            from .handlers.message import _format_error_message

            await progress_msg.edit_text(_format_error_message(e), parse_mode="HTML")
            logger.error(
                "Claude voice processing failed", error=str(e), user_id=user_id
            )

    async def _handle_agentic_media_message(
        self,
        *,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        prompt: str,
        progress_msg: Any,
        user_id: int,
        chat: Any,
        images: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        """Run a media-derived prompt through Claude and send responses."""
        claude_integration = context.bot_data.get("claude_integration")
        if not claude_integration:
            await progress_msg.edit_text(
                "Claude integration not available. Check configuration."
            )
            return

        current_dir = context.user_data.get(
            "current_directory", self.settings.approved_directory
        )
        session_id = context.user_data.get("claude_session_id")
        force_new = bool(context.user_data.get("force_new_session"))

        verbose_level = self._get_verbose_level(context)
        tool_log: List[Dict[str, Any]] = []
        mcp_images_media: List[ImageAttachment] = []
        on_stream = self._make_stream_callback(
            verbose_level,
            progress_msg,
            tool_log,
            time.time(),
            mcp_images=mcp_images_media,
            approved_directory=self.settings.approved_directory,
        )

        heartbeat = self._start_typing_heartbeat(chat)
        try:
            claude_response = await claude_integration.run_command(
                prompt=prompt,
                working_directory=current_dir,
                user_id=user_id,
                session_id=session_id,
                on_stream=on_stream,
                force_new=force_new,
                images=images,
            )
        finally:
            heartbeat.cancel()

        if force_new:
            context.user_data["force_new_session"] = False

        context.user_data["claude_session_id"] = claude_response.session_id

        from .handlers.message import _update_working_directory_from_claude_response

        _update_working_directory_from_claude_response(
            claude_response, context, self.settings, user_id
        )

        from .utils.formatting import ResponseFormatter

        formatter = ResponseFormatter(self.settings)
        formatted_messages = formatter.format_claude_response(claude_response.content)

        try:
            await progress_msg.delete()
        except Exception:
            logger.debug("Failed to delete progress message, ignoring")

        # Use MCP-collected images (from send_image_to_user tool calls).
        images: List[ImageAttachment] = mcp_images_media

        caption_sent = False
        if images and len(formatted_messages) == 1:
            msg = formatted_messages[0]
            if msg.text and len(msg.text) <= 1024:
                try:
                    caption_sent = await self._send_images(
                        update,
                        images,
                        reply_to_message_id=update.message.message_id,
                        caption=msg.text,
                        caption_parse_mode=msg.parse_mode,
                    )
                except Exception as img_err:
                    logger.warning("Image+caption send failed", error=str(img_err))

        if not caption_sent:
            for i, message in enumerate(formatted_messages):
                if not message.text or not message.text.strip():
                    continue
                await update.message.reply_text(
                    message.text,
                    parse_mode=message.parse_mode,
                    reply_markup=None,
                    reply_to_message_id=(update.message.message_id if i == 0 else None),
                )
                if i < len(formatted_messages) - 1:
                    await asyncio.sleep(0.5)

            if images:
                try:
                    await self._send_images(
                        update,
                        images,
                        reply_to_message_id=update.message.message_id,
                    )
                except Exception as img_err:
                    logger.warning("Image send failed", error=str(img_err))

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

    async def _handle_stop_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle stop: callbacks — interrupt a running Claude request."""
        query = update.callback_query
        target_user_id = int(query.data.split(":", 1)[1])

        # Only the requesting user can stop their own request
        if query.from_user.id != target_user_id:
            await query.answer(
                "Only the requesting user can stop this.", show_alert=True
            )
            return

        active = self._active_requests.get(target_user_id)
        if not active:
            await query.answer("Already completed.", show_alert=False)
            return
        if active.interrupted:
            await query.answer("Already stopping...", show_alert=False)
            return

        active.interrupt_event.set()
        active.interrupted = True
        await query.answer("Stopping...", show_alert=False)

        try:
            await active.progress_msg.edit_text("Stopping...", reply_markup=None)
        except Exception:
            pass

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
        """Interrupt the current turn immediately (like Ctrl-C / Esc).

        If bound to a tmux session, send the pane's interrupt key -- Esc for a
        Claude session (stops thinking without quitting), Ctrl-C for a plain
        shell. Otherwise signal the active headless request to abort.
        """
        chat_id = update.effective_chat.id
        binding = self._binding(chat_id)

        audit_logger = context.bot_data.get("audit_logger")
        if audit_logger:
            await audit_logger.log_command(
                user_id=update.effective_user.id,
                command="stop",
                args=[binding["target"]] if binding else [],
                success=True,
            )

        if binding:
            target = binding["target"]
            bridge = self._tmux(target)
            ok, detail = await bridge.available()
            if not ok:
                self._clear_binding(chat_id)
                self._stop_jsonl_mirror(chat_id)
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
            return

        # Headless mode: trip the active request's interrupt event.
        active = self._active_requests.get(update.effective_user.id)
        if active and not active.interrupt_event.is_set():
            active.interrupt_event.set()
            await update.message.reply_text("🛑 Stopping…")
        else:
            await update.message.reply_text("Nothing running to stop.")

    # --- tmux bridge (multi-session "tabs") ---

    def _tmux(self, target: str) -> TmuxBridge:
        return TmuxBridge(target)

    def _format_pane(self, snapshot: str, max_chars: int = 3500) -> str:
        """Render a pane snapshot as a Telegram-safe monospace block."""
        snapshot = snapshot.rstrip()
        if not snapshot:
            return "<i>(pane is empty)</i>"
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
    ) -> Optional[InlineKeyboardMarkup]:
        """Build an inline 'tabs' keyboard of every live tmux session."""
        sessions = await tmux_list_sessions()
        if not sessions:
            return None
        rows: List[list] = []  # type: ignore[type-arg]
        for s in sessions:
            name = str(s["name"])
            mark = "🟢 " if name == current else "▫️ "
            att = " ·live" if s["attached"] else ""
            label = f"{mark}{name} ({s['windows']}w{att})"
            rows.append([InlineKeyboardButton(label, callback_data=f"tmuxsel:{name}")])
        if current:
            rows.append([InlineKeyboardButton("🔌 Unbind", callback_data="tmuxoff")])
        return InlineKeyboardMarkup(rows)

    async def _bind_and_snapshot(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        target: str,
        *,
        edit: bool = False,
    ) -> None:
        """Bind the chat to a tmux session.

        If the pane is running a Claude session, start a live jsonl mirror that
        streams its messages here. Otherwise fall back to snapshot mode.
        """
        chat_id = update.effective_chat.id
        self._stop_jsonl_mirror(chat_id)
        await self._audit_tmux(update, context, f"bind:{target}")

        bridge = self._tmux(target)
        cwd = await bridge.pane_cwd()
        cmd = await bridge.pane_command()
        session_file = resolve_session_file(Path(cwd)) if cwd else None
        # A pane running `claude` is a mirror target even before its first
        # prompt (no .jsonl yet) -- the loop waits for the log to appear.
        is_claude = bool(session_file) or (cmd or "").lower() == "claude"

        if is_claude and cwd:
            self._set_binding(chat_id, target, "mirror")
            self._start_jsonl_mirror(
                chat_id,
                context.bot,
                target,
                claude_project_dir(Path(cwd)),
                session_file,
            )
            mode_line = (
                "🪞 <b>live mirror</b> on — I'll stream Claude's replies and "
                "tool calls here as they happen."
            )
        else:
            self._set_binding(chat_id, target, "snapshot")
            mode_line = (
                "📸 snapshot mode (not a Claude session) — /peek to refresh."
            )

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
        chat_id = update.effective_chat.id
        binding = self._binding(chat_id)
        if not binding:
            return
        target = binding["target"]
        mode = binding.get("mode", "snapshot")
        bridge = self._tmux(target)
        ok, detail = await bridge.available()
        if not ok:
            self._clear_binding(chat_id)
            self._stop_jsonl_mirror(chat_id)
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
            self._note_echo(chat_id, text)
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

    # --- live jsonl mirror ---

    def _start_jsonl_mirror(
        self,
        chat_id: int,
        bot: Any,
        target: str,
        proj_dir: Path,
        session_file: Optional[Path] = None,
    ) -> None:
        """(Re)start the background tailer streaming a session .jsonl to a chat."""
        self._stop_jsonl_mirror(chat_id)
        self._jsonl_mirrors[chat_id] = asyncio.create_task(
            self._jsonl_mirror_loop(chat_id, bot, target, proj_dir, session_file)
        )

    def _stop_jsonl_mirror(self, chat_id: int) -> None:
        task = self._jsonl_mirrors.pop(chat_id, None)
        if task and not task.done():
            task.cancel()
        self._mirror_echo.pop(chat_id, None)

    def _note_echo(self, chat_id: int, text: str) -> None:
        """Record a phone-sent prompt so its mirror echo can be suppressed."""
        lst = self._mirror_echo.setdefault(chat_id, [])
        lst.append(text.strip())
        del lst[:-10]  # keep only the last few

    def _consume_echo(self, chat_id: int, text: str) -> bool:
        """True if ``text`` matches a pending phone-sent prompt (and removes it)."""
        lst = self._mirror_echo.get(chat_id)
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
        self, bot: Any, chat_id: int, text: str, mode: Optional[str]
    ) -> None:
        """Deliver one rendered event, falling back to plain text on HTML error."""
        for chunk in self._split_message(text):
            try:
                await bot.send_message(chat_id=chat_id, text=chunk, parse_mode=mode)
            except Exception:
                try:
                    await bot.send_message(chat_id=chat_id, text=chunk)
                except Exception as e:
                    logger.warning("mirror send failed", chat_id=chat_id, error=str(e))
            await asyncio.sleep(0.4)  # gentle per-chat rate limit

    async def _check_pane_prompt(
        self, bot: Any, chat_id: int, target: str, last_prompt: Optional[str]
    ) -> Optional[str]:
        """Push an interactive TUI prompt to the chat if one is waiting.

        Returns the signature of the shown prompt (to dedupe), or None when no
        prompt is on screen (so the next one re-pushes).
        """
        try:
            screen = await self._tmux(target).capture()
        except Exception:
            return last_prompt
        if not is_interactive_prompt(screen):
            return None
        sig = prompt_signature(screen)
        if sig == last_prompt:
            return last_prompt  # already shown this exact prompt
        await self._mirror_send(
            bot,
            chat_id,
            "⌨️ <b>Claude needs your input:</b>\n"
            + self._format_pane(screen)
            + "\nReply with the option number, or "
            "/key Up · /key Down · /key Enter.",
            "HTML",
        )
        return sig

    async def _jsonl_mirror_loop(
        self,
        chat_id: int,
        bot: Any,
        target: str,
        proj_dir: Path,
        session_file: Optional[Path] = None,
    ) -> None:
        """Tail a Claude session .jsonl and push each new record to the chat.

        If no log exists yet (pane just started ``claude``), waits for one to
        appear and streams it from the start. An existing log is tailed from
        EOF (only new events from bind onward). Follows the newest file in the
        project dir so a /clear (new session file) is picked up, and resets on
        truncation.

        Also watches the pane screen for interactive prompts (menus, permission
        requests) -- those are drawn by the TUI and never hit the .jsonl, so
        they'd otherwise be invisible.
        """
        import json as _json
        from .features.claude_jsonl import _newest_jsonl

        current = session_file
        try:
            offset = current.stat().st_size if current else 0
        except OSError:
            offset = 0
        buf = ""
        poll = 0
        last_prompt: Optional[str] = None
        try:
            while True:
                await asyncio.sleep(1.0)
                poll += 1

                # Surface interactive TUI prompts (not present in the .jsonl).
                if poll % 2 == 0:
                    last_prompt = await self._check_pane_prompt(
                        bot, chat_id, target, last_prompt
                    )

                # Waiting for the first log, or following a new one (/clear).
                if current is None:
                    current = _newest_jsonl(proj_dir)
                    if current is None:
                        continue
                    offset, buf = 0, ""  # brand-new session: stream from start
                elif poll % 5 == 0:
                    newest = _newest_jsonl(proj_dir)
                    if newest and newest != current:
                        current, offset, buf = newest, 0, ""
                try:
                    size = current.stat().st_size
                except OSError:
                    continue
                if size < offset:  # truncated/rotated
                    offset, buf = 0, ""
                if size == offset:
                    continue
                with current.open("r", encoding="utf-8", errors="replace") as f:
                    f.seek(offset)
                    buf += f.read()
                    offset = f.tell()
                lines = buf.split("\n")
                buf = lines.pop()  # keep the incomplete trailing line
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
                    if raw is not None and self._consume_echo(chat_id, raw):
                        continue
                    rendered = render_record(obj)
                    if rendered:
                        await self._mirror_send(bot, chat_id, rendered[0], rendered[1])
        except asyncio.CancelledError:
            pass
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("jsonl mirror loop crashed", chat_id=chat_id, error=str(e))

    async def agentic_tmux(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Bind chat to a tmux session: /tmux [name|off]. No arg = tab picker."""
        args = update.message.text.split()[1:] if update.message.text else []
        b = self._binding(update.effective_chat.id)
        current = b["target"] if b else None

        if args and args[0].lower() == "off":
            self._clear_binding(update.effective_chat.id)
            self._stop_jsonl_mirror(update.effective_chat.id)
            await self._audit_tmux(update, context, "off")
            await update.message.reply_text(
                "🔌 Unbound. Messages go to headless Claude again."
            )
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
        kb = await self._tmux_session_keyboard(current)
        if kb is None:
            await update.message.reply_text(
                "No live tmux sessions. On the PC, start one, e.g.:\n"
                "<code>tmux new -s claude claude</code>",
                parse_mode="HTML",
            )
            return
        bound = (
            f"<code>{escape_html(current)}</code>" if current else "none"
        )
        await update.message.reply_text(
            f"🖥 <b>tmux sessions</b> — bound: {bound}\nTap a tab to switch:",
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
            self._clear_binding(update.effective_chat.id)
            self._stop_jsonl_mirror(update.effective_chat.id)
            await self._audit_tmux(update, context, "off")
            await query.edit_message_text(
                "🔌 Unbound. Messages go to headless Claude again."
            )
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

    async def agentic_peek(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Snapshot the bound tmux session: /peek [scrollback_lines]."""
        b = self._binding(update.effective_chat.id)
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
            self._clear_binding(update.effective_chat.id)
            self._stop_jsonl_mirror(update.effective_chat.id)
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
        b = self._binding(update.effective_chat.id)
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
            self._clear_binding(update.effective_chat.id)
            self._stop_jsonl_mirror(update.effective_chat.id)
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
