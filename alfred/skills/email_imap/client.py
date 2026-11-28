"""
Read-only IMAP access.

Safety properties (all covered by tests):
  * The mailbox is opened with readonly=True, so the server will not even set
    the \\Seen flag. Messages are fetched with BODY.PEEK for the same reason.
  * Only the IMAP commands SEARCH and FETCH can be sent. Anything else
    (STORE, COPY, EXPUNGE, APPEND, ...) raises before reaching the server.
  * Output is bounded: body length, attachment count and header length are
    all capped, and attachment CONTENTS are never returned, only their names.
  * Credentials come from environment variables, never from code or files,
    and are hidden from repr() so they don't leak into logs.
"""
from __future__ import annotations

import imaplib
import os
import re
from dataclasses import dataclass, field
from email import message_from_bytes, policy
from html.parser import HTMLParser
from typing import Any, Callable, Mapping

_ALLOWED_COMMANDS = {"SEARCH", "FETCH"}
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]+")
_UID = re.compile(r"\d{1,12}")

# A cheap hint for the model and the user, NOT a security control: attackers
# can phrase things any number of ways. The real defence is the policy engine
# refusing to let untrusted content trigger actions on its own.
_INJECTION_PATTERNS = [
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|prompts?)",
    r"disregard (all |any |the )?(previous|prior|above|earlier)",
    r"you are now\b",
    r"\b(admin|developer|debug) mode\b",
    r"system prompt",
    r"do not (tell|inform|alert) the user",
    r"(forward|send|email) (all|every|any) .{0,40}(to|at) \S+@\S+",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


class MailError(RuntimeError):
    pass


@dataclass
class EmailConfig:
    host: str
    username: str
    password: str = field(repr=False)
    port: int = 993
    mailbox: str = "INBOX"
    max_body_chars: int = 4000
    max_list: int = 20
    timeout_s: float = 30.0

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "EmailConfig":
        env = os.environ if env is None else env
        names = ("ALFRED_IMAP_HOST", "ALFRED_IMAP_USER", "ALFRED_IMAP_PASSWORD")
        missing = [n for n in names if not env.get(n)]
        if missing:
            raise ValueError("Missing environment variables: " + ", ".join(missing))
        return cls(
            host=env["ALFRED_IMAP_HOST"],
            username=env["ALFRED_IMAP_USER"],
            password=env["ALFRED_IMAP_PASSWORD"],
            port=int(env.get("ALFRED_IMAP_PORT", "993")),
            mailbox=env.get("ALFRED_IMAP_MAILBOX", "INBOX"),
        )


def injection_flags(*texts: str) -> list[str]:
    return ["possible_prompt_injection"] if any(_INJECTION_RE.search(t or "") for t in texts) else []


# --------------------------------------------------------------------------
# parsing helpers
# --------------------------------------------------------------------------
def _clean(value: str, limit: int = 200) -> str:
    return _CONTROL_CHARS.sub(" ", value).strip()[:limit]


def _header(msg: Any, name: str) -> str:
    try:
        return _clean(str(msg.get(name, "") or ""))
    except Exception:  # malformed headers must not break listing
        return ""


class _TextExtractor(HTMLParser):
    _BLOCK = {"p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote"}
    _SKIP = {"script", "style", "head", "title"}

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._BLOCK:
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag in self._BLOCK:
            self._parts.append("\n")

    def handle_data(self, data):
        if not self._skip_depth:
            self._parts.append(data)

    def text(self) -> str:
        return "".join(self._parts)


def _html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    return parser.text()


def _body_text(msg: Any, limit: int) -> str:
    part = msg.get_body(preferencelist=("plain", "html"))
    if part is None:
        return "(no text body)"
    try:
        content = part.get_content()
    except Exception:
        return "(could not decode body)"
    if part.get_content_type() == "text/html":
        content = _html_to_text(content)
    content = re.sub(r"[ \t]+\n", "\n", content)
    content = re.sub(r"\n{3,}", "\n\n", content).strip()
    if len(content) > limit:
        content = content[:limit] + f"\n[truncated {len(content) - limit} more characters]"
    return content or "(empty body)"


# --------------------------------------------------------------------------
# client
# --------------------------------------------------------------------------
class ImapClient:
    def __init__(
        self,
        config: EmailConfig,
        connect: Callable[[EmailConfig], Any] | None = None,
    ) -> None:
        self._cfg = config
        self._connect = connect or self._default_connect
        # IMAP UIDs are only meaningful together with the mailbox's UIDVALIDITY:
        # if the server changes it, every old UID is invalid. Set on each open.
        self.last_uidvalidity: str | None = None

    @property
    def mailbox_id(self) -> str:
        return f"{self._cfg.username}@{self._cfg.host}/{self._cfg.mailbox}"

    @staticmethod
    def _default_connect(cfg: EmailConfig) -> Any:
        return imaplib.IMAP4_SSL(cfg.host, cfg.port, timeout=cfg.timeout_s)

    def _open(self) -> Any:
        conn = self._connect(self._cfg)
        try:
            conn.login(self._cfg.username, self._cfg.password)
            status, _ = conn.select(self._cfg.mailbox, readonly=True)
            if status != "OK":
                raise MailError(f"could not open mailbox {self._cfg.mailbox!r}")
            try:
                _, validity = conn.response("UIDVALIDITY")
                self.last_uidvalidity = validity[0].decode() if validity and validity[0] else None
            except Exception:
                self.last_uidvalidity = None
        except Exception:
            self._close(conn)
            raise
        return conn

    @staticmethod
    def _close(conn: Any) -> None:
        try:
            conn.logout()
        except Exception:
            pass

    @staticmethod
    def _uid(conn: Any, command: str, *args: Any) -> list[Any]:
        if command.upper() not in _ALLOWED_COMMANDS:
            raise MailError(f"IMAP command {command!r} is not allowed (read-only skill)")
        status, data = conn.uid(command, *args)
        if status != "OK":
            raise MailError(f"IMAP {command} failed: {status}")
        return data

    # ------------------------------------------------------------------
    def list_recent(self, limit: int, unread_only: bool = False) -> list[dict[str, Any]]:
        """Newest first. Each item: uid, from, subject, date, unread, flags."""
        conn = self._open()
        try:
            data = self._uid(conn, "SEARCH", None, "UNSEEN" if unread_only else "ALL")
            found = data[0].split() if data and data[0] else []
            newest = [u.decode() for u in found[-limit:]][::-1]
            if not newest:
                return []
            data = self._uid(
                conn, "FETCH", ",".join(newest),
                "(UID FLAGS BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)])",
            )
            by_uid: dict[str, dict[str, Any]] = {}
            for item in data:
                if not isinstance(item, tuple):
                    continue
                meta = item[0].decode(errors="replace")
                uid = re.search(r"UID (\d+)", meta)
                if not uid:
                    continue
                flags = re.search(r"FLAGS \(([^)]*)\)", meta)
                headers = message_from_bytes(item[1], policy=policy.default)
                subject = _header(headers, "Subject") or "(no subject)"
                by_uid[uid.group(1)] = {
                    "uid": uid.group(1),
                    "from": _header(headers, "From") or "(unknown sender)",
                    "subject": subject,
                    "date": _header(headers, "Date") or "(no date)",
                    "unread": "\\Seen" not in (flags.group(1) if flags else ""),
                    "flags": injection_flags(subject),
                }
            return [by_uid[u] for u in newest if u in by_uid]
        finally:
            self._close(conn)

    def read(self, uid: str) -> dict[str, Any]:
        if not _UID.fullmatch(uid or ""):
            raise MailError(f"invalid uid {uid!r}")
        conn = self._open()
        try:
            data = self._uid(conn, "FETCH", uid, "(BODY.PEEK[])")
            raw = next((item[1] for item in data if isinstance(item, tuple)), None)
            if raw is None:
                raise MailError(f"no message with uid {uid}")
            msg = message_from_bytes(raw, policy=policy.default)
            subject = _header(msg, "Subject") or "(no subject)"
            body = _body_text(msg, self._cfg.max_body_chars)
            attachments = [
                {
                    "filename": _clean(part.get_filename() or "(unnamed)", 120),
                    "type": part.get_content_type(),
                    "bytes": len(part.get_payload(decode=True) or b""),
                }
                for part in list(msg.iter_attachments())[:20]
            ]
            return {
                "uid": uid,
                "from": _header(msg, "From") or "(unknown sender)",
                "to": _header(msg, "To") or "(unknown)",
                "subject": subject,
                "date": _header(msg, "Date") or "(no date)",
                "body": body,
                "attachments": attachments,
                "flags": injection_flags(subject, body),
            }
        finally:
            self._close(conn)
