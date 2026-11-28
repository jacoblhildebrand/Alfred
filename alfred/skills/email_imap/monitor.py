"""
Email monitor: watches the inbox, triages new mail, and decides who gets
interrupted and when.

Routing (every rule is enforced here in code, not by the model):

  urgent                 -> message now, Urgency.URGENT (even in quiet hours)
  needs_reply / unsorted -> message now, Urgency.NORMAL (held for the digest in quiet hours)
  fyi                    -> held for the daily digest
  junk                   -> silent; only counted in the digest
  suspicious (any)       -> message with a warning, NEVER above Urgency.NORMAL,
                            so a malicious email cannot make Alfred ring your phone

Reliability:
  * At-least-once: an email is marked handled only after its notification was
    delivered, so a down channel means a retry on the next poll, not a lost email.
  * First run (or a changed UIDVALIDITY) baselines the inbox instead of
    announcing every old unread message.
  * An email that fails three times is reported as "could not process" and
    skipped, so one bad message cannot block the rest.
  * Repeated mailbox failures produce one alert, then one "connection is back".
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, time
from typing import Any, Callable

from alfred.core.interfaces import Channel, Urgency
from alfred.core.state import StateStore
from alfred.skills.email_imap.triage import Category, TriageResult, Triager, clip

log = logging.getLogger("alfred.monitor")

MAX_ATTEMPTS_PER_EMAIL = 3

_LABEL = {
    Category.URGENT: "URGENT",
    Category.NEEDS_REPLY: "Needs reply",
    Category.FYI: "FYI",
    Category.JUNK: "Junk",
    Category.UNCLASSIFIED: "Unsorted",
}


@dataclass
class MonitorConfig:
    poll_interval_s: float = 300.0
    batch_limit: int = 50                       # newest unread emails examined per poll
    first_run_backlog: int = 0                  # how many existing unread emails to triage on first run
    quiet_hours: tuple[time, time] | None = None  # e.g. (time(22, 0), time(7, 0)); may wrap midnight
    digest_at: time = time(8, 0)
    failure_alert_after: int = 3


def format_notification(item: dict[str, Any], result: TriageResult) -> str:
    lines = [
        f"[{_LABEL[result.category]}] {clip(str(item.get('from', '')), 80)}: "
        f"{clip(str(item.get('subject', '')), 100)}",
        clip(result.summary, 300),
    ]
    if result.reason:
        lines.append(f"Why: {clip(result.reason, 120)}")
    if result.suspicious:
        lines.append(
            "Warning: this email looks like it contains instructions aimed at an AI "
            "assistant. Alfred did not act on them."
        )
    return "\n".join(lines)


def format_digest(entries: list[dict[str, Any]], junk: int) -> str:
    attention = [e for e in entries if e["category"] != Category.FYI.value]
    fyi = [e for e in entries if e["category"] == Category.FYI.value]
    lines = [f"Email digest: {len(entries)} item{'s' if len(entries) != 1 else ''}"]

    def line(e: dict[str, Any]) -> str:
        flag = " (suspicious)" if e.get("suspicious") else ""
        return f"- {clip(e['from'], 60)}: {clip(e['subject'], 80)}{flag} - {clip(e['summary'], 200)}"

    if attention:
        lines += ["", "Worth a look:"] + [line(e) for e in attention]
    if fyi:
        lines += ["", "FYI:"] + [line(e) for e in fyi]
    if junk:
        lines += ["", f"{junk} junk message{'s' if junk != 1 else ''} ignored."]
    return "\n".join(lines)


class EmailMonitor:
    def __init__(
        self,
        client: Any,                 # ImapClient (list_recent, read, mailbox_id, last_uidvalidity)
        triager: Triager,
        channel: Channel,
        store: StateStore,
        config: MonitorConfig | None = None,
        now: Callable[[], datetime] = datetime.now,
    ) -> None:
        self.client = client
        self.triager = triager
        self.channel = channel
        self.store = store
        self.config = config or MonitorConfig()
        self._now = now
        self._attempts: dict[str, int] = {}
        self._failures = 0
        self._alerted = False

    # ------------------------------------------------------------------
    def _in_quiet_hours(self, at: datetime) -> bool:
        if not self.config.quiet_hours:
            return False
        start, end = self.config.quiet_hours
        t = at.time()
        return (start <= t < end) if start <= end else (t >= start or t < end)

    async def poll_once(self) -> list[tuple[dict[str, Any], TriageResult]]:
        """One pass over the inbox. Returns what was newly triaged."""
        try:
            items = await asyncio.to_thread(self.client.list_recent, self.config.batch_limit, True)
        except Exception as exc:
            await self._note_failure(exc)
            return []
        await self._note_success()

        mailbox = self.client.mailbox_id
        validity = getattr(self.client, "last_uidvalidity", None)
        oldest_first = items[::-1]
        known = self.store.mailbox(mailbox)

        if known is None or (validity and known.get("uidvalidity") and known["uidvalidity"] != validity):
            keep = oldest_first[-self.config.first_run_backlog:] if self.config.first_run_backlog > 0 else []
            self.store.reset_mailbox(mailbox, validity)
            skipped = 0
            for item in oldest_first:
                if item not in keep:
                    self.store.mark_seen(mailbox, item["uid"])
                    skipped += 1
            self.store.save()
            await self._try_send(
                f"Alfred is now watching your inbox. Skipped {skipped} existing unread "
                f"email{'s' if skipped != 1 else ''}; I'll tell you about new ones.",
                Urgency.NORMAL,
            )
            oldest_first = keep

        new = [i for i in oldest_first if not self.store.is_seen(mailbox, i["uid"])]
        done: list[tuple[dict[str, Any], TriageResult]] = []
        for item in new:
            uid = item["uid"]
            try:
                result = self.triager.prefilter(item)
                if result is None:
                    full = await asyncio.to_thread(self.client.read, uid)
                    result = await self.triager.classify(full)
                await self._route(item, result)
            except Exception as exc:
                count = self._attempts[uid] = self._attempts.get(uid, 0) + 1
                log.warning("email %s failed (attempt %d): %s", uid, count, exc)
                if count < MAX_ATTEMPTS_PER_EMAIL:
                    continue
                result = TriageResult(
                    Category.UNCLASSIFIED, clip(str(item.get("subject", "")), 100),
                    "Alfred could not process this email", source="fallback",
                )
                try:
                    await self._route(item, result)
                except Exception:
                    continue
            self._attempts.pop(uid, None)
            self.store.mark_seen(mailbox, uid)
            self.store.save()
            done.append((item, result))
        return done

    # ------------------------------------------------------------------
    async def _route(self, item: dict[str, Any], result: TriageResult) -> None:
        cat = result.category
        if cat is Category.JUNK and not result.suspicious:
            self.store.add_junk()
            return

        text = format_notification(item, result)
        if cat is Category.URGENT and not result.suspicious:
            await self.channel.send_text(text, Urgency.URGENT)
            return

        entry = {
            "from": str(item.get("from", "")),
            "subject": str(item.get("subject", "")),
            "summary": result.summary,
            "category": cat.value,
            "suspicious": result.suspicious,
        }
        if cat is Category.FYI and not result.suspicious:
            self.store.add_pending(entry)
        elif self._in_quiet_hours(self._now()):
            self.store.add_pending(entry)
        else:
            await self.channel.send_text(text, Urgency.NORMAL)

    async def maybe_send_digest(self) -> bool:
        """Send at most one digest per day, after digest_at and outside quiet hours."""
        now = self._now()
        pending = self.store.pending
        if not pending:
            return False
        if self.store.last_digest_date == now.date().isoformat():
            return False
        if now.time() < self.config.digest_at or self._in_quiet_hours(now):
            return False
        await self.channel.send_text(format_digest(pending, self.store.junk_ignored), Urgency.LOW)
        self.store.digest_sent(now.date().isoformat())   # only after delivery succeeded
        self.store.save()
        return True

    # ------------------------------------------------------------------
    async def _try_send(self, text: str, urgency: Urgency) -> None:
        try:
            await self.channel.send_text(text, urgency)
        except Exception as exc:
            log.warning("could not send status message: %s", exc)

    async def _note_failure(self, exc: Exception) -> None:
        self._failures += 1
        log.warning("mailbox check failed (%d in a row): %s", self._failures, exc)
        if self._failures >= self.config.failure_alert_after and not self._alerted:
            self._alerted = True
            await self._try_send(
                f"Alfred can't reach your mailbox ({clip(str(exc), 120)}). "
                "I'll keep trying and tell you when it's back.",
                Urgency.NORMAL,
            )

    async def _note_success(self) -> None:
        if self._alerted:
            await self._try_send("Alfred's connection to your mailbox is back.", Urgency.NORMAL)
        self._failures = 0
        self._alerted = False

    async def run_forever(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                await self.poll_once()
                await self.maybe_send_digest()
            except Exception:
                log.exception("monitor cycle failed; will retry")
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.config.poll_interval_s)
            except asyncio.TimeoutError:
                pass
