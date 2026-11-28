"""Email monitor tests: fake mailbox, fake model, fake channel, fake clock."""
import asyncio
import json
import unittest
from datetime import datetime, time

from alfred.core.interfaces import Urgency
from alfred.core.state import StateStore
from alfred.skills.email_imap.client import MailError
from alfred.skills.email_imap.monitor import EmailMonitor, MonitorConfig
from alfred.skills.email_imap.triage import Category, TriageRules, Triager
from alfred.testing import FakeChannel, FakeProvider, say

NOON = datetime(2026, 11, 27, 12, 0)
LATE = datetime(2026, 11, 27, 23, 0)


def verdict(category, summary="Summary text", reason="because", suspicious=False):
    return say(json.dumps({"category": category, "summary": summary, "reason": reason, "suspicious": suspicious}))


class FakeMailClient:
    mailbox_id = "me@example.com/INBOX"

    def __init__(self):
        self.last_uidvalidity = "100"
        self.messages = {}
        self.unread = set()
        self.fail = False
        self.read_fail = set()
        self.list_calls = 0
        self.read_calls = []

    def add(self, uid, sender="Sam <sam@example.com>", subject="Hi", body="body", flags=None):
        self.messages[uid] = {"from": sender, "subject": subject, "date": "Fri", "body": body, "flags": flags or []}
        self.unread.add(uid)

    def list_recent(self, limit, unread_only=False):
        self.list_calls += 1
        if self.fail:
            raise MailError("connection lost")
        uids = sorted((u for u in self.messages if not unread_only or u in self.unread), reverse=True)[:limit]
        return [
            {"uid": str(u), "from": self.messages[u]["from"], "subject": self.messages[u]["subject"],
             "date": "Fri", "unread": True, "flags": self.messages[u]["flags"]}
            for u in uids
        ]

    def read(self, uid):
        self.read_calls.append(uid)
        if int(uid) in self.read_fail:
            raise MailError("cannot read")
        return {"uid": uid, **self.messages[int(uid)]}


class FlakyChannel(FakeChannel):
    def __init__(self):
        super().__init__()
        self.fail_next = 0

    async def send_text(self, text, urgency=Urgency.NORMAL):
        if self.fail_next:
            self.fail_next -= 1
            raise RuntimeError("channel down")
        await super().send_text(text, urgency)


def build(script=(), rules=None, config=None, channel=None, clock=NOON, backlog_emails=()):
    """Returns monitor, client, provider, channel, store, clock. Inbox is baselined if backlog_emails given."""
    client = FakeMailClient()
    for uid in backlog_emails:
        client.add(uid)
    provider = FakeProvider(list(script))
    channel = channel or FakeChannel()
    store = StateStore()
    clock_box = {"now": clock}
    monitor = EmailMonitor(client, Triager(provider, rules), channel, store,
                           config or MonitorConfig(), now=lambda: clock_box["now"])
    return monitor, client, provider, channel, store, clock_box


async def baselined(script=(), **kw):
    monitor, client, provider, channel, store, clock = build(script, backlog_emails=[1], **kw)
    await monitor.poll_once()                 # baseline: skips email 1
    channel.texts.clear()
    return monitor, client, provider, channel, store, clock


def texts(channel):
    return [t for t, _ in channel.texts]


class FirstRun(unittest.IsolatedAsyncioTestCase):
    async def test_existing_unread_mail_is_skipped_not_announced(self):
        monitor, client, provider, channel, store, _ = build(backlog_emails=[1, 2, 3])
        self.assertEqual(await monitor.poll_once(), [])
        self.assertEqual(provider.seen, [])                      # no model calls
        self.assertEqual(client.read_calls, [])
        self.assertEqual(len(channel.texts), 1)
        self.assertIn("Skipped 3", channel.texts[0][0])
        self.assertTrue(all(store.is_seen(client.mailbox_id, str(u)) for u in (1, 2, 3)))

    async def test_new_mail_after_baseline_is_triaged(self):
        monitor, client, provider, channel, *_ = await baselined([verdict("needs_reply")])
        client.add(2, subject="Lunch?")
        done = await monitor.poll_once()
        self.assertEqual([(i["uid"], r.category) for i, r in done], [("2", Category.NEEDS_REPLY)])
        self.assertEqual(len(channel.texts), 1)

    async def test_backlog_option_triages_only_the_newest_existing_emails(self):
        monitor, client, provider, channel, *_ = build([verdict("fyi")], backlog_emails=[1, 2, 3],
                                                      config=MonitorConfig(first_run_backlog=1))
        done = await monitor.poll_once()
        self.assertEqual([i["uid"] for i, _ in done], ["3"])
        self.assertEqual(len(provider.seen), 1)

    async def test_changed_uidvalidity_rebaselines_instead_of_reannouncing(self):
        monitor, client, provider, channel, *_ = await baselined()
        client.add(2)
        client.last_uidvalidity = "200"
        self.assertEqual(await monitor.poll_once(), [])
        self.assertEqual(provider.seen, [])
        self.assertIn("Skipped", channel.texts[0][0])


class Routing(unittest.IsolatedAsyncioTestCase):
    async def run_one(self, completion, **kw):
        monitor, client, provider, channel, store, clock = await baselined([completion], **kw)
        client.add(2, subject="Subject line")
        await monitor.poll_once()
        return channel, store, monitor

    async def test_urgent_goes_out_now_as_urgent(self):
        channel, *_ = await self.run_one(verdict("urgent"))
        self.assertEqual([u for _, u in channel.texts], [Urgency.URGENT])
        self.assertIn("[URGENT]", channel.texts[0][0])

    async def test_needs_reply_goes_out_now_as_normal(self):
        channel, *_ = await self.run_one(verdict("needs_reply"))
        self.assertEqual([u for _, u in channel.texts], [Urgency.NORMAL])

    async def test_fyi_waits_for_the_digest(self):
        channel, store, _ = await self.run_one(verdict("fyi"))
        self.assertEqual(channel.texts, [])
        self.assertEqual(len(store.pending), 1)

    async def test_junk_is_silent_and_counted(self):
        channel, store, _ = await self.run_one(verdict("junk"))
        self.assertEqual((channel.texts, store.pending, store.junk_ignored), ([], [], 1))

    async def test_failed_classification_still_shows_the_email(self):
        channel, *_ = await self.run_one(say("total nonsense"))
        self.assertEqual(len(channel.texts), 1)
        self.assertIn("[Unsorted]", channel.texts[0][0])
        self.assertIn("Subject line", channel.texts[0][0])

    async def test_suspicious_urgent_can_never_be_urgent(self):
        channel, *_ = await self.run_one(verdict("urgent", suspicious=True))
        self.assertEqual([u for _, u in channel.texts], [Urgency.NORMAL])
        self.assertIn("instructions aimed at an AI", channel.texts[0][0])

    async def test_suspicious_junk_and_fyi_are_surfaced_not_hidden(self):
        for category in ("junk", "fyi"):
            with self.subTest(category=category):
                channel, store, _ = await self.run_one(verdict(category, suspicious=True))
                self.assertEqual(len(channel.texts), 1)
                self.assertEqual(store.junk_ignored, 0)

    async def test_scanner_flag_on_the_email_marks_it_suspicious(self):
        monitor, client, provider, channel, *_ = await baselined([verdict("fyi")])
        client.add(2, flags=["possible_prompt_injection"])
        await monitor.poll_once()
        self.assertEqual(len(channel.texts), 1)
        self.assertIn("Warning", channel.texts[0][0])

    async def test_ignored_sender_never_reaches_the_model_or_the_body_fetch(self):
        rules = TriageRules(ignore_senders={"@promo.example"})
        monitor, client, provider, channel, store, _ = await baselined(rules=rules)
        client.add(2, sender="Deals <x@promo.example>")
        await monitor.poll_once()
        self.assertEqual((provider.seen, client.read_calls, channel.texts), ([], [], []))
        self.assertEqual(store.junk_ignored, 1)

    async def test_vip_junk_becomes_a_notification(self):
        rules = TriageRules(vip_senders={"boss@corp.com"})
        monitor, client, provider, channel, *_ = await baselined([verdict("junk")], rules=rules)
        client.add(2, sender="Boss <boss@corp.com>")
        await monitor.poll_once()
        self.assertEqual([u for _, u in channel.texts], [Urgency.NORMAL])


class QuietHoursAndDigest(unittest.IsolatedAsyncioTestCase):
    QUIET = MonitorConfig(quiet_hours=(time(22, 0), time(7, 0)))

    async def test_quiet_hours_hold_normal_mail_but_not_urgent(self):
        monitor, client, provider, channel, store, _ = await baselined(
            [verdict("needs_reply"), verdict("urgent")], config=self.QUIET, clock=LATE)
        client.add(2)
        client.add(3)
        await monitor.poll_once()
        self.assertEqual([u for _, u in channel.texts], [Urgency.URGENT])   # only the urgent one
        self.assertEqual(len(store.pending), 1)

    async def test_quiet_hours_window_wraps_midnight(self):
        monitor, *_ = build(config=self.QUIET)
        cases = [(21, 30, False), (22, 0, True), (23, 0, True), (0, 0, True),
                 (6, 30, True), (7, 0, False), (12, 0, False)]
        for hour, minute, expected in cases:
            with self.subTest(at=f"{hour:02d}:{minute:02d}"):
                self.assertEqual(monitor._in_quiet_hours(datetime(2026, 11, 27, hour, minute)), expected)

    async def test_digest_summarises_held_items_and_junk_once_per_day(self):
        monitor, client, provider, channel, store, clock = await baselined(
            [verdict("fyi", summary="Receipt"), verdict("junk"), verdict("junk"), verdict("needs_reply")], config=self.QUIET, clock=LATE)
        for uid in (2, 3, 4, 5):
            client.add(uid)
        await monitor.poll_once()
        self.assertEqual(channel.texts, [])                       # all held overnight
        clock["now"] = datetime(2026, 11, 28, 9, 0)
        self.assertTrue(await monitor.maybe_send_digest())
        digest, urgency = channel.texts[0]
        self.assertEqual(urgency, Urgency.LOW)
        self.assertTrue(digest.startswith("Email digest: 2 items"))
        self.assertIn("Receipt", digest)
        self.assertIn("2 junk messages ignored", digest)
        self.assertEqual((store.pending, store.junk_ignored), ([], 0))
        client.add(6)
        monitor.triager._provider._script.append(verdict("fyi"))
        await monitor.poll_once()
        self.assertFalse(await monitor.maybe_send_digest())       # already sent today

    async def test_digest_waits_for_digest_time_and_for_quiet_hours_to_end(self):
        monitor, client, provider, channel, store, clock = await baselined([verdict("fyi")], config=self.QUIET)
        client.add(2)
        await monitor.poll_once()
        for early in (datetime(2026, 11, 27, 6, 0), datetime(2026, 11, 27, 7, 30), datetime(2026, 11, 27, 23, 0)):
            clock["now"] = early
            with self.subTest(at=early):
                self.assertFalse(await monitor.maybe_send_digest())
        clock["now"] = NOON
        self.assertTrue(await monitor.maybe_send_digest())

    async def test_no_digest_for_junk_alone(self):
        monitor, client, provider, channel, store, _ = await baselined([verdict("junk")])
        client.add(2)
        await monitor.poll_once()
        self.assertFalse(await monitor.maybe_send_digest())

    async def test_failed_digest_delivery_keeps_the_items(self):
        channel = FlakyChannel()
        monitor, client, provider, _, store, _ = await baselined([verdict("fyi")], channel=channel)
        client.add(2)
        await monitor.poll_once()
        channel.fail_next = 1
        with self.assertRaises(RuntimeError):
            await monitor.maybe_send_digest()
        self.assertEqual(len(store.pending), 1)
        self.assertTrue(await monitor.maybe_send_digest())


class Reliability(unittest.IsolatedAsyncioTestCase):
    async def test_already_handled_mail_is_not_announced_again(self):
        monitor, client, provider, channel, *_ = await baselined([verdict("needs_reply")])
        client.add(2)
        await monitor.poll_once()
        await monitor.poll_once()
        await monitor.poll_once()
        self.assertEqual(len(channel.texts), 1)
        self.assertEqual(len(provider.seen), 1)

    async def test_channel_failure_means_retry_not_a_lost_email(self):
        channel = FlakyChannel()
        monitor, client, provider, _, store, _ = await baselined([verdict("urgent"), verdict("urgent")], channel=channel)
        client.add(2)
        channel.fail_next = 1
        self.assertEqual(await monitor.poll_once(), [])           # delivery failed: not marked handled
        self.assertFalse(store.is_seen(client.mailbox_id, "2"))
        done = await monitor.poll_once()                          # retried and delivered
        self.assertEqual(len(done), 1)
        self.assertEqual(len(channel.texts), 1)

    async def test_an_unreadable_email_is_reported_after_three_tries_and_skipped(self):
        monitor, client, provider, channel, store, _ = await baselined([verdict("fyi")])
        client.add(2, subject="Broken one")
        client.add(3)
        client.read_fail = {2}
        await monitor.poll_once()                                  # 3 is fine, 2 fails (attempt 1)
        self.assertEqual(len(provider.seen), 1)
        await monitor.poll_once()                                  # attempt 2
        self.assertEqual(texts(channel), [])
        await monitor.poll_once()                                  # attempt 3 -> reported
        self.assertEqual(len(channel.texts), 1)
        self.assertIn("could not process", channel.texts[0][0])
        self.assertIn("Broken one", channel.texts[0][0])
        await monitor.poll_once()
        self.assertEqual(len(channel.texts), 1)                    # and never again

    async def test_one_bad_email_does_not_block_the_next(self):
        monitor, client, provider, channel, *_ = await baselined([verdict("needs_reply")])
        client.add(2)
        client.add(3, subject="Good one")
        client.read_fail = {2}
        done = await monitor.poll_once()
        self.assertEqual([i["uid"] for i, _ in done], ["3"])

    async def test_mailbox_outage_alerts_once_then_reports_recovery(self):
        monitor, client, provider, channel, *_ = await baselined()
        client.fail = True
        for _ in range(2):
            await monitor.poll_once()
        self.assertEqual(channel.texts, [])                        # below the threshold
        for _ in range(4):
            await monitor.poll_once()
        alerts = [t for t in texts(channel) if "can't reach" in t]
        self.assertEqual(len(alerts), 1)                           # one alert, not one per poll
        client.fail = False
        await monitor.poll_once()
        self.assertEqual(len([t for t in texts(channel) if "is back" in t]), 1)

    async def test_run_forever_polls_repeatedly_and_stops_on_request(self):
        monitor, client, *_ = build(config=MonitorConfig(poll_interval_s=0.01), backlog_emails=[1])
        stop = asyncio.Event()
        task = asyncio.create_task(monitor.run_forever(stop))
        for _ in range(300):
            if client.list_calls >= 3:
                break
            await asyncio.sleep(0.01)
        stop.set()
        await asyncio.wait_for(task, 1)
        self.assertGreaterEqual(client.list_calls, 3)

    async def test_run_forever_survives_an_unexpected_error(self):
        monitor, client, *_ = build(config=MonitorConfig(poll_interval_s=0.01), backlog_emails=[1])
        original, calls = monitor.poll_once, []

        async def flaky():
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("surprise")
            return await original()

        monitor.poll_once = flaky
        stop = asyncio.Event()
        task = asyncio.create_task(monitor.run_forever(stop))
        for _ in range(300):
            if len(calls) >= 3:
                break
            await asyncio.sleep(0.01)
        stop.set()
        await asyncio.wait_for(task, 1)
        self.assertGreaterEqual(len(calls), 3)


if __name__ == "__main__":
    unittest.main()
