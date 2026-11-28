"""
Watch your real inbox with Alfred's triage. Read-only: nothing is sent, marked as
read, moved or deleted. Notifications print in this terminal until the Telegram
channel exists.

    py run_monitor.py --once                  # one pass, then exit (try this first)
    py run_monitor.py                         # keep watching, poll every 5 minutes
    py run_monitor.py --digest-now --once     # also send the daily digest right now

Useful options:
    --model qwen3.8:27b        which Ollama model to use for triage
    --backlog 3                on the first run, also triage the 3 newest existing unread emails
    --vip boss@corp.com        always notify about this sender (repeatable; "@corp.com" = whole domain)
    --ignore @promo.example    never bother me about this sender (repeatable)
    --quiet 22:00-07:00        hold non-urgent notifications overnight
    --context "I'm a student; emails about classes and deadlines matter"

Needs the ALFRED_IMAP_* environment variables (see check_email.py) and a running
Ollama. State (which emails were already handled) is kept in alfred_state.json.
"""
import argparse
import asyncio
import logging
from datetime import time

from alfred.channels.cli import CliChannel
from alfred.core.state import StateStore
from alfred.providers.openai_compat import OpenAICompatProvider
from alfred.skills.email_imap import EmailConfig, ImapClient
from alfred.skills.email_imap.monitor import EmailMonitor, MonitorConfig
from alfred.skills.email_imap.triage import TriageRules, Triager
from smoke_test import preflight


def hhmm(text: str) -> time:
    hour, minute = text.split(":")
    return time(int(hour), int(minute))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", default="qwen3.8:27b")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--digest-now", action="store_true")
    ap.add_argument("--interval", type=float, default=300.0, help="seconds between polls")
    ap.add_argument("--backlog", type=int, default=0)
    ap.add_argument("--vip", action="append", default=[])
    ap.add_argument("--ignore", action="append", default=[])
    ap.add_argument("--quiet", help="e.g. 22:00-07:00")
    ap.add_argument("--digest-at", default="08:00")
    ap.add_argument("--context", default="")
    ap.add_argument("--state", default="alfred_state.json")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        email_config = EmailConfig.from_env()
    except ValueError as exc:
        raise SystemExit(f"{exc}\nSee check_email.py for how to set them.")
    preflight(args.base_url, args.model)

    quiet = None
    if args.quiet:
        start, end = args.quiet.split("-")
        quiet = (hhmm(start), hhmm(end))
    config = MonitorConfig(
        poll_interval_s=args.interval, first_run_backlog=args.backlog,
        quiet_hours=quiet, digest_at=hhmm(args.digest_at),
    )
    if args.digest_now:
        config.digest_at, config.quiet_hours = time(0, 0), None

    monitor = EmailMonitor(
        ImapClient(email_config),
        Triager(
            OpenAICompatProvider(args.model, base_url=args.base_url),
            TriageRules(set(args.vip), set(args.ignore), args.context),
        ),
        CliChannel(), StateStore(args.state), config,
    )

    async def run() -> None:
        if args.once:
            done = await monitor.poll_once()
            await monitor.maybe_send_digest()
            print(f"\nTriaged {len(done)} new email(s).")
        else:
            print(f"Watching {monitor.client.mailbox_id} every {args.interval:.0f}s. Ctrl+C to stop.")
            await monitor.run_forever(asyncio.Event())

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
