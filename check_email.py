"""
Try the read-only email skill against a real mailbox. No model involved.

    py check_email.py          # list the 5 newest emails
    py check_email.py 123      # show email uid 123 (body truncated)

Credentials come from environment variables (PowerShell, current window only):

    $env:ALFRED_IMAP_HOST     = "imap.gmail.com"
    $env:ALFRED_IMAP_USER     = "you@gmail.com"
    $env:ALFRED_IMAP_PASSWORD = "your-app-password"

Use an app password, not your normal one. Work and school accounts often block
password logins for IMAP; if yours does, start with a personal account.
"""
import imaplib
import json
import sys

from alfred.skills.email_imap import EmailConfig, ImapClient, MailError


def main() -> None:
    try:
        config = EmailConfig.from_env()
    except ValueError as exc:
        raise SystemExit(f"{exc}\n\n{__doc__}")
    client = ImapClient(config)
    try:
        if len(sys.argv) > 1:
            print(json.dumps(client.read(sys.argv[1]), indent=2, ensure_ascii=False))
        else:
            for m in client.list_recent(5):
                mark = "*" if m["unread"] else " "
                print(f"{mark} uid={m['uid']:<6} {m['date'][:25]:<26} {m['from'][:30]:<31} {m['subject'][:50]}")
            print("\n(* = unread. Nothing was marked as read, changed or sent.)")
    except imaplib.IMAP4.error as exc:
        raise SystemExit(f"Login or mailbox error: {exc}\nCheck the username, app password and IMAP access.")
    except MailError as exc:
        raise SystemExit(f"Mail error: {exc}")
    except OSError as exc:
        raise SystemExit(f"Could not connect to {config.host}:{config.port}: {exc}")


main()
