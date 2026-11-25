"""
Offline demo: no model, no network. A scripted "model" asks to read your calendar
(allowed automatically) and then send an email (needs your approval in the terminal).

    python demo.py
"""
import asyncio

from alfred.channels.cli import CliChannel
from alfred.core.audit import InMemoryAuditLog
from alfred.core.interfaces import Autonomy, RiskTier, ToolResult
from alfred.core.orchestrator import Orchestrator, OrchestratorConfig
from alfred.core.policy import DefaultPolicyEngine
from alfred.core.registry import Registry
from alfred.testing import FakeProvider, FakeTool, say, use


async def main() -> None:
    calendar = FakeTool("calendar.read", RiskTier.READ_ONLY,
                        result=ToolResult(True, data="Thu 2pm: dentist"))
    email = FakeTool("email.send", RiskTier.EXTERNAL, domain="email",
                     properties={"to": {"type": "string"}, "body": {"type": "string"}},
                     required=["to", "body"])

    registry = Registry()
    registry.register(calendar)
    registry.register(email)

    provider = FakeProvider([
        use("calendar.read", "checking your schedule first"),
        use("email.send", "you asked me to tell Sam", to="sam@example.com",
            body="I can't make Thursday at 2pm."),
        say("All done. I checked your calendar and emailed Sam."),
    ])

    orch = Orchestrator(
        provider, CliChannel(), registry, DefaultPolicyEngine(), InMemoryAuditLog(),
        OrchestratorConfig(autonomy=Autonomy.ASK_EXTERNAL),
    )
    print(await orch.run_task("Tell Sam I can't make Thursday."))
    print("\nAudit log:")
    for e in reversed(orch.audit.recent()):
        print(f"  {e.call.tool:15} decision={e.decision.value:5} approved_by={e.approved_by}")
    print(f"\nemail.send actually ran {len(email.calls)} time(s)")


asyncio.run(main())
