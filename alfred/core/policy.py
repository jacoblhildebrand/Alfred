"""
Alfred policy engine.

Every tool call passes through PolicyEngine.decide() before it runs. This is
the safety boundary of the project, so it is deliberately small, pure (no I/O,
no clock, no model calls) and fails closed: anything unexpected becomes ASK or
DENY, never ALLOW.

Order of checks (earlier checks cannot be overridden by later ones):
  1. Sanity       - the call must match the spec it claims to be
  2. Hard limits  - blocked targets, spend cap            -> DENY
  3. Untrusted    - call influenced by outside content    -> at least ASK
  4. Autonomy     - risk tier vs. the user's ceiling      -> ALLOW or ASK
"""
from __future__ import annotations

from numbers import Real

from alfred.core.interfaces import (
    Autonomy,
    Decision,
    PolicyContext,
    RiskTier,
    ToolCall,
    ToolSpec,
)

# Highest risk tier that may run WITHOUT asking, per autonomy preset.
# None means "ask for everything", including read-only actions.
_CEILING: dict[Autonomy, RiskTier | None] = {
    Autonomy.ASK_EVERYTHING: None,
    Autonomy.ASK_EXTERNAL: RiskTier.REVERSIBLE,
    Autonomy.ACT_THEN_TELL: RiskTier.EXTERNAL,
    Autonomy.FULL_AUTO: RiskTier.IRREVERSIBLE,
}

# Argument names that identify who or what an action touches. Skills should
# use these names so the blocked list works on them automatically.
_TARGET_KEYS = ("to", "cc", "bcc", "recipient", "recipients", "path", "target")


def _targets(call: ToolCall) -> list[str]:
    found: list[str] = []
    for key in _TARGET_KEYS:
        value = call.arguments.get(key)
        if isinstance(value, str):
            found.append(value)
        elif isinstance(value, (list, tuple, set)):
            found.extend(v for v in value if isinstance(v, str))
    return found


def _is_blocked(target: str, blocked: set[str]) -> bool:
    """Exact match, or the target sits inside a blocked folder.

    NOTE: this is string matching only. Skills that take file paths must
    resolve them (realpath) and skills that take email addresses must strip
    display names ("Sam <sam@x.com>") before building the ToolCall.
    """
    t = target.strip().lower()
    for entry in blocked:
        b = entry.strip().lower()
        if not b:
            continue
        if t == b or t.startswith(b.rstrip("/") + "/"):
            return True
    return False


class DefaultPolicyEngine:
    def evaluate(
        self, call: ToolCall, spec: ToolSpec, ctx: PolicyContext
    ) -> tuple[Decision, str]:
        """Return the decision and a human-readable reason (for approvals/audit)."""

        # 1. Sanity
        if call.tool != spec.name:
            return Decision.DENY, f"call to {call.tool!r} does not match spec {spec.name!r}"

        # 2. Hard limits: nothing below this point can override them
        blocked = {b for b in ctx.blocked_targets}
        for target in _targets(call):
            if _is_blocked(target, blocked):
                return Decision.DENY, f"{target!r} is on the blocked list"

        if ctx.spend_cap_remaining is not None:
            has_amount = "amount" in call.arguments
            if has_amount or spec.risk is RiskTier.IRREVERSIBLE:
                amount = call.arguments.get("amount")
                valid = (
                    isinstance(amount, Real)
                    and not isinstance(amount, bool)
                    and amount >= 0
                )
                if not valid:
                    return Decision.ASK, "spending limit is set but the amount is missing or invalid"
                if amount > ctx.spend_cap_remaining:
                    return Decision.DENY, f"amount {amount} exceeds remaining budget {ctx.spend_cap_remaining}"

        # 3. Prompt-injection guard: outside content may never trigger a
        #    state-changing action on its own, whatever the autonomy setting.
        if ctx.triggered_by_untrusted and spec.risk >= RiskTier.REVERSIBLE:
            return Decision.ASK, "action was influenced by untrusted content (email/web)"

        # 4. Autonomy. A per-domain override replaces the global setting, but
        #    only for this step; it cannot bypass anything above.
        autonomy = ctx.domain_overrides.get(spec.domain, ctx.autonomy)
        ceiling = _CEILING.get(autonomy)  # unknown value -> None -> ask
        if ceiling is None:
            return Decision.ASK, f"autonomy {autonomy!s} requires approval for every action"
        if spec.risk <= ceiling:
            return Decision.ALLOW, f"{spec.risk.name} is within the {autonomy.value} ceiling"
        return Decision.ASK, f"{spec.risk.name} exceeds the {autonomy.value} ceiling"

    def decide(self, call: ToolCall, spec: ToolSpec, ctx: PolicyContext) -> Decision:
        return self.evaluate(call, spec, ctx)[0]
