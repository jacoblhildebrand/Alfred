"""Policy tests. Runs with either `pytest` or `python -m unittest`."""
import unittest

from alfred.core.interfaces import (
    Autonomy,
    Decision,
    PolicyContext,
    RiskTier,
    ToolCall,
    ToolSpec,
)
from alfred.core.policy import DefaultPolicyEngine

A, D, K = Decision.ALLOW, Decision.ASK, Decision.DENY
engine = DefaultPolicyEngine()


def spec(risk=RiskTier.READ_ONLY, domain="general", name="t.do"):
    return ToolSpec(name=name, description="", parameters={}, risk=risk, domain=domain)


def call(name="t.do", **args):
    return ToolCall(tool=name, arguments=args)


def ctx(autonomy=Autonomy.ASK_EXTERNAL, **kw):
    return PolicyContext(autonomy=autonomy, **kw)


class AutonomyMatrix(unittest.TestCase):
    def test_every_autonomy_and_risk_combination(self):
        R = RiskTier
        expected = {
            Autonomy.ASK_EVERYTHING: [D, D, D, D],
            Autonomy.ASK_EXTERNAL:   [A, A, D, D],
            Autonomy.ACT_THEN_TELL:  [A, A, A, D],
            Autonomy.FULL_AUTO:      [A, A, A, A],
        }
        for autonomy, row in expected.items():
            for risk, want in zip([R.READ_ONLY, R.REVERSIBLE, R.EXTERNAL, R.IRREVERSIBLE], row):
                with self.subTest(autonomy=autonomy.value, risk=risk.name):
                    got = engine.decide(call(), spec(risk), ctx(autonomy))
                    self.assertEqual(got, want)

    def test_unknown_autonomy_fails_closed(self):
        got = engine.decide(call(), spec(), PolicyContext(autonomy="yolo"))  # type: ignore[arg-type]
        self.assertEqual(got, D)


class HardLimits(unittest.TestCase):
    def test_blocked_target_denied_even_in_full_auto(self):
        c = ctx(Autonomy.FULL_AUTO, blocked_targets={"boss@corp.com"})
        got = engine.decide(call(to="boss@corp.com"), spec(RiskTier.EXTERNAL), c)
        self.assertEqual(got, K)

    def test_blocked_match_is_case_insensitive_and_checks_lists(self):
        c = ctx(Autonomy.FULL_AUTO, blocked_targets={"Boss@Corp.com"})
        got = engine.decide(call(to="a@x.com", cc=["b@x.com", " BOSS@corp.com "]), spec(RiskTier.EXTERNAL), c)
        self.assertEqual(got, K)

    def test_blocked_folder_covers_children_but_not_siblings(self):
        c = ctx(Autonomy.FULL_AUTO, blocked_targets={"/home/u/secret"})
        s = spec(RiskTier.REVERSIBLE)
        self.assertEqual(engine.decide(call(path="/home/u/secret/keys.txt"), s, c), K)
        self.assertEqual(engine.decide(call(path="/home/u/secret"), s, c), K)
        self.assertEqual(engine.decide(call(path="/home/u/secret-notes/a.txt"), s, c), A)

    def test_empty_blocked_entry_is_ignored(self):
        c = ctx(Autonomy.FULL_AUTO, blocked_targets={""})
        self.assertEqual(engine.decide(call(path="/anything"), spec(RiskTier.REVERSIBLE), c), A)

    def test_spend_over_cap_denied(self):
        c = ctx(Autonomy.FULL_AUTO, spend_cap_remaining=50.0)
        self.assertEqual(engine.decide(call(amount=75), spec(RiskTier.IRREVERSIBLE), c), K)

    def test_spend_within_cap_allowed_in_full_auto(self):
        c = ctx(Autonomy.FULL_AUTO, spend_cap_remaining=50.0)
        self.assertEqual(engine.decide(call(amount=20), spec(RiskTier.IRREVERSIBLE), c), A)

    def test_spend_within_cap_still_asks_when_autonomy_is_lower(self):
        c = ctx(Autonomy.ACT_THEN_TELL, spend_cap_remaining=50.0)
        self.assertEqual(engine.decide(call(amount=20), spec(RiskTier.IRREVERSIBLE), c), D)

    def test_bad_or_missing_amount_asks(self):
        c = ctx(Autonomy.FULL_AUTO, spend_cap_remaining=50.0)
        s = spec(RiskTier.IRREVERSIBLE)
        for args in ({}, {"amount": "20"}, {"amount": True}, {"amount": -5}, {"amount": None}):
            with self.subTest(args=args):
                self.assertEqual(engine.decide(call(**args), s, c), D)

    def test_amount_is_checked_on_lower_tiers_too(self):
        c = ctx(Autonomy.FULL_AUTO, spend_cap_remaining=10.0)
        self.assertEqual(engine.decide(call(amount=99), spec(RiskTier.EXTERNAL), c), K)

    def test_no_cap_means_amount_not_checked(self):
        c = ctx(Autonomy.FULL_AUTO, spend_cap_remaining=None)
        self.assertEqual(engine.decide(call(amount=10**6), spec(RiskTier.IRREVERSIBLE), c), A)


class UntrustedContent(unittest.TestCase):
    def test_untrusted_forces_ask_above_read_only_even_in_full_auto(self):
        c = ctx(Autonomy.FULL_AUTO, triggered_by_untrusted=True)
        for risk in (RiskTier.REVERSIBLE, RiskTier.EXTERNAL, RiskTier.IRREVERSIBLE):
            with self.subTest(risk=risk.name):
                self.assertEqual(engine.decide(call(), spec(risk), c), D)

    def test_untrusted_still_allows_read_only(self):
        c = ctx(Autonomy.FULL_AUTO, triggered_by_untrusted=True)
        self.assertEqual(engine.decide(call(), spec(RiskTier.READ_ONLY), c), A)

    def test_untrusted_does_not_downgrade_a_deny(self):
        c = ctx(Autonomy.FULL_AUTO, triggered_by_untrusted=True, blocked_targets={"x@y.com"})
        self.assertEqual(engine.decide(call(to="x@y.com"), spec(RiskTier.EXTERNAL), c), K)


class DomainOverrides(unittest.TestCase):
    def test_stricter_override_applies(self):
        c = ctx(Autonomy.FULL_AUTO, domain_overrides={"email": Autonomy.ASK_EVERYTHING})
        self.assertEqual(engine.decide(call(), spec(RiskTier.READ_ONLY, domain="email"), c), D)
        self.assertEqual(engine.decide(call(), spec(RiskTier.READ_ONLY, domain="calendar"), c), A)

    def test_looser_override_applies(self):
        c = ctx(Autonomy.ASK_EXTERNAL, domain_overrides={"calendar": Autonomy.ACT_THEN_TELL})
        self.assertEqual(engine.decide(call(), spec(RiskTier.EXTERNAL, domain="calendar"), c), A)
        self.assertEqual(engine.decide(call(), spec(RiskTier.EXTERNAL, domain="email"), c), D)

    def test_override_cannot_bypass_hard_limits_or_untrusted(self):
        c = ctx(
            Autonomy.ASK_EVERYTHING,
            domain_overrides={"email": Autonomy.FULL_AUTO},
            blocked_targets={"x@y.com"},
        )
        s = spec(RiskTier.EXTERNAL, domain="email")
        self.assertEqual(engine.decide(call(to="x@y.com"), s, c), K)
        c2 = ctx(
            Autonomy.ASK_EVERYTHING,
            domain_overrides={"email": Autonomy.FULL_AUTO},
            triggered_by_untrusted=True,
        )
        self.assertEqual(engine.decide(call(), s, c2), D)


class Sanity(unittest.TestCase):
    def test_call_must_match_spec(self):
        got = engine.decide(call(name="email.send"), spec(name="calendar.read"), ctx(Autonomy.FULL_AUTO))
        self.assertEqual(got, K)

    def test_evaluate_always_gives_a_reason(self):
        for c in (ctx(), ctx(Autonomy.FULL_AUTO, blocked_targets={"a"}), ctx(triggered_by_untrusted=True)):
            _, reason = engine.evaluate(call(to="a"), spec(RiskTier.EXTERNAL), c)
            self.assertTrue(reason)


if __name__ == "__main__":
    unittest.main()
