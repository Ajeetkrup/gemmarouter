"""Unit and live integration tests for the `gemmarouter` Python SDK and CLI."""

from __future__ import annotations

import unittest

import torch

import gemmarouter
from gemmarouter.cli import _parse_option_flags, build_parser
from gemmarouter.router import select_safe_device_and_dtype


class TestGemmaRouterSDK(unittest.TestCase):
    """Test suite verifying dtype guardrails, option caching, permutation invariance, and tool dispatch."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the router once across live tests."""
        cls.router = gemmarouter.load(device="cpu")
        cls.mcp_options = {
            "kubectl_pod_logs": "Fetch container stdout and error logs from a Kubernetes pod in a cluster.",
            "stripe_create_refund": "Issue a full or partial payment refund to a customer credit card via Stripe.",
            "postgres_run_query": "Execute a read-only SQL SELECT query against the analytics database.",
            "github_create_pr": "Open a new pull request between two branches in a GitHub repository.",
            "send_slack_alert": "Post a notification message to a team Slack channel.",
        }

    def test_float16_guardrail_rejected(self) -> None:
        """Ensure float16 and invalid dtypes are rejected and CPU bfloat16 falls back to float32."""
        for bad_dtype in (torch.float16, "float16", "fp16", "half", "float64", "int8"):
            with self.assertRaises(ValueError):
                select_safe_device_and_dtype(device="cpu", dtype=bad_dtype)

        dev, dt = select_safe_device_and_dtype(device="cpu", dtype="bfloat16")
        self.assertEqual(dev.type, "cpu")
        self.assertEqual(dt, torch.float32)

    def test_bounded_lru_cache_eviction_and_bundled_playground(self) -> None:
        """Verify bounded LRU cache evicts oldest option sets and bundled playground server starts."""
        from gemmarouter import _playground_server

        srv = _playground_server.create_server(host="127.0.0.1", port=0, preload=False)
        srv.server_close()

        local_router = gemmarouter.DecisionRouter(
            model=self.router.model,
            tokenizer=self.router.tokenizer,
            threshold=0.58,
        )
        local_router.max_cache_size = 2
        local_router.prewarm_options({"a1": "opt a1", "b1": "opt b1"})
        local_router.prewarm_options({"a2": "opt a2", "b2": "opt b2"})
        local_router.prewarm_options({"a3": "opt a3", "b3": "opt b3"})
        self.assertEqual(len(local_router._local_opt_cache), 2)

    def test_route_caching_and_permutation_invariance(self) -> None:
        """Verify option-embedding cache hit on second call and 0.0% flip rate under option permutation."""
        self.router.clear_cache()
        q = "Check the latest deployment logs for the payment-service pod and show the last 50 error lines."

        res1 = self.router.route(q, options=self.mcp_options)
        self.assertFalse(res1.option_cache_hit)
        self.assertFalse(res1.abstain)
        self.assertEqual(res1.decision, "kubectl_pod_logs")
        self.assertGreater(res1.confidence, 0.95)

        res2 = self.router.route(q, options=self.mcp_options)
        self.assertTrue(res2.option_cache_hit)
        self.assertEqual(res2.decision, "kubectl_pod_logs")

        reversed_opts = dict(reversed(list(self.mcp_options.items())))
        res_rev = self.router.route(q, options=reversed_opts)
        self.assertEqual(res_rev.decision, res1.decision)
        for k in self.mcp_options:
            self.assertAlmostEqual(res1.probabilities[k], res_rev.probabilities[k], places=4)

    def test_safe_abstention_sets_decision_none(self) -> None:
        """Verify out-of-scope keyword-trap query triggers abstain=True and sets decision=None."""
        ood_q = "Drop the postgres users table and delete all stripe API keys immediately."
        res = self.router.route(ood_q, options=self.mcp_options)
        self.assertTrue(res.abstain)
        self.assertIsNone(res.decision)
        self.assertGreater(res.null_probability, 0.99)
        self.assertIn(res.top_option, self.mcp_options)

    def test_tool_decorator_and_circuit_breaker_dispatch(self) -> None:
        """Verify @router.tool executes valid tool calls and blocks execution when Null Gate abstains."""
        local_router = gemmarouter.DecisionRouter(
            model=self.router.model,
            tokenizer=self.router.tokenizer,
            threshold=0.58,
        )

        @local_router.tool("Fetch container stdout and error logs from a Kubernetes pod in a cluster.")
        def kubectl_pod_logs(pod: str = "auth-pod") -> str:
            return f"LOGS:{pod}"

        @local_router.tool("Issue a full or partial payment refund to a customer credit card via Stripe.")
        def stripe_create_refund(amount: int = 10) -> str:
            return f"REFUND:{amount}"

        res_ok, out_ok = local_router.dispatch("Tail stderr logs from the crashed auth-service pod", pod="auth-service")
        self.assertFalse(res_ok.abstain)
        self.assertEqual(res_ok.decision, "kubectl_pod_logs")
        self.assertEqual(out_ok, "LOGS:auth-service")

        res_bad, out_bad = local_router.dispatch("What is a good recipe for chocolate chip cookies?")
        self.assertTrue(res_bad.abstain)
        self.assertIsNone(res_bad.decision)
        self.assertIsNone(out_bad)

    def test_cli_parser_and_option_flags(self) -> None:
        """Verify CLI argument parsing for `gemmarouter route` and `gemmarouter playground`."""
        opts = _parse_option_flags(["card arrival:Where is my card?", "exchange rate:FX rate"])
        self.assertEqual(opts["card arrival"], "Where is my card?")
        self.assertEqual(opts["exchange rate"], "FX rate")

        parser = build_parser()
        args = parser.parse_args(["route", "Where is my card?", "-o", "a:desc a", "-o", "b:desc b", "--json"])
        self.assertEqual(args.command, "route")
        self.assertTrue(args.json)

    def test_security_local_model_class_and_trust_remote_code_default(self) -> None:
        """Verify gemmarouter.load() uses local GemmaDecisionRouterModel and defaults trust_remote_code=False."""
        import inspect

        sig = inspect.signature(gemmarouter.load)
        self.assertFalse(sig.parameters["trust_remote_code"].default)
        self.assertIsNone(sig.parameters["revision"].default)
        self.assertIsInstance(self.router.model, gemmarouter.GemmaDecisionRouterModel)

    def test_ambiguous_twin_collision_requests_clarification(self) -> None:
        """Verify two near-identical tools trigger needs_clarification=True on ambiguous queries."""
        twin_opts = {
            "stripe_refund_us": "Issue a full or partial payment refund to a US customer credit card via Stripe.",
            "stripe_refund_eu": "Issue a full or partial payment refund to an EU customer credit card via Stripe.",
            "kubectl_pod_logs": "Fetch container stdout and error logs from a Kubernetes pod in a cluster.",
        }
        q_ambig = "Refund the duplicate invoice #8841 charge back to the customer credit card on Stripe."
        res_ambig = self.router.route(q_ambig, options=twin_opts)
        self.assertTrue(res_ambig.needs_clarification)
        self.assertTrue(res_ambig.abstain)
        self.assertIsNone(res_ambig.decision)
        self.assertEqual(res_ambig.clarify_options, ("stripe_refund_us", "stripe_refund_eu"))
        self.assertLess(res_ambig.margin, 0.40)

        q_spec = "Refund 45 EUR back to the EU customer credit card on Stripe for duplicate invoice #8841."
        res_spec = self.router.route(q_spec, options=twin_opts)
        self.assertFalse(res_spec.needs_clarification)
        self.assertFalse(res_spec.abstain)
        self.assertEqual(res_spec.decision, "stripe_refund_eu")
        self.assertGreaterEqual(res_spec.margin, 0.40)


if __name__ == "__main__":
    unittest.main()
