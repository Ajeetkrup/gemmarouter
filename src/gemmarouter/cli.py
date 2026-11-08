"""Command-line interface for gemmarouter (`gemmarouter route` and `gemmarouter playground`)."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from .router import DEFAULT_REPO_ID, load


def _parse_option_flags(raw_options: list[str]) -> dict[str, str]:
    """Parse `-o 'key:description'` CLI flags into an options dictionary."""
    parsed: dict[str, str] = {}
    for item in raw_options:
        if ":" in item:
            k, v = item.split(":", 1)
            parsed[k.strip()] = v.strip()
        else:
            parsed[item.strip()] = item.strip()
    return parsed


def _run_route_cmd(args: argparse.Namespace) -> int:
    """Execute the `gemmarouter route` CLI subcommand."""
    options = _parse_option_flags(args.option)
    router = load(
        model_id=args.model,
        device=args.device,
        threshold=args.threshold,
        margin_threshold=args.margin_threshold,
    )
    res = router.route(
        args.query,
        options=options,
        threshold=args.threshold,
        margin_threshold=args.margin_threshold,
    )
    if args.json:
        print(json.dumps(res.to_dict(), indent=2))
        return 0

    if res.needs_clarification:
        status = f"CLARIFICATION NEEDED -> {res.clarify_options}"
    elif res.abstain:
        status = "ABSTAIN (None of the options fit)"
    else:
        status = f"ROUTED -> {res.decision}"
    print("=" * 68)
    print(f"Query          : {args.query}")
    print(f"Decision       : {status}")
    if res.clarification_prompt:
        print(f"Clarification  : {res.clarification_prompt}")
    print(f"Top Candidate  : {res.top_option} ({res.confidence * 100:.2f}%, margin={res.margin:.4f})")
    print(f"P(null)        : {res.null_probability:.4f} (threshold={res.threshold:.2f})")
    print(f"Latency        : {res.latency_ms:.2f} ms (Head: {res.head_latency_ms:.3f} ms)")
    print("=" * 68)
    return 0


def _run_playground_cmd(args: argparse.Namespace) -> int:
    """Execute the `gemmarouter playground` CLI subcommand."""
    from .playground import launch_playground

    launch_playground(
        host=args.host,
        port=args.port,
        our_repo=args.model,
        open_browser=not args.no_browser,
        preload=not args.no_preload,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level `gemmarouter` CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="gemmarouter",
        description="Set-Equivariant Decision & Tool Router on EmbeddingGemma 2",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_route = sub.add_parser("route", help="Route a query across candidate options from the terminal")
    p_route.add_argument("query", help="Incoming user query or agent prompt")
    p_route.add_argument(
        "-o",
        "--option",
        action="append",
        required=True,
        help="Candidate option in 'key:description' format (pass at least two -o flags)",
    )
    p_route.add_argument("--threshold", type=float, default=0.58, help="Null Gate abstention threshold (default: 0.58)")
    p_route.add_argument("--margin-threshold", type=float, default=0.40, help="Top-2 margin threshold for twin-tool clarification (default: 0.40)")
    p_route.add_argument("--model", default=DEFAULT_REPO_ID, help="Hugging Face model ID or local path")
    p_route.add_argument("--device", default=None, help="PyTorch device ('cpu' or 'cuda')")
    p_route.add_argument("--json", action="store_true", help="Output raw JSON")

    p_play = sub.add_parser("playground", help="Launch the interactive 3-column Live Decision Playground web UI")
    p_play.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    p_play.add_argument("--port", type=int, default=7860, help="Bind port (default: 7860)")
    p_play.add_argument("--model", default=DEFAULT_REPO_ID, help="Hugging Face model ID or local path")
    p_play.add_argument("--no-browser", action="store_true", help="Do not auto-open browser tab")
    p_play.add_argument("--no-preload", action="store_true", help="Lazy-load models on first request")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entrypoint for the `gemmarouter` console script."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "route":
        return _run_route_cmd(args)
    if args.command == "playground":
        return _run_playground_cmd(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
