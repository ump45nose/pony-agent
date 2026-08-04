"""`pony` command."""

from __future__ import annotations

import os
import sys

from pony_cli.bootstrap import bootstrap_legacy_environment


def _capture_agent_core(argv: list[str]) -> None:
    selected = "pony"
    for index, arg in enumerate(argv):
        if arg.startswith("--agent-core="):
            selected = arg.split("=", 1)[1]
            break
        if arg == "--agent-core" and index + 1 < len(argv):
            selected = argv[index + 1]
            break
    os.environ["PONY_AGENT_CORE"] = selected


def main() -> None:
    _capture_agent_core(sys.argv[1:])
    bootstrap_legacy_environment()
    if sys.argv[1:] in (["--help"], ["-h"]):
        from hermes_cli._parser import build_top_level_parser

        parser, _, _ = build_top_level_parser()
        parser.print_help()
        return
    if sys.argv[1:] in (["--version"], ["-V"]):
        from pony_agent import __version__

        print(f"Pony Agent v{__version__}")
        return
    from hermes_cli.main import main as legacy_main

    legacy_main()
